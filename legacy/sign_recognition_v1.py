"""
sign_recognition.py
────────────────────
SignSpeak core recognition engine.
Loads trained 1D CNN model, processes webcam frames,
outputs recognized signs as both text overlay and audio.
"""

import cv2
import numpy as np
import tensorflow as tf
import threading
import queue
import time
import json
import os
from collections import Counter, deque
from typing import Optional
from hand_tracking import HandTracker
from gtts import gTTS
import tempfile


# ── TTS Engine (non-blocking) ──────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self._queue = queue.Queue(maxsize=3)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        print("TTS engine: gTTS")

    def _worker(self):
        try:
            import pygame
            pygame.mixer.init()
        except Exception as e:
            print(f"TTS init error: {e}")
            return

        # Fixed file avoids Windows file-lock race on temp files
        audio_file = os.path.join(tempfile.gettempdir(), "signspeak_tts.mp3")

        while True:
            try:
                text = self._queue.get(timeout=1)
                tts = gTTS(text=text, lang='en', tld='com.au', slow=False)
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                tts.save(audio_file)
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.unload()
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")

    def speak(self, text: str):
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            pass

    def format_for_speech(self, sign: str) -> str:
        if len(sign) == 1 and sign.isalpha():
            return f"The letter {sign}"
        elif sign.isdigit():
            return f"The number {sign}"
        else:
            return sign


# ── Sign Recognizer ────────────────────────────────────────────────────────────

class SignRecognizer:
    """
    Real-time ASL sign recognizer.

    Pipeline:
        webcam frame
            → HandTracker (MediaPipe landmarks)
            → normalize features (63-dim)
            → 1D CNN model
            → stability filter (prevents flickering)
            → TTS + text overlay
    """

    def __init__(self, model_path: str, label_map_path: str):
        """
        Args:
            model_path:     path to trained .keras or .h5 model
            label_map_path: path to label_map.json
        """
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Train it first using colab/03_train_model.py"
            )
        self.model = tf.keras.models.load_model(model_path)
        print(f"Model loaded: {model_path}")
        print(f"Input shape: {self.model.input_shape}")

        # Load label map
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"Label map not found: {label_map_path}")
        with open(label_map_path) as f:
            label_map = json.load(f)
        # Invert: {index: sign_name}
        self.idx_to_sign = {int(v): k for k, v in label_map.items()}
        print(f"Labels loaded: {len(self.idx_to_sign)} classes")

        # Components
        _task = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(_task):
            _task = 'hand_landmarker.task'
        self.hand_tracker = HandTracker(
            model_path=_task,
            detection_confidence=0.6,
            tracking_confidence=0.5
        )
        self.tts = TTSEngine()

        # Recognition state
        self.current_sign: str = "No Sign"
        self.current_confidence: float = 0.0
        self.last_spoken_sign: Optional[str] = None

        # Stability buffer — prevents flickering between frames
        # Only announce a sign after it appears consistently
        self._prediction_buffer = deque(maxlen=15)
        self.STABLE_THRESHOLD = 8        # sign must appear 8/15 frames
        self.CONFIDENCE_THRESHOLD = 0.65

        # Cooldown — don't repeat same sign too quickly
        self._last_sign_time = 0
        self.SIGN_COOLDOWN = 1.5         # seconds

        # FPS tracking
        self._fps_buffer = deque(maxlen=30)
        self._last_frame_time = time.time()

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        """
        Run model inference on 63-dim feature vector.

        Returns:
            (predicted_class_index, confidence)
        """
        if features is None or features.shape[0] != 63:
            return -1, 0.0

        x = features.reshape(1, 63)

        try:
            probs = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            return idx, conf
        except Exception as e:
            print(f"Prediction error: {e}")
            return -1, 0.0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline: detect → predict → stabilize → output.

        Args:
            frame: BGR webcam frame

        Returns:
            annotated frame
        """
        # Track FPS
        now = time.time()
        self._fps_buffer.append(1.0 / max(now - self._last_frame_time, 1e-6))
        self._last_frame_time = now
        fps = np.mean(self._fps_buffer)

        # Hand detection + feature extraction
        frame, features_list = self.hand_tracker.detect_hands(frame)

        if features_list:
            features = features_list[0]
            idx, confidence = self.predict(features)

            if idx != -1 and confidence >= self.CONFIDENCE_THRESHOLD:
                self._prediction_buffer.append(idx)

                # Check for stable recognition
                if len(self._prediction_buffer) >= self.STABLE_THRESHOLD:
                    most_common_idx, count = Counter(self._prediction_buffer).most_common(1)[0]

                    if count >= self.STABLE_THRESHOLD:
                        sign_name = self.idx_to_sign.get(most_common_idx, "Unknown")
                        self.current_sign = sign_name
                        self.current_confidence = confidence

                        # Speak if it's a new sign and cooldown has passed
                        if (sign_name != self.last_spoken_sign or
                                now - self._last_sign_time > self.SIGN_COOLDOWN * 3):
                            if now - self._last_sign_time > self.SIGN_COOLDOWN:
                                speech_text = self.tts.format_for_speech(sign_name)
                                self.tts.speak(speech_text)
                                self.last_spoken_sign = sign_name
                                self._last_sign_time = now
            else:
                self._prediction_buffer.append(-1)
        else:
            # No hand — clear buffer gradually
            self._prediction_buffer.append(-1)
            if self._prediction_buffer.count(-1) > 10:
                self.current_sign = "No Sign"
                self.current_confidence = 0.0

        # Draw overlay
        self._draw_overlay(frame, fps)

        return frame

    def _draw_overlay(self, frame: np.ndarray, fps: float):
        """Draw recognition results and status overlay on frame."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # ── Recognized sign (large, centered) ──
        sign = self.current_sign
        conf = self.current_confidence

        # Color based on confidence
        if sign == "No Sign":
            text_color = (120, 120, 120)
        elif conf > 0.85:
            text_color = (80, 255, 80)   # bright green — high confidence
        elif conf > 0.70:
            text_color = (80, 200, 255)  # yellow-ish — medium confidence
        else:
            text_color = (80, 150, 255)  # orange — lower confidence

        # Large sign text
        font = cv2.FONT_HERSHEY_SIMPLEX
        sign_text = sign if sign == "No Sign" else sign.upper()
        (tw, th), _ = cv2.getTextSize(sign_text, font, 1.4, 3)
        tx = (w - tw) // 2
        cv2.putText(frame, sign_text, (tx, 65), font, 1.4, text_color, 3)

        # ── Confidence bar ──
        if sign != "No Sign" and conf > 0:
            bar_x, bar_y = 15, 78
            bar_w_total = w - 30
            bar_h = 10
            filled = int(bar_w_total * conf)

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_total, bar_y + bar_h),
                         (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                         text_color, -1)
            cv2.putText(frame, f'{conf*100:.0f}%', (bar_x + bar_w_total + 5, bar_y + 9),
                       font, 0.45, text_color, 1)

        # ── FPS (top right) ──
        cv2.putText(frame, f'FPS: {fps:.0f}', (w - 90, 25), font, 0.6, (150, 150, 150), 1)

        # ── Bottom instruction bar ──
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, h - 35), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "Press Q to quit  |  SignSpeak v2.0",
                   (10, h - 10), font, 0.5, (150, 150, 150), 1)

    def run(self, camera_index: int = 0):
        """
        Main loop — opens webcam and runs real-time recognition.

        Args:
            camera_index: webcam device index (usually 0)
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        print("\nSignSpeak running. Press Q to quit.")
        print("Hold a sign gesture clearly in view of the camera.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame read failed.")
                    break

                frame = cv2.flip(frame, 1)  # mirror — more natural
                output = self.process_frame(frame)

                cv2.imshow('SignSpeak', output)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()
            print("SignSpeak closed.")

    def release(self):
        self.hand_tracker.release()
