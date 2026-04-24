"""
sign_recognition.py
────────────────────
SignSpeak core recognition engine.
Loads trained 1D CNN model, processes webcam frames,
outputs recognized signs as both text overlay and audio.

Sentence Builder controls:
    SPACE       — add current sign immediately
    Hold 2s     — auto-add current sign to sentence
    BACKSPACE   — delete last character/word
    C           — clear entire sentence
    ENTER / S   — speak full sentence aloud
    Q           — quit
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
import ctypes
import sys

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

WINDOW_TITLE = 'SignSpeak'

# ── TTS Engine (non-blocking) ──────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self._queue = queue.Queue(maxsize=5)
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

        audio_file = os.path.join(tempfile.gettempdir(), "signspeak_tts.mp3")

        while True:
            try:
                text = self._queue.get(timeout=1)
                
                # Stop and fully release pygame before anything
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                time.sleep(0.2)  # give Windows time to release handle
                
                tts = gTTS(text=text, lang='en', tld='com.au', slow=False)
                tts.save(audio_file)
                
                pygame.mixer.music.load(audio_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                time.sleep(0.2)
                
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


# ── Sentence Builder ───────────────────────────────────────────────────────────

class SentenceBuilder:
    """
    Manages the sentence being built from recognised signs.

    Tokens are stored as a list so backspace can remove the last
    item whether it was a single letter or a whole word sign.
    The rendered sentence joins them with a space only between
    word-signs and letters that were not meant to chain together.
    For simplicity every token is kept separate; display joins them.
    """

    # Word-level signs (multi-char labels from label_map)
    WORD_SIGNS = {
        "Hello", "ThankYou", "Yes", "No",
        "Please", "Sorry", "Good", "Bad"
    }

    # How long a sign must be held to auto-add (seconds)
    AUTO_ADD_HOLD = 3.0

    # After adding a sign, ignore the same sign for this many seconds
    # (prevents double-adding while still holding)
    POST_ADD_LOCKOUT = 1.5

    def __init__(self):
        self.tokens: list[str] = []          # each recognised sign token
        self._hold_sign: Optional[str] = None
        self._hold_start: float = 0.0
        self._last_added_time: float = 0.0
        self._last_added_sign: Optional[str] = None
        self._auto_speak_timer: float = 0.0  # time of last sentence change
        self.AUTO_SPEAK_PAUSE = 4.0           # speak sentence after 4s no new tokens

    @property
    def sentence(self) -> str:
        """Render tokens into a display/speech string."""
        parts = []
        i = 0
        while i < len(self.tokens):
            tok = self.tokens[i]
            if tok in self.WORD_SIGNS:
                # Flush pending letters first
                parts.append(tok)
                i += 1
            else:
                # Chain consecutive single letters into a word
                word = ""
                while i < len(self.tokens) and self.tokens[i] not in self.WORD_SIGNS:
                    word += self.tokens[i]
                    i += 1
                if word:
                    parts.append(word)
        return " ".join(parts)

    def add(self, sign: str) -> bool:
        """
        Manually add a sign (e.g. SPACE key).
        Returns True if something was actually added.
        """
        if not sign or sign == "No Sign":
            return False
        now = time.time()
        # Prevent adding same sign twice in quick succession
        if sign == self._last_added_sign and now - self._last_added_time < self.POST_ADD_LOCKOUT:
            return False
        self.tokens.append(sign)
        self._last_added_sign = sign
        self._last_added_time = now
        self._auto_speak_timer = now
        # Reset hold tracking so same sign doesn't also auto-add
        self._hold_sign = None
        return True

    def update_hold(self, sign: str, now: float) -> Optional[str]:
        """
        Call every frame with the current stable sign.
        Returns the sign name if it was auto-added this frame, else None.
        """
        if not sign or sign == "No Sign":
            self._hold_sign = None
            self._hold_start = 0.0
            return None

        # Different sign — restart hold timer
        if sign != self._hold_sign:
            self._hold_sign = sign
            self._hold_start = now
            return None

        # Same sign being held — check duration
        held_for = now - self._hold_start
        if held_for >= self.AUTO_ADD_HOLD:
            # Check lockout (prevent re-adding immediately after)
            if sign == self._last_added_sign and now - self._last_added_time < self.POST_ADD_LOCKOUT:
                return None
            self.tokens.append(sign)
            self._last_added_sign = sign
            self._last_added_time = now
            self._auto_speak_timer = now
            # Reset so user must release and re-hold to add again
            self._hold_sign = None
            self._hold_start = 0.0
            return sign

        return None

    def hold_progress(self, now: float) -> float:
        """
        Returns 0.0–1.0 progress toward auto-add hold threshold.
        0.0 means no active hold.
        """
        if not self._hold_sign or self._hold_sign == "No Sign":
            return 0.0
        held = now - self._hold_start
        return min(held / self.AUTO_ADD_HOLD, 1.0)

    def backspace(self):
        """Remove the last token."""
        if self.tokens:
            self.tokens.pop()
            self._auto_speak_timer = time.time()

    def clear(self):
        """Clear all tokens."""
        self.tokens.clear()
        self._hold_sign = None
        self._auto_speak_timer = time.time()

    def check_auto_speak(self, now: float) -> Optional[str]:
        """
        Returns sentence text if auto-speak should trigger, else None.
        Auto-speak fires when sentence is non-empty and nothing was
        added for AUTO_SPEAK_PAUSE seconds.
        """
        if not self.tokens:
            return None
        if now - self._auto_speak_timer >= self.AUTO_SPEAK_PAUSE:
            self._auto_speak_timer = now  # reset so it doesn't spam
            return self.sentence
        return None

# window icon 

def set_window_icon(window_title, icon_path):

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('SignSpeak.v2')
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd:
        # Large icon for taskbar (32x32)
        icon_large = ctypes.windll.user32.LoadImageW(
            None, icon_path, 1, 16, 16, 0x10 | 0x20
        )
        # Small icon for titlebar (16x16)
        icon_small = ctypes.windll.user32.LoadImageW(
            None, icon_path, 1, 16, 16, 0x10
        )
        ctypes.windll.user32.SendMessageW(hwnd, 0x80, 1, icon_large)
        ctypes.windll.user32.SendMessageW(hwnd, 0x80, 0, icon_small)



# ── Sign Recognizer ────────────────────────────────────────────────────────────

class SignRecognizer:
    """
    Real-time ASL sign recognizer with sentence builder.

    Pipeline:
        webcam frame
            → HandTracker (MediaPipe landmarks)
            → normalize features (63-dim)
            → 1D CNN model
            → stability filter (prevents flickering)
            → SentenceBuilder (hold / SPACE to commit)
            → TTS + text overlay
    """

    # ── layout constants ───────────────────────────────────────────────────────
    TOP_BAR_H    = 100   # recognition + confidence bar
    SENTENCE_H   = 90    # sentence builder panel
    HINT_H       = 30    # keyboard hints bar

    def __init__(self, model_path: str, label_map_path: str):
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
        self.idx_to_sign = {int(v): k for k, v in label_map.items()}
        print(f"Labels loaded: {len(self.idx_to_sign)} classes")

        # Components - change
        self.hand_tracker = HandTracker(
            model_path=resource_path('hand_landmarker.task'),
            detection_confidence=0.6,
            tracking_confidence=0.5
        )
        self.tts = TTSEngine()
        self.sentence_builder = SentenceBuilder()

        # Recognition state
        self.current_sign: str = "No Sign"
        self.current_confidence: float = 0.0
        self.last_spoken_sign: Optional[str] = None

        # Stability buffer
        self._prediction_buffer = deque(maxlen=15)
        self.STABLE_THRESHOLD = 8
        self.CONFIDENCE_THRESHOLD = 0.65

        # Per-sign TTS cooldown (live announcement only)
        self._last_sign_time = 0.0
        self.SIGN_COOLDOWN = 1.5

        # FPS tracking
        self._fps_buffer = deque(maxlen=30)
        self._last_frame_time = time.time()

        # Flash feedback when a token is added
        self._flash_until: float = 0.0
        self._flash_text: str = ""

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> tuple[int, float]:
        if features is None or features.shape[0] != 63:
            return -1, 0.0
        x = features.reshape(1, 63)
        try:
            probs = self.model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            return idx, float(probs[idx])
        except Exception as e:
            print(f"Prediction error: {e}")
            return -1, 0.0

    # ── main frame pipeline ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, key: int) -> np.ndarray:
        """
        Full pipeline: detect → predict → stabilize → sentence → draw.

        Args:
            frame: BGR webcam frame (already mirrored)
            key:   cv2.waitKey result for this frame
        Returns:
            annotated frame
        """
        now = time.time()

        # FPS
        self._fps_buffer.append(1.0 / max(now - self._last_frame_time, 1e-6))
        self._last_frame_time = now
        fps = float(np.mean(self._fps_buffer))

        # ── Hand detection ────────────────────────────────────────────────────
        frame, features_list = self.hand_tracker.detect_hands(frame)

        if features_list:
            idx, confidence = self.predict(features_list[0])

            if idx != -1 and confidence >= self.CONFIDENCE_THRESHOLD:
                self._prediction_buffer.append(idx)

                if len(self._prediction_buffer) >= self.STABLE_THRESHOLD:
                    top_idx, count = Counter(self._prediction_buffer).most_common(1)[0]
                    if count >= self.STABLE_THRESHOLD:
                        sign_name = self.idx_to_sign.get(top_idx, "No Sign")
                        self.current_sign = sign_name
                        self.current_confidence = confidence

                        # Live TTS for current sign
                        if (sign_name != self.last_spoken_sign or
                                now - self._last_sign_time > self.SIGN_COOLDOWN * 3):
                            if now - self._last_sign_time > self.SIGN_COOLDOWN:
                                self.tts.speak(self.tts.format_for_speech(sign_name))
                                self.last_spoken_sign = sign_name
                                self._last_sign_time = now
            else:
                self._prediction_buffer.append(-1)
        else:
            self._prediction_buffer.append(-1)
            if self._prediction_buffer.count(-1) > 10:
                self.current_sign = "No Sign"
                self.current_confidence = 0.0

        # ── Sentence builder — auto-hold ──────────────────────────────────────
        auto_added = self.sentence_builder.update_hold(self.current_sign, now)
        if auto_added:
            self._flash(f"Added: {auto_added}")

        # ── Keyboard controls ─────────────────────────────────────────────────
        if key != -1:
            self._handle_key(key, now)

        # ── Auto-speak sentence on pause ──────────────────────────────────────
        '''auto_sentence = self.sentence_builder.check_auto_speak(now)
        if auto_sentence:
            self.tts.speak(auto_sentence)'''

        # ── Draw everything ───────────────────────────────────────────────────
        self._draw_overlay(frame, fps, now)
        return frame

    def _handle_key(self, key: int, now: float):
        sign = self.current_sign

        if key == ord(' '):                       # SPACE — add now
            added = self.sentence_builder.add(sign)
            if added:
                self._flash(f"Added: {sign}")

        elif key == 8 or key == 127:              # BACKSPACE
            self.sentence_builder.backspace()
            self._flash("Deleted last")

        elif key == ord('c') or key == ord('C'): # C — clear
            self.sentence_builder.clear()
            self._flash("Cleared")

        elif key == 13 or key == ord('s') or key == ord('S'):  # ENTER / S — speak
            sentence = self.sentence_builder.sentence
            if sentence:
                self.tts.speak(sentence)
                self._flash("Speaking…")
            else:
                self._flash("Nothing to speak")

    def _flash(self, msg: str, duration: float = 0.9):
        self._flash_text = msg
        self._flash_until = time.time() + duration

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_overlay(self, frame: np.ndarray, fps: float, now: float):
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        # ── Top bar (recognition) ──────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.TOP_BAR_H), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

        sign = self.current_sign
        conf = self.current_confidence

        if sign == "No Sign":
            text_color = (120, 120, 120)
        elif conf > 0.85:
            text_color = (80, 255, 80)
        elif conf > 0.70:
            text_color = (80, 200, 255)
        else:
            text_color = (80, 150, 255)

        sign_text = sign if sign == "No Sign" else sign.upper()
        (tw, _), _ = cv2.getTextSize(sign_text, font, 1.4, 3)
        cv2.putText(frame, sign_text, ((w - tw) // 2, 65), font, 1.4, text_color, 3)

        # Confidence bar
        if sign != "No Sign" and conf > 0:
            bx, by, bw, bh = 15, 78, w - 30, 10
            filled = int(bw * conf)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
            cv2.rectangle(frame, (bx, by), (bx + filled, by + bh), text_color, -1)
            cv2.putText(frame, f'{conf*100:.0f}%',
                        (bx + bw + 5, by + 9), font, 0.45, text_color, 1)

        # Hold progress arc — shown inside top bar (right side)
        hold_p = self.sentence_builder.hold_progress(now)
        if hold_p > 0:
            cx, cy, r = w - 28, 28, 18
            # Background circle
            cv2.circle(frame, (cx, cy), r, (60, 60, 60), 3)
            # Arc — drawn as a series of lines approximating the arc
            angle_end = int(-90 + 360 * hold_p)
            cv2.ellipse(frame, (cx, cy), (r, r), 0, -90, angle_end, (0, 230, 255), 3)
            pct_str = f"{int(hold_p*100)}%"
            (pw, _), _ = cv2.getTextSize(pct_str, font, 0.32, 1)
            cv2.putText(frame, pct_str, (cx - pw // 2, cy + 4), font, 0.32, (200, 200, 200), 1)

        # FPS
        cv2.putText(frame, f'FPS: {fps:.0f}', (10, 25), font, 0.55, (150, 150, 150), 1)

        # ── Sentence panel ────────────────────────────────────────────────────
        panel_y = h - self.SENTENCE_H - self.HINT_H
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, panel_y), (w, panel_y + self.SENTENCE_H),
                      (20, 20, 35), -1)
        cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

        # Panel border
        cv2.rectangle(frame, (0, panel_y), (w, panel_y + self.SENTENCE_H),
                      (60, 60, 100), 1)

        # Label
        cv2.putText(frame, "SENTENCE", (12, panel_y + 18),
                    font, 0.45, (100, 100, 160), 1)

        # Sentence text — scroll if too long
        sentence = self.sentence_builder.sentence
        display_sentence = sentence if sentence else "—"
        # Fit text in panel width; truncate left if too wide
        max_chars = (w - 30) // 13   # rough char width estimate at scale 0.8
        if len(display_sentence) > max_chars:
            display_sentence = "…" + display_sentence[-(max_chars - 1):]

        (sw, _), _ = cv2.getTextSize(display_sentence, font, 0.85, 2)
        sentence_color = (240, 240, 240) if sentence else (80, 80, 80)
        cv2.putText(frame, display_sentence, (12, panel_y + 52),
                    font, 0.85, sentence_color, 2)

        # Blinking cursor
        if int(now * 2) % 2 == 0:
            cursor_x = 12 + sw + 4
            cv2.rectangle(frame, (cursor_x, panel_y + 36),
                          (cursor_x + 3, panel_y + 56), (180, 180, 180), -1)

        # Flash feedback (e.g. "Added: A")
        if now < self._flash_until:
            flash_alpha = min(1.0, (self._flash_until - now) / 0.3)
            flash_color = (
                int(80 * flash_alpha),
                int(255 * flash_alpha),
                int(150 * flash_alpha)
            )
            cv2.putText(frame, self._flash_text,
                        (12, panel_y + 78), font, 0.45, flash_color, 1)

        # ── Hint bar ──────────────────────────────────────────────────────────
        hint_y = h - self.HINT_H
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (0, hint_y), (w, h), (12, 12, 12), -1)
        cv2.addWeighted(overlay3, 0.80, frame, 0.20, 0, frame)

        hints = "SPACE: add  |  Hold 2s: auto-add  |  BKSP: delete  |  C: clear  |  S/↵: speak  |  Q: quit"
        cv2.putText(frame, hints, (8, h - 9), font, 0.38, (130, 130, 130), 1)

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self, camera_index: int = 0):
        """Open webcam and run real-time recognition + sentence builder."""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 60)

        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"\nActual camera: {actual_w}x{actual_h} @ {actual_fps}fps")

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_index}")

        print("\nSignSpeak running.")
        print("  SPACE     — add current sign to sentence")
        print("  Hold 2s   — auto-add current sign")
        print("  BACKSPACE — delete last token")
        print("  C         — clear sentence")
        print("  S / ENTER — speak full sentence")
        print("  Q         — quit\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame read failed.")
                    break

                frame = cv2.flip(frame, 1)

                key = cv2.waitKey(1) & 0xFF
                # Treat Q separately so we can break cleanly
                if key == ord('q'):
                    break

                output = self.process_frame(frame, key if key != 255 else -1)
                cv2.imshow(WINDOW_TITLE, output)

                if not hasattr(self, '_icon_set'):
                    set_window_icon(WINDOW_TITLE, resource_path('ss_v3.ico'))
                    self._icon_set = True

        except KeyboardInterrupt:
            print("\nStopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.hand_tracker.release()
            print("SignSpeak closed.")

    def release(self):
        self.hand_tracker.release()