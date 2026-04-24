import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode
)
import os

class HandTracker:
    def __init__(self, model_path='hand_landmarker.task',
                 max_hands=1, detection_confidence=0.6, tracking_confidence=0.5):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"hand_landmarker.task not found at: {model_path}")
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def extract_features(self, hand_landmarks):
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
        coords -= coords[0]
        scale = np.linalg.norm(coords[9])
        if scale > 1e-6:
            coords /= scale
        return coords.flatten()

    def detect_hands(self, frame):
        frame_drawn = frame.copy()
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        features_list = []
        if result.hand_landmarks:
            for idx, hand_lms in enumerate(result.hand_landmarks):
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                connections = [
                    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
                ]
                for a, b in connections:
                    cv2.line(frame_drawn, points[a], points[b], (200,200,200), 2)
                for i, pt in enumerate(points):
                    cv2.circle(frame_drawn, pt, 4, (0,255,0) if i==0 else (255,100,100), -1)
                xs, ys = [p[0] for p in points], [p[1] for p in points]
                cv2.rectangle(frame_drawn, (max(0,min(xs)-15), max(0,min(ys)-15)),
                              (min(w,max(xs)+15), min(h,max(ys)+15)), (0,200,100), 2)
                features_list.append(self.extract_features(hand_lms))
        return frame_drawn, features_list

    def release(self):
        self.landmarker.close()