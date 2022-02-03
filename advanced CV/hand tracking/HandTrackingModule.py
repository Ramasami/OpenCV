import math

import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, track_confidence=0.5):
        self.lm_list = None
        self.results = None
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            self.mode, self.max_hands, self.model_complexity, self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)
        if self.results.multi_hand_landmarks:
            for hand_land_mark in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_land_mark,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def get_bbox(self, img, draw=True):
        if len(self.lm_list) == 0:
            return
        x_list = [c[1] for c in self.lm_list]
        y_list = [c[2] for c in self.lm_list]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)
        bbox = x_min, x_max, y_min, y_max
        if draw:
            cv2.rectangle(img, (x_min - 20, y_min - 20), (x_max + 20, y_max + 20), (0, 255, 0), 2)
        return bbox

    def fingers_up(self):
        fingers_open = []
        if self.lm_list is None or len(self.lm_list) == 0:
            return [False, False, False, False, False]
        tip_ids = [8, 12, 16, 20]
        if (self.lm_list[4][1] < self.lm_list[20][1] and self.lm_list[4][1] < self.lm_list[3][1]) or (
                self.lm_list[4][1] > self.lm_list[20][1] and self.lm_list[4][1] > self.lm_list[3][1]):
            fingers_open.append(True)
        else:
            fingers_open.append(False)

        for tip in tip_ids:
            if self.lm_list[tip][2] < self.lm_list[tip - 2][2]:
                fingers_open.append(True)
            else:
                fingers_open.append(False)
        return fingers_open

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        if self.lm_list is None or len(self.lm_list) == 0:
            return None, img, None
        _, x1, y1 = self.lm_list[p1]
        _, x2, y2 = self.lm_list[p2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
