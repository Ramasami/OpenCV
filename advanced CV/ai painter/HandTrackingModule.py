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

    def fingers_up(self):
        fingers_open = []
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
