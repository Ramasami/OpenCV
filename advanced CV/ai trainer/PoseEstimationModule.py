import cv2
import mediapipe as mp
import math


class PoseDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False,
                 smooth_segmentation=True, detection_confidence=0.5, track_confidence=0.5):
        self.lm_list = None
        self.results = None
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.track_confidence = track_confidence

        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation,
            self.detection_confidence, self.track_confidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_pose(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img_rgb)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def find_position(self, img, draw=True):
        self.lm_list = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return self.lm_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        _, x1, y1 = self.lm_list[p1]
        _, x2, y2 = self.lm_list[p2]
        _, x3, y3 = self.lm_list[p3]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))

        if angle < 0:
            angle += 360
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.line(img, (x3, y3), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (x1, y1), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x1, y1), 10, (255, 0, 0), 1)
            cv2.circle(img, (x2, y2), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 0), 1)
            cv2.circle(img, (x3, y3), 5, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 10, (255, 0, 0), 1)
            cv2.putText(img, str(int(angle)), (x2 - 20, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        return angle

