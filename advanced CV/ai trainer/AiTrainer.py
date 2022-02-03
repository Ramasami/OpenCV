import time

import cv2
import numpy as np

import PoseEstimationModule as pem

cap = cv2.VideoCapture("../pose estimation/Resources/Triceps.mp4")
detector = pem.PoseDetector()


def display(img):
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


def add_count(img, count):
    img = cv2.putText(img, str(int(count)), (img.shape[1] - 100, img.shape[0] - 50),
                      cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    return img


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    return img, current_time


def add_bar(img, per):
    if 100 > per > 0:
        color = (0, 255, 0)
    else:
        color = (0, 0, 255)
    cv2.rectangle(img, (50, 150), (80, 400), color, 2)
    cv2.rectangle(img, (50, int(np.interp(per, [0, 100], [400, 150]))), (80, 400), color, cv2.FILLED)

    return img


count = 0
direction = 0
prev_time = 0
while True:
    success, img = cap.read()
    img = detector.find_pose(img, False)
    lmList = detector.find_position(img, False)
    if len(lmList) > 0:
        angle = detector.find_angle(img, 11, 13, 15)
        if angle > 180:
            angle = 360 - angle
        per = np.interp(angle, (90, 160), (100, 0))

        if per == 100 and direction == 0:
            direction = 1
        elif per == 0 and direction == 1:
            direction = 0
            count += 1
    img = add_bar(img, per)
    img = add_count(img, count)
    img, prev_time = add_fps(img, prev_time)
    if not display(img) or not success:
        break
