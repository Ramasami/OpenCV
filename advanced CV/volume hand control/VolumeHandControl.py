import math
import time
from ctypes import cast, POINTER

import cv2
import numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

import HandTrackingModule as htm


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.flip(img, 1)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img, current_time


def display(img):
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


cap = cv2.VideoCapture(0)
prev_time = 0
detector = htm.HandDetector(detection_confidence=0.7)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
minVol, maxVol, _ = volume.GetVolumeRange()

while True:
    _, img = cap.read()
    img = detector.find_hands(img)
    lm_list = detector.find_position(img, 0, False)
    if len(lm_list) != 0:
        x1, y1 = lm_list[detector.mpHands.HandLandmark.THUMB_TIP][1:3]
        x2, y2 = lm_list[detector.mpHands.HandLandmark.INDEX_FINGER_TIP][1:3]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 2550, 3))
        length = math.hypot(x2 - x1, y2 - y1)
        vol = np.interp(length, [50, 300], [minVol, maxVol])
        if length > 50:
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        else:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img, (50, 150), (58, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(np.interp(vol, [minVol, maxVol], [400, 150]))), (58, 400), (0, 255, 0), cv2.FILLED)

    img, prev_time = add_fps(img, prev_time)
    if not display(img):
        break
