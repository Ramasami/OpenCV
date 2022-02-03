import math
import os

import cv2
import numpy as np

import HandTrackingModule as htm


def add_header(img, header):
    img[0:125, 0:1280] = header
    return img


def display(img, name="Image"):
    cv2.imshow(name, img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


path = "header/"
myList = os.listdir(path)
overLayList = [cv2.imread(path + imPath) for imPath in myList]
drawColors = [(201, 205, 152), (124, 173, 235), (196, 102, 255), (148, 234, 123), (255, 255, 255)]
color_mode = 0
header = overLayList[color_mode]
drawColor = drawColors[color_mode]
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = htm.HandDetector()

brushThickness = 15
xp, yp = None, None
canvas = np.zeros((720, 1280, 3), dtype='uint8')
eraser = False
plain_banner = np.zeros((125, 1280, 3))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.find_hands(img, False)
    lm_list = detector.find_position(img, draw=False)
    if len(lm_list) != 0:
        _, x1, y1 = lm_list[8]  # Index Finger
        _, x2, y2 = lm_list[12]  # Middle Finger
        fingers = detector.fingers_up()
        # Selection mode - two fingers up
        if fingers[1] and fingers[2] and math.hypot((x1 - x2), (y1 - y2)) < 65:
            if y1 < 125:
                if 380 < x1 <= 500:
                    color_mode = 0
                elif 500 < x1 <= 630:
                    color_mode = 1
                elif 630 < x1 <= 775:
                    color_mode = 2
                elif 775 < x1 <= 955:
                    color_mode = 3
                elif 1080 < x1 < 1220:
                    color_mode = 4
                eraser = color_mode == 4
                header = overLayList[color_mode]
                drawColor = drawColors[color_mode]

            cv2.circle(img, (x1, y1), 15, drawColor, 2)

        # Draw mode - one fingers up
        elif fingers[1]:
            if xp is not None and yp is not None:
                if eraser:
                    color = (0, 0, 0)
                    thickness = 30
                else:
                    color = drawColor
                    thickness = brushThickness
                canvas = cv2.line(canvas, (xp, yp), (x1, y1), color, thickness)
                cv2.circle(img, (x1, y1), thickness, drawColor, cv2.FILLED)
        xp, yp = x1, y1

    img = add_header(img, header)
    canvas = add_header(canvas, plain_banner)
    canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, canvasInv = cv2.threshold(canvasGray, 50, 255, cv2.THRESH_BINARY_INV)
    canvasInv = cv2.cvtColor(canvasInv, cv2.COLOR_GRAY2BGR)
    img = img & canvasInv
    img = img | canvas
    if not display(img):
        break
