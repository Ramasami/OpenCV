import time

import cv2
import numpy as np
import pyautogui

import HandTrackingModule as htm


def display(img, name="Image"):
    cv2.imshow(name, img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    return img, current_time


frameR = 100
wCam, hCam = (640, 480)
smoothening = 2
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.HandDetector()
prev_time = 0
wScreen, hScreen = pyautogui.size()
print(wScreen, hScreen)
px = 0
py = 0
while True:
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # Find Hand Landmarks
    img = detector.find_hands(img, False)
    lm_list = detector.find_position(img, draw=False)
    bbox = detector.get_bbox(img, False)

    if len(lm_list) != 0:
        # Get the tip of the index and middle fingers
        _, x1, y1 = lm_list[8]
        _, x2, y2 = lm_list[12]

        # Check which finger are up
        fingers = detector.fingers_up()

        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (0, 255, 0), 2)

        # Only index finger: move mouse
        if fingers[1] and not fingers[2]:
            # Convert coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

            # Smoothen Values
            cx = px + (x3 - px) // smoothening
            cy = py + (y3 - py) // smoothening
            # Move Mouse
            pyautogui.moveTo(cx, cy)

            px = cx
            py = cy

            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        # Both Index and middle fingers are up: Clicking mode
        if fingers[1] and fingers[2]:
            # Find distance between fingers
            length, img, _ = detector.find_distance(8, 12, img, False)
            if length is not None and length < 40:
                # Click mouse if distance is short
                cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                pyautogui.leftClick()
            else:
                cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
    # Frame Rate
    img, prev_time = add_fps(img, prev_time)
    if not display(img):
        break
