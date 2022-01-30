import time

import cv2

import HandTrackingModule as htm

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
prev_time = 0
detector = htm.HandDetector(detection_confidence=0.75)


def get_finger_images():
    fingers_image = cv2.imread("fingers.jpg")
    w, h, c = fingers_image.shape
    fingers = []
    fw, fh = 100, 100
    for i in range(3):
        for j in range(4):
            finger = fingers_image[(w // 3) * i:(w // 3) * (i + 1), (h // 4) * j:(h // 4) * (j + 1), :]
            fingers.append(cv2.resize(finger, (fw, fh)))
    return fingers, fw, fh


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


fingers, fw, fh = get_finger_images()

tipIds = [8, 12, 16, 20]
while True:
    _, img = cap.read()
    detector.find_hands(img)
    lmList = detector.find_position(img, draw=False)

    if len(lmList) != 0:
        fingersOpen = []

        if (lmList[4][1] < lmList[20][1] and lmList[4][1] < lmList[3][1]) or (
                lmList[4][1] > lmList[20][1] and lmList[4][1] > lmList[3][1]):
            fingersOpen.append(1)
        else:
            fingersOpen.append(0)

        for tip in tipIds:
            if lmList[tip][2] < lmList[tip - 2][2]:
                fingersOpen.append(1)
            else:
                fingersOpen.append(0)
        count = sum(fingersOpen)
        print(fingersOpen, sum(fingersOpen))
        img[0:fh, 0:fw] = fingers[count]
    img, prev_time = add_fps(img, prev_time)
    if not display(img):
        break
