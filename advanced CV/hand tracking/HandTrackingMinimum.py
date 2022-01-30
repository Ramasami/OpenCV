# %%
import cv2
import mediapipe as mp
import time

# %%
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# %%
cap = cv2.VideoCapture(0)

ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for hand_land_mark in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_land_mark.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 0:
                    # cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand_land_mark,
                                  mpHands.HAND_CONNECTIONS)

    ctime = time.time()
    fps = 1 / (ctime-ptime)
    ptime = ctime
    img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
