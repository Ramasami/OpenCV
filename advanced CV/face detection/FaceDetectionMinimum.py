import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpFaceDetection = mp.solutions.face_detection
face = mpFaceDetection.FaceDetection()
mpDraw = mp.solutions.drawing_utils


def add_fps(img, prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    img = cv2.putText(img, str(int(fps)), (10, 70),
                      cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    return img, current_time


def display(img):
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        return False
    return True


prev_time = 0
while True:
    _, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face.process(img_rgb)
    if (results.detections):
        for id, detection in enumerate(results.detections):
            mpDraw.draw_detection(img, detection)
            print(detection)
            bboxC = detection.location_data.relative_bounding_box
            h, w, c = img.shape
            bbox = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

    img, prev_time = add_fps(img, prev_time)
    if not display(img):
        break
cap.release()
cv2.destroyAllWindows()
