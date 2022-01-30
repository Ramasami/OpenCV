import time

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
mpFaceMesh = mp.solutions.face_mesh
face = mpFaceMesh.FaceMesh()
connections = mp.solutions.face_mesh_connections
mpDraw = mp.solutions.drawing_utils
drawSpec = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)


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
    if (results.multi_face_landmarks):
        for face_landmarks in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, face_landmarks, connections.FACEMESH_TESSELATION, drawSpec, drawSpec)
            for id, lm in enumerate(face_landmarks.landmark):
                h, w, c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                print(id, x, y)

    img, prev_time = add_fps(img, prev_time)
    if not display(img):
        break
cap.release()
cv2.destroyAllWindows()
