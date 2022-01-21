# https://github.com/opencv/opencv/tree/master/data/haarcascades

# %% imports
import cv2 as cv

# %% Value Settings
dir = "../Resources/"


# %% Read Image
img = cv.imread(dir + "Photos/group 1.jpg")
cv.imshow("Image", img)
haar_cascade = cv.CascadeClassifier(
    dir + "Faces/haar_cascade_face_default.xml")

# %% Detect Face


def detectFace(frame, minNeighors=3):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # tune the paramter minNeighbors to get better results
    faces_rect = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=minNeighors)
    print("Faces Found", len(faces_rect))
    for (x, y, w, h) in faces_rect:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
    return frame

def flip(frame):
    return frame[:, ::-1, :]

# image Face Detect
cv.imshow("Faces Detected",  detectFace(img))

# %% Video Face Detect
capture = cv.VideoCapture(0)
while(capture.isOpened()):
    _, frame = capture.read()
    cv.imshow('Video', flip(detectFace(frame)))
    if cv.waitKey(20) == ord('d'):
        break
capture.release()
cv.destroyAllWindows()
