# https://github.com/opencv/opencv/tree/master/data/haarcascades

# %% imports
import cv2 as cv

# %% Value Settings
dir = "../Resources/"

# %% Read Image
img = cv.imread(dir + "Photos/group 1.jpg")
cv.imshow("Image", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier(
    dir + "Faces/haar_cascade_face_default.xml")

# tune the paramter minNeighbors to get better results
faces_rect = haar_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=1)

print("Faces Found", len(faces_rect))

for (x, y, w, h) in faces_rect:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)
cv.imshow("Faces Detected",  img)
# %%
cv.waitKey(0)
