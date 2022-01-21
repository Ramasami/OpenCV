# %% imports
import os
import cv2 as cv
import numpy as np

# %% Value Settings
dir = "../Resources/Faces/"
people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
vdir = os.path.join(dir, 'val')

haar_cascade = cv.CascadeClassifier(
    dir + "haar_cascade_face_default.xml")

features = np.load(dir + 'trained/features.npy', allow_pickle=True)
label = np.load(dir + 'trained/labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(dir + 'trained/face_trained.yml')

# %%
path = os.path.join(vdir, 'ben_afflek', '1.jpg')
img = cv.imread(path)

# %% Detect Face in image

def detect(frame, name):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces_rect:
        faces_roi = gray[y:y+h, x:x+w]
        label, confidence = face_recognizer.predict(faces_roi)
        print(f'Label = {people[label]} with a confidence of {confidence}')
        cv.putText(frame, str(people[label]), (20, 20),
                cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), thickness=2)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv.imshow(name, frame)

for person in os.listdir(vdir):
    path = os.path.join(vdir, person)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        img_array = cv.imread(img_path)
        detect(img_array, img_path)
        
cv.waitKey(0)
