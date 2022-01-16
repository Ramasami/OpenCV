# %% Imports
import cv2 as cv

# %% Value Settings
dir = "../Resources/"

# %% Rescale Frame
def rescaleFrame(frame, scale=0.75):
    #Images, Videos, LiveVideos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

def changeRes(capture, width, height):
    #LiveVideos
    capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)

# %% Reading Images

img = cv.imread(dir + 'Photos/cat.jpg')
cv.imshow('Cat', img)
cv.waitKey(0)

cv.imshow('Cat', rescaleFrame(img, 2))
cv.waitKey(0)

# %% Reading Videos
capture = cv.VideoCapture(dir+"Videos/dog.mp4")
while(capture.isOpened()):
    _, frame = capture.read()
    cv.imshow('Cat', rescaleFrame(frame))
    if(cv.waitKey(20) == ord('d')):
        break
capture.release()
cv.destroyAllWindows()

# %% Reading Webcam
capture = cv.VideoCapture(0)
changeRes(capture, 2000,2000)
while(capture.isOpened()):
    _, frame = capture.read()
    cv.imshow('Cat', rescaleFrame(frame))
    if(cv.waitKey(20) == ord('d')):
        break
capture.release()
cv.destroyAllWindows()