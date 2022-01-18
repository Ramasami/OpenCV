# %% Imports
import cv2 as cv
import numpy as  np

# %% Value Settings
dir = "../Resources/"

# %% Read Image
img = cv.imread(dir + 'Photos/cat.jpg')
cv.imshow('Cat', img)
cv.waitKey(0)

# %% Blank Image
blank = np.zeros((500,500, 3),dtype='uint8')
cv.imshow('Blank', blank)

blank[:] = 0,255,0
cv.imshow('Green', blank)

blank[:] = 0,0,255
cv.imshow('Red', blank)

blank[:] = 255,0,0
cv.imshow('Blue', blank)
cv.waitKey(0)

# %% Rectangle
blank = np.zeros((500,500, 3),dtype='uint8')
cv.rectangle(blank, (0,0),(250,500), (0,255,0), thickness=2)
cv.imshow('Rectangle', blank)

cv.rectangle(blank, (0,0),(250,500), (0,255,0), thickness=-1)
# cv.rectangle(blank, (0,0),(250,500), (0,255,0), thickness=cv.FILLED)
cv.imshow('Solid-Rectangle', blank)
cv.waitKey(0)

# %% Circle
blank = np.zeros((500,500, 3),dtype='uint8')
cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2), 125, (255,0,0), thickness=2)
cv.imshow('Circle', blank)

cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2), 125, (255,0,0), thickness=-1)
# cv.circle(blank, (blank.shape[1]//2,blank.shape[0]//2), 125, (255,0,0), thickness=cv.FILLED)
cv.imshow('Solid-Circle', blank)
cv.waitKey(0)

# %% Line
blank = np.zeros((500,500, 3),dtype='uint8')
cv.line(blank, (0,0), (blank.shape[1]//2,blank.shape[0]//2), (255,0,0), thickness=2)
cv.imshow('Line', blank)
cv.waitKey(0)

# %% Text
blank = np.zeros((500,500, 3),dtype='uint8')
cv.putText(blank, "Hello",(255,255),cv.FONT_HERSHEY_TRIPLEX, 1, (255,255,255))
cv.imshow('Text', blank)
cv.waitKey(0)