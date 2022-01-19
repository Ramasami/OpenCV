# %% Imports
import numpy as np
import cv2 as cv


# %% Value Settings
dir = "../Resources/"

# %% Read Image
image = cv.imread(dir + "Photos/cats.jpg")
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# %% Contours
canny = cv.Canny(gray, 125, 175)
contours1, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(contours1))

# %%
blur = cv.GaussianBlur(gray, (5, 5), cv.BORDER_DEFAULT)
canny = cv.Canny(blur, 125, 175)
contours2, heirarchies = cv.findContours(canny, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(contours2))

# %% Thresholding
ret, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY)

cv.imshow("Thresh", thresh)
cv.waitKey(0)

contours3, heirarchies = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
print(len(contours3))

# %% Draw
blank = np.zeros(image.shape, dtype='uint8')
cv.imshow('Blank', blank)
cv.waitKey(0)

blank = np.zeros(image.shape, dtype='uint8')
cv.drawContours(blank, contours1, -1, (0,0,255), 1)
cv.imshow('Blank', blank)
cv.waitKey(0)

blank = np.zeros(image.shape, dtype='uint8')
cv.drawContours(blank, contours2, -1, (0,0,255), 1)
cv.imshow('Blank', blank)
cv.waitKey(0)

blank = np.zeros(image.shape, dtype='uint8')
cv.drawContours(blank, contours3, -1, (0,0,255), 1)
cv.imshow('Blank', blank)
cv.waitKey(0)

cv.imshow('Blank', canny)
cv.waitKey(0)