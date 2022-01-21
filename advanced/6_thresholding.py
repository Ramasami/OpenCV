# %% Imports
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# %% Value Settings
dir = "../Resources/"

# %% Read Image
image = cv.imread(dir + "Photos/cats.jpg")
cv.imshow("Image", image)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# %% Threshold
'''
it created a binary image, 
i.e. if pixel is greater than threshold, we can set it to 255
else set it to 0
'''
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Simple Thresholded', thresh)

threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
cv.imshow('Simple Thresholded Inverse', thresh)

# %% Adaptive Thresholding
'''
In previous case, we have to manually set the threshold
Now we will let the program determine the threshold

ADAPTIVE_THRESH_MEAN_C -> computes a mean over a block of size 11 to compute a threshold
ADAPTIVE_THRESH_GAUSSIAN_C -> applies a weight and works similar to above

C -> correctness
'''
adaptive_thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Mean Thresholded', adaptive_thresh)

adaptive_thresh = cv.adaptiveThreshold(
    gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 3)
cv.imshow('Adaptive Guassian Thresholded', adaptive_thresh)

# %%
cv.waitKey(0)
