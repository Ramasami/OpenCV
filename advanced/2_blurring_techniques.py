# %%
import cv2 as cv
import numpy as np

# %%
dir = "../Resources/"

# %% Reading Images

img = cv.imread(dir + 'Photos/cats.jpg')
cv.imshow('Image', img)

# %% Average
'''
average of the surrounding pixles
123
456
789

5 = (1+2+3+4+6+7+8+9)/8
'''

avg = cv.blur(img, (5, 5))
cv.imshow('Average Blur', avg)

# %% Guassian Blur
'''
each surrounding pixel is given a weight
this gives less blurring, but image looks more natural
sigmaX = standard deviation
'''

guassian = cv.GaussianBlur(img, (7, 7), sigmaX=0.5)
cv.imshow('Guassian Blur', guassian)

# %% Median Blur
'''
median of the surrounding pixles
123
456
789

5 = (4+6)/2
tends to be more effective noise reduction
'''

median = cv.medianBlur(img, 3)
cv.imshow('Median Blur', median)

# %% Billateral Blur
'''
Most effective
Traditional blur the image without looking if edges are blurred or not
Bilateral blurs the image but retains the edges

sigmaSpace = how far pixels affect the blurring
'''

bilateral = cv.bilateralFilter(img, 10, 35, 25)
cv.imshow('Bilateral Blur', bilateral)

# %%
cv.waitKey(0)
