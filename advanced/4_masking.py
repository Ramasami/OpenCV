# %%
import cv2 as cv
import numpy as np

# %%
dir = "../Resources/"

# %% Reading Images

img = cv.imread(dir + 'Photos/cats.jpg')
cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')

mask = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
cv.imshow('Mask', mask)

masked_image = cv.bitwise_and(img, img, mask=mask)
green_masked_image = cv.bitwise_and(img, cv.merge([blank, mask, blank]))
cv.imshow('Masked', masked_image)
cv.imshow('Green_Masked', green_masked_image)


# %%
cv.waitKey(0)
