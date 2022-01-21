# %%
import cv2 as cv
import numpy as np

# %%
dir = "../Resources/"

# %% Reading Images

img = cv.imread(dir + 'Photos/lady.jpg')
cv.imshow('Image', img)

# %% Split Image
b, g, r = cv.split(img)

cv.imshow('B', b)
cv.imshow('G', g)
cv.imshow('R', r)

print(img.shape, b.shape, g.shape, r.shape)

# %% Merge
merged = cv.merge([b, g, r])
cv.imshow('Merged', merged)

# %% COlored Split Image
blank = np.zeros(img.shape[:2], dtype='uint8')
blue = cv.merge([b, blank, blank])
green = cv.merge([blank, g, blank])
red = cv.merge([blank, blank, r])
cv.imshow('Blue', blue)
cv.imshow('Green', green)
cv.imshow('Red', red)

# %%

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

colored = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
cv.imshow('Colored', colored)

# %%
cv.waitKey(0)
