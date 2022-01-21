# %%
import cv2 as cv
import numpy as np

# %% Blank
blank = np.zeros((400, 400), dtype='uint8')

# %% Shapes
rectangle = cv.rectangle(blank.copy(), (30, 30), (370, 370), 255, -1)
circle = cv.circle(blank.copy(), (200, 200), 200, 255, -1)

cv.imshow('Rectangle', rectangle)
cv.imshow('Circle', circle)

# %% Bitwise
cv.imshow('AND', rectangle & circle)
cv.imshow('OR', cv.bitwise_or(rectangle, circle))
cv.imshow('XOR', cv.bitwise_xor(rectangle, circle))
cv.imshow('NOT', cv.bitwise_not(rectangle))


# %%
cv.waitKey(0)
