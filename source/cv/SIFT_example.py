import cv2
import copy
from cv2 import ROTATE_90_CLOCKWISE
import numpy as np

filename = "./image/2007_000243.jpg"

image1 = cv2.imread(filename)
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

image2 = copy.copy(image1)
height, width = image2.shape[0], image2.shape[1]
image2 = cv2.resize(image2, (int(height*0.5), int(width*0.5)))
image2 = cv2.rotate(image2, ROTATE_90_CLOCKWISE)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints_1, descriptorrs_1 = sift.detectAndCompute(image1, None)
keypoints_2, descriptorrs_2 = sift.detectAndCompute(image2, None)

result1 = cv2.drawKeypoints(gray1, keypoints_1, image1)
result2 = cv2.drawKeypoints(gray2, keypoints_2, image2)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(descriptorrs_1, descriptorrs_2)
matches = sorted(matches, key = lambda x:x.distance)

result3 = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:50], image2, flags=2)

cv2.imshow('draw kepoints1', result1)
cv2.imshow('draw kepoints2', result2)
cv2.imshow('connected', result3)
cv2.waitKey(0)