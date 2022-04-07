import cv2
import camera
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from numpy import loadtxt
from numpy import genfromtxt

data_path = "/data/Datasets/metron_college1"
image1 = cv2.imread(f"{data_path}/images/001.jpg")
image2 = cv2.imread(f"{data_path}/images/002.jpg")

points2D = [loadtxt(data_path + '/2D/00' + str(i+1) + '.corners').T for i in range(3)]
points3D = loadtxt(f"{data_path}/3D/p3d").T

corr = genfromtxt(f'{data_path}/2D/nview-corners', dtype='int', missing_values='*')

P = [camera.Camera(loadtxt(data_path + '/2D/00' + str(i+1) + '.P')) for i in range(3)]

X = np.vstack((points3D, np.ones(points3D.shape[1])))
x = P[0].project(X)

plt.figure()
plt.imshow(image1)
plt.plot(points2D[0][0], points2D[0][1], '*')
plt.axis('off')

plt.figure()
plt.imshow(image2)
plt.plot(x[0], x[1], 'r.')
plt.axis('off')

plt.show()