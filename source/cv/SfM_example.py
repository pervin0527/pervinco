from matplotlib import projections
import sfm
import camera

import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import axes3d

def compute_fundamantal(x1, x2):
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    A = np.zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
                x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
                x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F

def compute_epipole(F):
    U, S, V = np.linalg.svd(F)
    e = V[-1]

    return e / e[2]

if __name__ == "__main__":
    data_path = "/data/Datasets/metron_college1"
    image1 = cv2.imread(f"{data_path}/images/001.jpg")
    image2 = cv2.imread(f"{data_path}/images/002.jpg")

    points2D = [np.loadtxt(data_path + '/2D/00' + str(i+1) + '.corners').T for i in range(3)]
    points3D = np.loadtxt(f"{data_path}/3D/p3d").T

    corr = np.genfromtxt(f'{data_path}/2D/nview-corners', dtype='int', missing_values='*')

    P = [camera.Camera(np.loadtxt(data_path + '/2D/00' + str(i+1) + '.P')) for i in range(3)]

    X = np.vstack((points3D, np.ones(points3D.shape[1])))
    x = P[0].project(X)

    # plt.figure()
    # plt.imshow(image1)
    # plt.plot(points2D[0][0], points2D[0][1], '*')
    # plt.axis('off')

    # plt.figure()
    # plt.imshow(image2)
    # plt.plot(x[0], x[1], 'r.')
    # plt.axis('off')

    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(points3D[0], points3D[1], points3D[2], 'k.')
    # plt.show()

    ndx = (corr[:, 0] >= 0) & (corr[:, 1] >=0)
    x1 = points2D[0][:, corr[ndx, 0]]
    x1 = np.vstack((x1, np.ones(x1.shape[1])))
    x2 = points2D[1][:, corr[ndx, 1]]
    x2 = np.vstack((x2, np.ones(x2.shape[1])))

    F = sfm.compute_fundamental(x1, x2)
    e = sfm.compute_epipole(F)

    # plt.figure()
    # plt.imshow(image1)
    # for i in range(5):
    #     sfm.plot_epipolar_line(image1,F,x2[:,i],e,False)
    # plt.axis('off')
    # plt.figure()

    # plt.imshow(image2)
    # for i in range(5):
    #     plt.plot(x2[0,i],x2[1,i],'o')
    # plt.axis('off')
    # plt.show()

    Xtrue = points3D[:, ndx]
    Xtrue = np.vstack((Xtrue, np.ones(Xtrue.shape[1])))

    Xest = sfm.triangulate(x1, x2, P[0].P, P[1].P)

    print(Xest[:, :3])
    print(Xtrue[:, :3])

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(Xest[0], Xest[1], Xest[2], 'ko')
    # ax.plot(Xtrue[0], Xtrue[1], Xtrue[2], 'r.')
    # plt.axis('auto')
    # plt.show()

    corr = corr[:, 0]
    ndx3D = np.where(corr>=0)[0]
    ndx2D = corr[ndx3D]

    x = points2D[0][:, ndx2D]
    x = np.vstack((x, np.ones(x.shape[1])))
    X = points3D[:, ndx3D]
    X = np.vstack((X, np.ones(X.shape[1])))

    Pest = camera.Camera(sfm.compute_P(x, X))

    print(Pest.P / Pest.P[2, 3])
    print(P[0].P / P[0].P[2, 3])

    xest = Pest.project(X)

    plt.figure()
    plt.imshow(image1)
    plt.plot(x[0], x[1], 'bo')
    plt.plot(xest[0], xest[1], 'r.')
    plt.axis('off')
    plt.show()