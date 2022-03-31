import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2


class PinhomeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.distortion = (abs(k1) > 0.00000001)
        self.d = [self.k1, self.k2, self.p1, self.p2, self.k3]


class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.last_frame = None
        self.new_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_cur = None
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.detector = cv2.SIFT_create()
        self._FLANN_INDEX_KDTREE = 0
        self._FLANN_INDEX_KDTREE = 0
        self._MIN_MATCH_COUNT = 10
        self._index_params = dict(algorithm=self._FLANN_INDEX_KDTREE, trees=5)
        self._search_params = dict(checks=50)
        self._flann = cv2.FlannBasedMatcher(self._index_params, self._search_params)

    def featureMatching(self, image_ref, image_cur):
        matching_flag = False
        kp1, des1 = self.detector.detectAndCompute(image_ref, None)
        kp2, des2 = self.detector.detectAndCompute(image_cur, None)
        matches = self._flann.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > self._MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matching_flag = True
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 1.0)
            matchesMask = mask.ravel().tolist()
            _draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=matchesMask, flags=2)
            res = None
            res = cv2.drawMatches(image_ref, kp1, image_cur, kp2, good, res, **_draw_params)
            plt.imshow(res)
            # plt.show()

        return matching_flag, src_pts, dst_pts

    def processFisrtFrame(self, image):
        self.px_ref = image
        self.frame_stage = 1
        self.frame_stage = STAGE_SECOND_FRAME

    def processSecondFrame(self, image):
        matching_flag, self.px_cur, self.px_ref = self.featureMatching(self.px_ref, image)
        if matching_flag:
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.88, threshold=1.0)
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
            self.px_ref = image
            self.frame_stage = STAGE_DEFAULT_FRAME

    def processFrame(self, image):
        matching_flag, self.px_cur, self.px_ref = self.featureMatching(self.px_ref, image)
        if matching_flag:
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.88, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)
            self.cur_t = self.cur_t + self.cur_R.dot(t)
            self.cur_R = R.dot(self.cur_R)
            self.px_ref = image

    def update(self, img):
        if (self.frame_stage == STAGE_FIRST_FRAME):
            self.processFisrtFrame(img)
        elif (self.frame_stage == STAGE_SECOND_FRAME):
            self.processSecondFrame(img)
        else:
            self.processFrame(img)


def func(num, dataset, draw_line, draw_dot):
    # print('idx :{} , line_set_data:{}'.format(num, dataset[0:2, :num]))
    draw_line.set_data(dataset[0:2, :num])
    draw_line.set_3d_properties(dataset[2, :num])
    draw_dot.set_data(dataset[0:2, :num])
    draw_dot.set_3d_properties(dataset[2, :num])
    return draw_line


cam = PinhomeCamera(0, 0, fx=1767.637609, fy=1767.637609, cx=640.00, cy=512.00, k1=-0.153634, k2=-0.011194, p1=-0.008929, p2=-0.014213)
vo = VisualOdometry(cam)
# traj = np.zeros((0, 3), dtype=np.uint8)
x_list = []
y_list = []
z_list = []
traj_list = []
img_path = '/d/VIDEO/Gastre/stereo_1/left_15fps/'
img_list = os.listdir(img_path)
for img_id in tqdm(range(len(img_list))):
    if img_id > 50:
        break
    img = cv2.imread(img_path + "/" + img_list[img_id], 0)
    vo.update(img)
    cur_t = vo.cur_t
    if (img_id > 0):
        x, y, z = cur_t[0][0], cur_t[1][0], cur_t[2][0]
    else:
        x, y, z = 0., 0., 0.
    # traj_list.append([x, y, z])
    x_list.append(x)
    y_list.append(y)
    z_list.append(z)
dataset = np.array([x_list, y_list, z_list])
if not os.path.exists('./traj_matching.npy'):
    np.save('./traj_matching.npy', dataset)
numDataPoints = dataset.shape[1]
fig = plt.figure()
ax = Axes3D(fig)
draw_dot = plt.plot(dataset[0], dataset[1], dataset[2], lw=2, c='r', marker='o')[0]
draw_line = plt.plot(dataset[0], dataset[1], dataset[2], lw=2, c='g')[0]
text_index = [i for i in range(numDataPoints)]
text_index = np.array(text_index)
text_index = text_index.reshape([1, numDataPoints])
dataset = np.vstack((dataset, text_index))
# draw_txt = plt.text(dataset[0], dataset[1], dataset[3])[0]
ax.set_xlabel('X(t)')
ax.set_ylabel('Y(t)')
ax.set_zlabel('Z(t)')
ax.set_title('Trajectory of VO')

line_ani = animation.FuncAnimation(fig, func, frames=numDataPoints, fargs=(dataset, draw_line, draw_dot),
                                   interval=1000, blit=False)
plt.show()
