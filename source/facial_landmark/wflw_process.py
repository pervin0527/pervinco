import os
import cv2
import sys
import shutil
import numpy as np

debug = False
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256, radians=False):

    assert landmarks_2D is not None, 'landmarks_2D is None'
    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  
        [1.330353, 7.122144, 6.903745],  
        [-1.330353, 7.122144, 6.903745], 
        [-6.825897, 6.760612, 4.402142], 
        [5.311432, 5.485328, 3.987654],  
        [1.789930, 5.393625, 4.413414],  
        [-1.789930, 5.393625, 4.413414], 
        [-5.311432, 5.485328, 3.987654], 
        [-2.005628, 1.409845, 6.165652], 
        [-2.005628, 1.409845, 6.165652], 
        [2.774015, -2.080775, 5.048531], 
        [-2.774015, -2.080775, 5.048531],
        [0.000000, -3.116408, 6.097667], 
        [0.000000, -7.415691, 4.070434], 
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, camera_distortion)
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    return map(lambda k: k[0], euler_angles)

def rotate(angle, center, landmark):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2], M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_

class ImageDate():
    def __init__(self, line, imgDir, image_size=112):
        self.image_size = image_size
        line = line.strip().split()
        assert(len(line) == 207)
        self.list = line
        self.landmark = np.asarray(list(map(float, line[:196])), dtype=np.float32).reshape(-1, 2)
        self.box = np.asarray(list(map(int, line[196:200])),dtype=np.int32)
        flag = list(map(int, line[200:206]))
        flag = list(map(bool, flag))
        self.pose = flag[0]
        self.expression = flag[1]
        self.illumination = flag[2]
        self.make_up = flag[3]
        self.occlusion = flag[4]
        self.blur = flag[5]
        self.path = os.path.join(imgDir, line[206])
        self.img = None

        self.imgs = []
        self.landmarks = []
        self.boxes = []

    def load_data(self, is_train, repeat, mirror=None):
        if (mirror is not None):
            with open(mirror, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 1
                mirror_idx = lines[0].strip().split(',')
                mirror_idx = list(map(int, mirror_idx))
        xy = np.min(self.landmark, axis=0).astype(np.int32) 
        zz = np.max(self.landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        img = cv2.imread(self.path)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        if imgT.shape[0] == 0 or imgT.shape[1] == 0:
            imgTT = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for x, y in (self.landmark+0.5).astype(np.int32):
                cv2.circle(imgTT, (x, y), 1, (0, 0, 255))
            cv2.imshow('0', imgTT)
            if cv2.waitKey(0) == 27:
                exit()
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmark = (self.landmark - xy)/boxsize
        assert (landmark >= 0).all(), str(landmark) + str([dx, dy])
        assert (landmark <= 1).all(), str(landmark) + str([dx, dy])
        self.imgs.append(imgT)
        self.landmarks.append(landmark)

        if is_train:
            while len(self.imgs) < repeat:
                angle = np.random.randint(-30, 30)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmark = rotate(angle, (cx,cy), self.landmark)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1]*1.1), int(img.shape[0]*1.1)))

                
                wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmark = (landmark - xy) / size
                if (landmark < 0).any() or (landmark > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if mirror is not None and np.random.choice((True, False)):
                    landmark[:,0] = 1 - landmark[:,0]
                    landmark = landmark[mirror_idx]
                    imgT = cv2.flip(imgT, 1)
                self.imgs.append(imgT)
                self.landmarks.append(landmark)

    def save_data(self, path, prefix):
        attributes = [self.pose, self.expression, self.illumination, self.make_up, self.occlusion, self.blur]
        attributes = np.asarray(attributes, dtype=np.int32)
        attributes_str = ' '.join(list(map(str, attributes)))
        labels = []
        TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
        for i, (img, lanmark) in enumerate(zip(self.imgs, self.landmarks)):
            assert lanmark.shape == (98, 2)
            save_path = os.path.join(path, prefix+'_'+str(i)+'.png')
            assert not os.path.exists(save_path), save_path
            cv2.imwrite(save_path, img)

            euler_angles_landmark = []
            for index in TRACKED_POINTS:
                euler_angles_landmark.append(lanmark[index])
            euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
            pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
            euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            landmark_str = ' '.join(list(map(str,lanmark.reshape(-1).tolist())))

            label = '{} {} {} {}\n'.format(save_path, landmark_str, attributes_str, euler_angles_str)

            labels.append(label)
        return labels
def get_dataset_list(imgDir, outDir, landmarkDir, is_train):
    with open(landmarkDir,'r') as f:
        lines = f.readlines()
        labels = []
        save_img = os.path.join(outDir, 'imgs')
        if not os.path.exists(save_img):
            os.mkdir(save_img)

        if debug:
            lines = lines[:100]
        for i, line in enumerate(lines):
            Img = ImageDate(line, imgDir)
            img_name = Img.path
            Img.load_data(is_train, 10, Mirror_file)
            _, filename = os.path.split(img_name)
            filename, _ = os.path.splitext(filename)
            label_txt = Img.save_data(save_img, str(i)+'_' + filename)
            labels.append(label_txt)
            if ((i + 1) % 100) == 0:
                print('file: {}/{}'.format(i+1, len(lines)))

    with open(os.path.join(outDir, 'list.txt'),'w') as f:
        for label in labels:
            f.writelines(label)

if __name__ == '__main__':
    root_dir = "/data/Datasets/WFLW"
    imageDirs = f'{root_dir}/WFLW_images'
    Mirror_file = f'{root_dir}/WFLW_annotations/Mirror98.txt'

    landmarkDirs = [f'{root_dir}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
                    f'{root_dir}/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt']

    outDirs = ['test_data', 'train_data']
    for landmarkDir, outDir in zip(landmarkDirs, outDirs):
        outDir = os.path.join(root_dir, outDir)
        print(outDir)
        if os.path.exists(outDir):
            shutil.rmtree(outDir)
        os.mkdir(outDir)
        if 'list_98pt_rect_attr_test.txt' in landmarkDir:
            is_train = False
        else:
            is_train = True
        imgs = get_dataset_list(imageDirs, outDir, landmarkDir, is_train)
    print('end')