import os
import cv2
import numpy as np
import albumentations as A

from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt


def visualize_sample(image, bbox, landmark):
    if type(image) == str:
        image = cv2.imread(f"{IMG_DIR}/{image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0))
    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 0, 0))

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.show()


def calculate_pitch_yaw_roll(landmarks_2D, cam_width, cam_height):
    """ Return the the pitch  yaw and roll angles associated with the input image.
    @param radians When True it returns the angle in radians, otherwise in degrees.
    """
    c_x = cam_width
    c_y = cam_height
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    #Estimated camera matrix values.
    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    #The dlib shape predictor returns 68 points, we are interested only in a few of those
    # TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    #wflw(98 landmark) trached points
    # TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    #X-Y-Z with X pointing forward and Y on the left and Z up.
    #The X-Y-Z coordinates used are like the standard
    # coordinates of ROS (robotic operative system)
    #OpenCV uses the reference usually used in computer vision:
    #X points to the right, Y down, Z to the front
    LEFT_EYEBROW_LEFT  = [6.825897, 6.760612, 4.402142]
    LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
    RIGHT_EYEBROW_RIGHT= [-6.825897, 6.760612, 4.402142]
    LEFT_EYE_LEFT  = [5.311432, 5.485328, 3.987654]
    LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
    RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
    RIGHT_EYE_RIGHT= [-5.311432, 5.485328, 3.987654]
    NOSE_LEFT  = [2.005628, 1.409845, 6.165652]
    NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
    MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
    MOUTH_RIGHT=[-2.774015, -2.080775, 5.048531]
    LOWER_LIP= [0.000000, -3.116408, 6.097667]
    CHIN     = [0.000000, -7.415691, 4.070434]

    landmarks_3D = np.float32( [LEFT_EYEBROW_LEFT,
                                LEFT_EYEBROW_RIGHT,
                                RIGHT_EYEBROW_LEFT,
                                RIGHT_EYEBROW_RIGHT,
                                LEFT_EYE_LEFT,
                                LEFT_EYE_RIGHT,
                                RIGHT_EYE_LEFT,
                                RIGHT_EYE_RIGHT,
                                NOSE_LEFT,
                                NOSE_RIGHT,
                                MOUTH_LEFT,
                                MOUTH_RIGHT,
                                LOWER_LIP,
                                CHIN])
                                
    #Return the 2D position of our landmarks
    assert landmarks_2D is not None ,'landmarks_2D is None'
    landmarks_2D = np.asarray(landmarks_2D,dtype=np.float32).reshape(-1,2)
    #Applying the PnP solver to find the 3D pose
    #of the head from the 2D position of the
    #landmarks.
    #retval - bool
    #rvec - Output rotation vector that, together with tvec, brings
    #points from the world coordinate system to the camera coordinate system.
    #tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    #Get as input the rotational vector
    #Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat,tvec))

    #euler_angles contain (pitch, yaw, roll)
    # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
    _, _, _, _, _, _,euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch,yaw,roll =map(lambda temp:temp[0],euler_angles)
    return pitch,yaw,roll


def calculate_euler_angles(landmark, attributes):
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    euler_angles_landmark = []
    for index in TRACKED_POINTS:
        euler_angles_landmark.append(landmark[index])
        euler_angles_landmark.append([landmark[index][0]*IMG_SIZE,landmark[index][1]*IMG_SIZE])

    euler_angles_landmark = np.array(euler_angles_landmark).reshape((-1, 28))
    pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0], cam_width=IMG_SIZE, cam_height=IMG_SIZE)
    euler_angles = np.array((pitch, yaw, roll), dtype=np.float32)
    euler_angles_str = " ".join(list(map(str, euler_angles)))

    landmark_str = ' '.join(list(map(str, landmark.reshape(-1).tolist())))
    attributes_str = ' '.join(list(map(str, attributes)))

    return landmark_str, attributes_str, euler_angles_str


def augment_data(image, landmark, coordinate, is_train):
    if is_train:
        transform = A.Compose([
            A.Crop(x_min=coordinate[0], y_min=coordinate[1], x_max=coordinate[2], y_max=coordinate[3], always_apply=True),
            A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            A.Rotate(limit=(-30, 30), border_mode=0, p=0.6),

            # A.OneOf([
            #     A.RandomRotate90(p=0.5),
            #     A.ShiftScaleRotate(p=0.5)
            # ], p=0.4),

            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5)
            ], p=0.7),

            # A.OneOf([
            #     A.VerticalFlip(p=0.5),
            #     A.HorizontalFlip(p=0.5)
            # ], p=0.4),

            A.Blur(blur_limit=(3, 5), p=0.3),
            A.OneOf([
                A.ChannelShuffle(p=0.2),
                A.ChannelDropout(p=0.3)
            ])

        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    else:
        transform = A.Compose([
            A.Crop(x_min=coordinate[0], y_min=coordinate[1], x_max=coordinate[2], y_max=coordinate[3], always_apply=True),
            A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),

        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    transformed = transform(image=image, keypoints=landmark)
    transformed_image, transformed_landmark = transformed['image'], transformed['keypoints']
    
    refine_shape = []
    for (x, y) in transformed_landmark:
        refine_shape.append([x, y])

    refine_shape = np.array(refine_shape, dtype=np.float32).reshape(-1, 2)
    transformed_landmark = refine_shape / IMG_SIZE

    binary = np.random.randint(0, 2)
    if binary:
        min, max, alpha = 0.2, 0.3, 1.0
        lam = np.clip(np.random.beta(alpha, alpha), min, max)

        mixup_files = sorted(glob(f"{MIXUP_DIR}/*"))
        file_idx = np.random.randint(0, len(mixup_files))
        mixup_image = cv2.imread(mixup_files[file_idx])
        mixup_image = cv2.resize(mixup_image, (IMG_SIZE, IMG_SIZE))

        transformed_image = (lam * mixup_image + (1 - lam) * transformed_image).astype(np.uint8)

    return transformed_image, transformed_landmark


def refine_bbox(image_file, landmark, is_train):
    xy = np.min(landmark, axis=0).astype(np.int32)
    zz = np.max(landmark, axis=0).astype(np.int32)
    wh = zz - xy + 1
    center = (xy + (wh / 2)).astype(np.int32)

    image = cv2.imread(f"{IMG_DIR}/{image_file}")
    height, width = image.shape[:2]
    boxsize = int(np.max(wh) * 1.2)

    xy = center - boxsize // 2
    x1, y1 = xy
    x2, y2 = xy + boxsize
    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    image, landmark = augment_data(image, landmark, [x1, y1, x2, y2], is_train)
    # visualize_sample(image, [0]*4, landmark * IMG_SIZE)
    
    return image, landmark


def processing(txt_file, folder_name, is_train):
    STEPS = 20
    if not is_train:
        STEPS = 1

    if not os.path.isdir(f"{SAVE_DIR}/{folder_name}"):
        os.makedirs(f"{SAVE_DIR}/{folder_name}/imgs")

    f_list = open(f"{SAVE_DIR}/{folder_name}/list.txt", "w")
    with open(txt_file, "r") as f:
        lines = f.readlines()

        for idx in tqdm(range(len(lines))):
            line = lines[idx]
            line = line.strip().split()
            assert(len(line) == 207)

            landmark = np.array(line[:196], dtype=np.float32).reshape(-1, 2)
            bbox = np.array(line[196:200], dtype=np.int32)
            attributes = np.array(line[200:206], dtype=np.int32)
            image_path = line[206]

            for step in range(STEPS):
                refine_image, refine_landmark = refine_bbox(image_path, landmark, is_train)
                landmark_str, attributes_str, euler_angles_str = calculate_euler_angles(refine_landmark, attributes)

                file_name = f"{idx:>06}_{step}.jpg"
                # print([f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}"])
                cv2.imwrite(f"{SAVE_DIR}/{folder_name}/imgs/{file_name}", refine_image)
                f_list.writelines(f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}\n")

    f.close()
    f_list.close()


if __name__ == "__main__":
    IMG_SIZE = 112
    SAVE_DIR = "/data/Datasets/WFLW/CUSTOM_WFLW"
    MIXUP_DIR = "/data/Datasets/Mixup_background"

    ROOT = "/data/Datasets/WFLW"
    IMG_DIR = f"{ROOT}/WFLW_images"
    ANNOTATION_DIR = f"{ROOT}/WFLW_annotations"

    train_txt = f"{ANNOTATION_DIR}/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_txt = f"{ANNOTATION_DIR}/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

    processing(train_txt, "train_data", True)
    processing(test_txt, "test_data", False)