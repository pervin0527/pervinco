import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
from glob import glob

def make_savedir(path):
    if not os.path.isdir(path):
        os.makedirs(f"{path}/imgs")


def calculate_pitch_yaw_roll(landmarks_2D, cam_w=256, cam_h=256, radians=False):
    c_x = cam_w/2
    c_y = cam_h/2
    f_x = c_x / np.tan(60/2 * np.pi / 180)
    f_y = f_x

    camera_matrix = np.float32([[f_x, 0.0, c_x],
                                [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])

    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

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

    landmarks_3D = np.float32([LEFT_EYEBROW_LEFT,
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

    assert landmarks_2D is not None, 'landmarks_2D is None'
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                      landmarks_2D,
                                      camera_matrix,
                                      camera_distortion)

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = map(lambda temp: temp[0], euler_angles)
    return pitch, yaw, roll


def visualize(image, landmarks):
    sample_image = image.copy()

    for x, y in landmarks:
        cv2.circle(sample_image, (int(x), int(y)), radius=2, thickness=-1, color=(0, 0, 255))

    cv2.imshow("vis", sample_image)
    cv2.waitKey(0)


def augmentation(image, keypoints):
    transformed = transform(image=image, keypoints=keypoints)
    transformed_image = transformed["image"]
    transformed_keypoints = transformed["keypoints"]
    
    flatten = []
    for (x, y) in transformed_keypoints:
        flatten.extend([x, y])

    transformed_keypoints = np.array(flatten).reshape(-1, 2)

    return transformed_image, transformed_keypoints


def get_data(line):
    line = line.strip().split()
    image_path = line[0]
    labels = line[1:]

    landmarks = np.array(labels[:136], dtype=np.float32).reshape(-1, 2) * [112, 112]
    attributes = np.array(labels[136:142], dtype=np.float32)
    euler_angles = np.array(labels[142:], dtype=np.float32)

    return image_path, landmarks, attributes, euler_angles


def get_euler_angles(landmark):
    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    euler_angles_landmark = []

    for index in TRACKED_POINTS:
        euler_angles_landmark.append(landmark[index])

    euler_angles_landmark = np.asarray(euler_angles_landmark).reshape((-1, 28))
    pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
    euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)

    return euler_angles


def mixup(fg, min=0.4, max=0.5, alpha=1.0):
    fg_height, fg_width = fg.shape[:2]
    lam = np.clip(np.random.beta(alpha, alpha), min, max)

    bg_transform = A.Compose([
        A.Resize(width=fg_width, height=fg_height, always_apply=True),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), p=0.3),
            A.HueSaturationValue(val_shift_limit=(40, 80), p=0.3),
            A.ChannelShuffle(p=0.3)
        ], p=1),
    ])

    bg_files = glob(f"{bg_dir}/*")
    random_idx = np.random.randint(0, len(bg_files))
    bg_image = cv2.imread(bg_files[random_idx])
    bg_result = bg_transform(image=bg_image)
    bg_image = bg_result['image']

    result_image = (lam * bg_image + (1 - lam) * fg).astype(np.uint8)

    return result_image


def read_txt(txt_path):
    f = open(txt_path, "r")
    lines = f.readlines()

    labels = []
    for idx in tqdm(range(len(lines))):
        line = lines[idx]
        image_path, landmarks, attributes, euler_angles = get_data(line)
        
        image = cv2.imread(image_path)
        # visualize(image, landmarks)

        for rpt in range(repeat):
            augment_image, augment_landmarks = augmentation(image, landmarks)
            
            if np.random.rand(1) > 0.5:
                augment_image = mixup(augment_image, min=0.2, max=0.4)

            euler_angles = get_euler_angles(augment_landmarks)
            # visualize(augment_image, augment_landmarks)

            attributes_str = ' '.join(list(map(str, attributes)))
            landmark_str = ' '.join(list(map(str, augment_landmarks.reshape(-1).tolist())))
            euler_angles_str = ' '.join(list(map(str, euler_angles)))

            cv2.imwrite(f"{SAVE_DIR}/imgs/{idx}_{rpt}.png", augment_image)
            label = f"{SAVE_DIR}/imgs/{idx}_{rpt}.png {landmark_str} {attributes_str} {euler_angles_str}\n"
            labels.append(label)

    return lines, labels


def write_total(origin, new):
    f = open(f"{SAVE_DIR}/list.txt", "w")
    total = origin.extend(new)

    for line in total:
        f.writelines(line)


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/WFLW"
    FILE_LIST = f"{ROOT_DIR}/train_data_68pts/list.txt"
    SAVE_DIR = f"{ROOT_DIR}/augment_data_68pts"

    repeat = 2
    bg_dir = "/data/Datasets/Mixup_background"

    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),

        A.Blur(blur_limit=(3, 5), p=0.4),
        A.ChannelShuffle(p=0.5),
        
        # A.CoarseDropout(max_holes=1, max_height=112//2, max_width=112//2, min_height=112//4, min_width=112//4, p=1),
        A.ShiftScaleRotate(rotate_limit=(0, 0), p=1)

    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    make_savedir(SAVE_DIR)
    original_list, new_list = read_txt(FILE_LIST)
    write_total(original_list, new_list)