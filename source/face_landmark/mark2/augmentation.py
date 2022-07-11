import os
import cv2
import numpy as np
import albumentations as A
from glob import glob
from tqdm import tqdm

def draw_landmark(image, landmark, name):
    result = image.copy()
    for (x, y) in landmark:
        cv2.circle(result, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)    

    result = cv2.resize(result, (640, 480))
    cv2.imshow(f"{name}", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_lines(txt_file):
    file = open(txt_file, "r")
    lines = file.readlines()

    return lines


def flatten_landmark(landmark):
    flatten = []
    for point in landmark:
        x, y = point
        flatten.extend([x, y])

    return flatten


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

    landmark_ = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                             M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmark])
    return M, landmark_


def crop_face(img_path, landmark, is_train, mirror):
    image = cv2.imread(img_path)
    height, width = image.shape[:2]

    if (mirror is not None):
        with open(mirror, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            mirror_idx = lines[0].strip().split(',')
            mirror_idx = list(map(int, mirror_idx))

    xy = np.min(landmark, axis=0).astype(np.int32)
    zz = np.max(landmark, axis=0).astype(np.int32)
    wh = zz - xy + 1

    center = (xy + wh / 2).astype(np.int32)
    box_size = int(np.max(wh) * 1.2)
    xy = center - box_size // 2
    x1, y1 = xy
    x2, y2 = xy + box_size

    dx = max(0, -x1)
    dy = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)

    edx = max(0, x2 - width)
    edy = max(0, y2 - height)
    x2 = min(width, x2)
    y2 = min(height, y2)

    crop_image = image[y1:y2, x1:x2]
    crop_image = cv2.resize(crop_image, (112, 112))
    crop_landmark = (landmark - xy) / box_size

    if is_train:
        angle = np.random.randint(-30, 30)
        cx, cy = center
        cx = cx + int(np.random.randint(-box_size*0.1, box_size*0.1))
        cy = cy + int(np.random.randint(-box_size * 0.1, box_size * 0.1))
        M, landmark = rotate(angle, (cx,cy), landmark)

        image = cv2.warpAffine(image, M, (int(image.shape[1]*1.1), int(image.shape[0]*1.1)))

        wh = np.ptp(landmark, axis=0).astype(np.int32) + 1
        size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
        xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
        crop_landmark = (landmark - xy) / size

        x1, y1 = xy
        x2, y2 = xy + size
        height, width, _ = image.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        crop_image = image[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx >0 or edy > 0):
            crop_image = cv2.copyMakeBorder(crop_image, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        crop_image = cv2.resize(crop_image, (112, 112))

        if mirror is not None and np.random.choice((True, False)):
            crop_landmark[:,0] = 1 - crop_landmark[:,0]
            crop_landmark = crop_landmark[mirror_idx]
            crop_image = cv2.flip(crop_image, 1)

    return crop_image, crop_landmark


def augmentation(image, landmarks):
    augment_result = augment_transfrom(image=image, keypoints=landmarks)
    augment_image, augment_keypoints = augment_result['image'], augment_result["keypoints"]
    augment_keypoints = np.array(flatten_landmark(augment_keypoints), dtype=np.float32).reshape(-1, 2)

    return augment_image, augment_keypoints


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

    bg_files = glob(f"{MIXUP_DIR}/*")
    random_idx = np.random.randint(0, len(bg_files))
    bg_image = cv2.imread(bg_files[random_idx])
    bg_result = bg_transform(image=bg_image)
    bg_image = bg_result['image']

    result_image = (lam * bg_image + (1 - lam) * fg).astype(np.uint8)

    return result_image


def calculate_pitch_yaw_roll(landmarks_2D,
                             cam_w=256,
                             cam_h=256,
                             radians=False):
    assert landmarks_2D is not None, 'landmarks_2D is None'

    c_x = cam_w / 2
    c_y = cam_h / 2
    f_x = c_x / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x
    camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y],
                                [0.0, 0.0, 1.0]])
    camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

    landmarks_3D = np.float32([
        [6.825897, 6.760612, 4.402142],  # LEFT_EYEBROW_LEFT, 
        [1.330353, 7.122144, 6.903745],  # LEFT_EYEBROW_RIGHT, 
        [-1.330353, 7.122144, 6.903745],  # RIGHT_EYEBROW_LEFT,
        [-6.825897, 6.760612, 4.402142],  # RIGHT_EYEBROW_RIGHT,
        [5.311432, 5.485328, 3.987654],  # LEFT_EYE_LEFT,
        [1.789930, 5.393625, 4.413414],  # LEFT_EYE_RIGHT,
        [-1.789930, 5.393625, 4.413414],  # RIGHT_EYE_LEFT,
        [-5.311432, 5.485328, 3.987654],  # RIGHT_EYE_RIGHT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_LEFT,
        [-2.005628, 1.409845, 6.165652],  # NOSE_RIGHT,
        [2.774015, -2.080775, 5.048531],  # MOUTH_LEFT,
        [-2.774015, -2.080775, 5.048531],  # MOUTH_RIGHT,
        [0.000000, -3.116408, 6.097667],  # LOWER_LIP,
        [0.000000, -7.415691, 4.070434],  # CHIN
    ])
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix,
                                 camera_distortion)

    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    return map(lambda k: k[0], euler_angles)


def get_euler_angles(landmarks):
    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    assert landmarks.shape == (68, 2)

    euler_angles_landmark = []
    for index in TRACKED_POINTS:
        euler_angles_landmark.append(landmarks[index])
    
    euler_angles_landmark = np.array(euler_angles_landmark).reshape((-1, 28))
    pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
    euler_angles = np.array((pitch, yaw, roll), dtype=np.float32)

    return euler_angles


def line_process(lines, is_train, steps, save_dir, mirror=None):
    total = []
    for index in tqdm(range(len(lines))):
        line = lines[index]
        line = line.strip().split()

        image_path = f"{IMAGES}/{line[-1]}"
        labels = line[:-1]

        landmarks = np.array(labels[:136], dtype=np.float64).reshape(-1, 2)
        bboxes = np.array(labels[136:140], dtype=np.int32)
        attributes = np.array(labels[140:], dtype=np.int32)

        crop_image, crop_landmarks = crop_face(image_path, landmarks, is_train, mirror=mirror)
        crop_landmarks = crop_landmarks * [112, 112]

        if is_train:
            for step in range(steps):
                final_image, final_landmarks = augmentation(crop_image, crop_landmarks)
                
                if np.random.rand(1) > 0.5:
                    final_image = mixup(final_image, min=0.2, max=0.25)

                if not VISUALIZE:
                    final_landmarks = final_landmarks / 112
                    euler_angles = get_euler_angles(final_landmarks)

                    image_str = f"{save_dir}/imgs/{index}_{step:>06}.png"
                    landmark_str = ' '.join(list(map(str, final_landmarks.reshape(-1).tolist())))
                    attributes_str = ' '.join(list(map(str, attributes)))
                    euler_angles_str = ' '.join(list(map(str, euler_angles)))
                    
                    cv2.imwrite(image_str, final_image)
                    label = '{} {} {} {}\n'.format(image_str, landmark_str, attributes_str, euler_angles_str)
                    total.append(label)

                else:
                    draw_landmark(final_image, final_landmarks, name=image_path)

        else:
            if not VISUALIZE:
                crop_landmarks = crop_landmarks / 112
                euler_angles = get_euler_angles(crop_landmarks)

                image_str = f"{save_dir}/imgs/{index:>06}.png"
                landmark_str = ' '.join(list(map(str, crop_landmarks.reshape(-1).tolist())))
                attributes_str = ' '.join(list(map(str, attributes)))
                euler_angles_str = ' '.join(list(map(str, euler_angles)))
                
                cv2.imwrite(image_str, crop_image)
                label = '{} {} {} {}\n'.format(image_str, landmark_str, attributes_str, euler_angles_str)
                total.append(label)

            else:
                draw_landmark(crop_image, crop_landmarks, name=image_path)

    return total


def write_txt(labels, dir):
    f = open(dir, "w")
    for label in labels:
        f.writelines(label)

    f.close()


if __name__ == "__main__":
    ROOT_DIR = "/data/Datasets/WFLW"
    ANNOTATIONS = f"{ROOT_DIR}/annotations/list_68pt_rect_attr_train_test"
    IMAGES = f"{ROOT_DIR}/WFLW_images"
    
    MIXUP_DIR = "/data/Datasets/Mixup_background"
    VISUALIZE = False

    txt_files = sorted(glob(f"{ANNOTATIONS}/*.txt"))

    augment_transfrom = A.Compose([
        A.ChannelShuffle(p=0.3),
        # A.ShiftScaleRotate(shift_limit=(-0.0, 0.0), scale_limit=(-0.4, 0.0), rotate_limit=(-0, 0), border_mode=0, p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5),
        ], p=1),

        A.OneOf([
            A.Blur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 20.0), p=0.3),
            A.RandomRain(blur_value=2, p=0.3)
        ], p=0.4),

    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    for txt_file in txt_files:
        lines = get_lines(txt_file)
        
        if txt_file.split('/')[-1] == "list_68pt_rect_attr_train.txt":
            if not os.path.isdir(f"{ROOT_DIR}/augment_train_68pts"):
                os.makedirs(f"{ROOT_DIR}/augment_train_68pts/imgs")

            final_labels = line_process(lines, True, 3, f"{ROOT_DIR}/augment_train_68pts", f"{ROOT_DIR}/WFLW_annotations/Mirror68.txt")
            write_txt(final_labels, f"{ROOT_DIR}/augment_train_68pts/list.txt")
        
        else:
            if not os.path.isdir(f"{ROOT_DIR}/augment_test_68pts"):
                os.makedirs(f"{ROOT_DIR}/augment_test_68pts/imgs")

            final_labels = line_process(lines, False, 1, f"{ROOT_DIR}/augment_test_68pts")
            print(len(final_labels))
            write_txt(final_labels, f"{ROOT_DIR}/augment_test_68pts/list.txt")