import os
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm


def draw_landmarks(image, landmarks, idx):
    sample_image = image.copy()
    for x, y in landmarks:
        cv2.circle(sample_image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imwrite(f"{save_dir}/imgs/sample_{idx}.jpg", sample_image)


def flatten_landmark(landmark):
    flatten = []
    for point in landmark:
        x, y = point
        flatten.extend([x, y])

    return flatten


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


def get_euler_angles(landmark):
    assert landmark.shape == (68, 2)

    TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
    
    euler_angles_landmark = []
    for index in TRACKED_POINTS:
        euler_angles_landmark.append(landmark[index])

    euler_angles_landmark = np.array(euler_angles_landmark).reshape((-1, 28))
    pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
    euler_angles = np.array((pitch, yaw, roll), dtype=np.float32)

    return euler_angles


def crop_face(image_path, landmark):
    xy = np.min(landmark, axis=0).astype(np.int32)
    zz = np.max(landmark, axis=0).astype(np.int32)
    wh = zz - xy + 1

    center = (xy + wh/2).astype(np.int32)
    img = cv2.imread(f"{image_dir}/{image_path}")
    boxsize = int(np.max(wh)*1.2)
    xy = center - boxsize//2
    x1, y1 = xy
    x2, y2 = xy + boxsize
    height, width, _ = img.shape

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width, x2)
    y2 = min(height, y2)

    crop_transform = A.Compose([
        A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, p=1),
        A.Resize(image_size, image_size, p=1)
    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    cropped_transform = crop_transform(image=img, keypoints=landmark)
    c_image, c_landmark = cropped_transform["image"], cropped_transform["keypoints"]

    return c_image, c_landmark


def augmentation(image, keypoints, transform):
    transformed = transform(image=image, keypoints=keypoints)
    transformed_image, transformed_keypoints = transformed["image"], transformed["keypoints"]
    transformed_keypoints = np.array(flatten_landmark(transformed_keypoints)).reshape(-1, 2)
    
    return transformed_image, transformed_keypoints


def make_label(save_dir, file_name, image, landmark, attributes):
    image_path = f"{file_name}.png"
    landmark_str = ' '.join(list(map(str, landmark.reshape(-1).tolist())))

    attributes = np.array(attributes, dtype=np.int32)
    attributes_str = ' '.join(list(map(str, attributes)))

    euler_angle = get_euler_angles(landmark)
    euler_angles_str = ' '.join(list(map(str, euler_angle)))
    label = '{} {} {} {}\n'.format(f"{save_dir}/imgs/{image_path}", landmark_str, attributes_str, euler_angles_str)
    cv2.imwrite(f"{save_dir}/imgs/{image_path}", image)

    return label


def write_txt(save_dir, labels):
    f = open(f"{save_dir}/list.txt", "w")
    for label in labels:
        f.writelines(label)


def read_txt(txt_file, is_train, transform=None):
    if is_train:
        output_dir = f"{save_dir}/train"

    else:
        output_dir = f"{save_dir}/test"

    if not os.path.isdir(f"{output_dir}"):
        os.makedirs(f"{output_dir}/imgs")

    f = open(txt_file, "r")
    lines = f.readlines()

    labels = []
    # for idx, line in enumerate(lines):
    for idx in tqdm(range(len(lines))):
        line = lines[idx]
        line = line.strip().split()
        assert(len(line) == 147)

        landmark = np.array(line[:136], dtype=np.float32).reshape(-1, 2)
        bbox = np.array(line[136:140], dtype=np.int32)

        flag = list(map(int, line[140:146]))
        flag = list(map(bool, flag))
        pose = flag[0]
        expression = flag[1]
        illumination = flag[2]
        make_up = flag[3]
        occlusion = flag[4]
        blur = flag[5]

        image_path = line[146]

        image, landmark = crop_face(image_path, landmark)
        if transform != None:
            for step in range(total_step):
                image, landmark = augmentation(image, landmark, transform)
                landmark = landmark / image_size
                # draw_landmarks(image, (landmark * image_size), step)
                label = make_label(output_dir, f"{idx}_{step:>06}", image, landmark, [pose, expression, illumination, make_up, occlusion, blur])
                labels.append(label)

        else:
            landmark = np.array(flatten_landmark(landmark)).reshape(-1, 2)
            landmark = landmark / image_size
            label = make_label(output_dir, f"{idx:>06}", image, landmark, [pose, expression, illumination, make_up, occlusion, blur])
            labels.append(label)
        # break

    write_txt(output_dir, labels)


if __name__ == '__main__':
    total_step = 10
    image_size = 112
    root_dir = "/home/ubuntu/Datasets/WFLW"
    save_dir = "/home/ubuntu/Datasets/WFLW/custom"
    image_dir = f'{root_dir}/WFLW_images'

    # landmarkDirs = [f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt',
    #                 f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt']

    train_transform = A.Compose([
        A.Rotate(limit=(-30, 30), p=0.8),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=(-.15, .15), contrast_limit=(-.15, .15)),
            A.HueSaturationValue(p=0.5, hue_shift_limit=(-.15, .15), sat_shift_limit=(-.15, .15), val_shift_limit=(.10, .10))
        ], p=1),

        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MultiplicativeNoise(p=0.5)
        ], p=0.4),

    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))


    train_txt = f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_train.txt'
    read_txt(train_txt, True, train_transform)

    test_txt = f'{root_dir}/annotations/list_68pt_rect_attr_train_test/list_68pt_rect_attr_test.txt'
    read_txt(test_txt, False)