import os
import cv2
import numpy as np
import albumentations as A

from tqdm import tqdm
from matplotlib import pyplot as plt


def visualize_sample(image, bbox, landmark):
    if type(image) == str:
        image = cv2.imread(f"{IMG_DIR}/{image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=2)
    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 0, 0), thickness=-1)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.show()


def augment_data(image, landmark, xmin, ymin, xmax, ymax):
    transform = A.Compose([
        A.Crop(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, always_apply=True),
        A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),

        A.OneOf([
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5)
        ], p=0.4),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.5)
        ], p=0.7),

        A.OneOf([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5)
        ], p=0.4)

    ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

    transformed = transform(image=image, keypoints=landmark)
    transformed_image, transformed_landmark = transformed['image'], transformed['keypoints']
    
    refine_shape = []
    for (x, y) in transformed_landmark:
        refine_shape.append([x, y])

    refine_shape = np.array(refine_shape, dtype=np.float32).reshape(-1, 2)
    transformed_landmark = refine_shape / IMG_SIZE

    return transformed_image, transformed_landmark


def refine_bbox(image_file, landmark, augmentation):
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

    if not augmentation:
        # imgT = image[y1:y2, x1:x2]
        # imgT = cv2.resize(imgT, (IMG_SIZE, IMG_SIZE))
        # landmark = (landmark - xy) / boxsize
        # visualize_sample(imgT, [0, 0, 0, 0], landmark * IMG_SIZE)
        # return imgT, landmark

        normal_transform = A.Compose([
            A.Crop(x_min=x1, y_min=y1, x_max=x2, y_max=y2, always_apply=True),
            A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
        ], keypoint_params=A.KeypointParams(format="xy", remove_invisible=False))

        normal_transformed = normal_transform(image=image, keypoints=landmark)
        normal_transformed_image, normal_transformed_landmark = normal_transformed['image'], normal_transformed['keypoints']
        
        refine_shape = []
        for (x, y) in normal_transformed_landmark:
            refine_shape.append([x, y])

        refine_shape = np.array(refine_shape, dtype=np.float32).reshape(-1, 2)
        normal_transformed_landmark = refine_shape / IMG_SIZE

        return normal_transformed_image, normal_transformed_landmark

    else:
        image, landmark = augment_data(image, landmark, x1, y1, x2, y2)
        # visualize_sample(image, [0]*4, landmark * IMG_SIZE)
        return image, landmark


def calculate_pitch_yaw_roll(landmarks_2D, cam_width=112, cam_height=112):
    f_x = cam_width / np.tan(60 / 2 * np.pi / 180)
    f_y = f_x

    camera_matrix = np.float32([[f_x, 0.0, cam_width], [0.0, f_y, cam_height], [0.0, 0.0, 1.0]])
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
    landmarks_2D = np.array(landmarks_2D, dtype=np.float32).reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(landmarks_3D, landmarks_2D, camera_matrix, camera_distortion)
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    return map(lambda k: k[0], euler_angles)


def calculate_euler_angles(landmark, attributes):
    TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    euler_angles_landmark = []
    for index in TRACKED_POINTS:
        euler_angles_landmark.append(landmark[index])

    euler_angles_landmark = np.array(euler_angles_landmark).reshape((-1, 28))
    pitch, yaw, roll = calculate_pitch_yaw_roll(euler_angles_landmark[0])
    euler_angles = np.array((pitch, yaw, roll), dtype=np.float32)
    euler_angles_str = " ".join(list(map(str, euler_angles)))

    landmark_str = ' '.join(list(map(str, landmark.reshape(-1).tolist())))
    attributes_str = ' '.join(list(map(str, attributes)))

    return landmark_str, attributes_str, euler_angles_str


def train_processing(txt_file, folder_name):
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

            for step in range(10):
                refine_image, refine_landmark = refine_bbox(image_path, landmark, augmentation=True)
                landmark_str, attributes_str, euler_angles_str = calculate_euler_angles(refine_landmark, attributes)

                file_name = f"{idx:>06}_{step}.jpg"
                # print([f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}"])
                cv2.imwrite(f"{SAVE_DIR}/{folder_name}/imgs/{file_name}", refine_image)
                f_list.writelines(f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}\n")

    f.close()
    f_list.close()


def valid_processing(txt_file, folder_name):
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

            refine_image, refine_landmark = refine_bbox(image_path, landmark, augmentation=False)
            landmark_str, attributes_str, euler_angles_str = calculate_euler_angles(refine_landmark, attributes)

            file_name = f"{idx:>06}.jpg"
            # print([f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}"])
            cv2.imwrite(f"{SAVE_DIR}/{folder_name}/imgs/{file_name}", refine_image)
            f_list.writelines(f"{SAVE_DIR}/{folder_name}/imgs/{file_name} {landmark_str} {attributes_str} {euler_angles_str}\n")

    f.close()
    f_list.close()


if __name__ == "__main__":
    STEPS = 10
    IMG_SIZE = 112
    SAVE_DIR = "/data/Datasets/CUSTOM_WFLW"

    ROOT = "/data/Datasets/WFLW"
    IMG_DIR = f"{ROOT}/WFLW_images"
    ANNOTATION_DIR = f"{ROOT}/WFLW_annotations"

    train_txt = f"{ANNOTATION_DIR}/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt"
    test_txt = f"{ANNOTATION_DIR}/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt"

    train_processing(train_txt, "train_data")
    valid_processing(test_txt, "test_data")