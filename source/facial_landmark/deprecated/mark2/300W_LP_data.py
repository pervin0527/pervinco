import cv2
import numpy as np
from scipy import io
from glob import glob


def visualize(image_file, landmark):
    image = cv2.imread(image_file)
    # landmark = (landmark * [input_size / 2, input_size / 2]) + [input_size / 2, input_size / 2]
    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=3)

    cv2.imshow("result", image)
    cv2.waitKey(0)


def read_mat(file_path):
    matfile = io.loadmat(file_path)
    print(matfile)
    label = matfile["pts_3d"]
    label = np.array(np.transpose(label)).astype("float").reshape(-1, 2)
    # norm = [input_size / 2, input_size / 2]
    # label = ((matfile['pts_2d'] - norm) / norm).astype(np.float32)

    print(label.shape)
    return label


if __name__ == "__main__":
    show_sample = True
    input_size = 368
    image_dir = "/data/Datasets/300W_LP/AFW"
    landmark_dir = "/data/Datasets/300W_LP/landmarks/AFW"

    image_files, landmark_files = sorted(glob(f"{image_dir}/*.jpg")), sorted(glob(f"{landmark_dir}/*.mat"))
    print(len(image_files), len(landmark_files))

    for image_file, landmark_file in zip(image_files, landmark_files):
        image = cv2.imread(image_file)
        landmark = read_mat(landmark_file)

        if show_sample:
            visualize(image_file, landmark)
