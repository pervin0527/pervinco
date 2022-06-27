import os
import cv2
import numpy as np
from glob import glob

def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(f"{path}/images")
        os.makedirs(f"{path}/img_with_pts")


def read_pts_file(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


def draw_landmark(frame, landmarks):
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius=2, color=(255, 255, 0))

    return frame


def process(folder_path):
    folder_name = folder_path.split('/')[-1]
    make_dir(f"{save_dir}/{folder_name}")

    video_file = f"{folder_path}/vid.avi"
    annotations = sorted(glob(f"{folder_path}/annot/*.pts"))

    capture = cv2.VideoCapture(video_file)
    idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break

        landmark = read_pts_file(annotations[idx])
        print(landmark)
        sample_frame = draw_landmark(frame, landmark)

        cv2.imshow("SAMPLE", sample_frame)
        cv2.waitKey(0)
        idx += 1

        cv2.imwrite(f"{save_dir}/{folder_name}/images/{idx:>06}.jpg", frame)
        cv2.imwrite(f"{save_dir}/{folder_name}/img_with_pts/{idx:>06}.jpg", sample_frame)



if __name__ == "__main__":
    save_dir = "/data/Datasets/300VW_Dataset_2015_12_14/extracted"
    data_dir = "/data/Datasets/300VW_Dataset_2015_12_14/original"
    dataset = sorted(glob(f"{data_dir}/*"))

    for folder in dataset:
        process(folder)
        break