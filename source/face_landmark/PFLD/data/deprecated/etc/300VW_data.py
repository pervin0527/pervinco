import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm

def flatten(landmarks):
    points = ""
    for x, y in landmarks:
        points += f"{x} {y} "

    return points


def make_dir(path):
    if not os.path.isdir((f"{path}/images")):
        os.makedirs(f"{path}/images")
        os.makedirs(f"{path}/img_with_pts")


def read_pts_file(filename):
    return np.loadtxt(filename, comments=("version:", "n_points:", "{", "}"))


def draw_landmark(frame, landmarks):
    for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), radius=2, color=(255, 255, 0))

    return frame


def process(folder_path, f, save_path):
    folder_name = folder_path.split('/')[-1]
    video_file = f"{folder_path}/vid.avi"
    annotations = sorted(glob(f"{folder_path}/annot/*.pts"))

    capture = cv2.VideoCapture(video_file)

    idx = 0
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        
        image = frame.copy()
        landmark = read_pts_file(annotations[idx])

        xy = np.min(landmark, axis=0).astype(np.int32) 
        zz = np.max(landmark, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        boxsize = int(np.max(wh)*1.2)
        xy = center - boxsize//2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = image.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        image = image[y1:y2, x1:x2]
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        landmark = (landmark - xy)/boxsize
        cv2.imwrite(f"{save_path}/images/{folder_name}_{idx}.jpg", image)

        sample_image = draw_landmark(image, landmark*IMG_SIZE)
        cv2.imwrite(f"{save_path}/img_with_pts/{folder_name}_{idx}.jpg", sample_image)
        landmark = flatten(landmark)

        f.writelines(f"{save_path}/images/{folder_name}_{idx}.jpg {landmark}\n")
        idx += 1

        # if idx == 5:
        #     break


def start(path, dataset):
    make_dir(path)
    f = open(f"{path}/list.txt", "w")
    for index in tqdm(range(len(dataset))):
        process(dataset[index], f, path)
        # break


if __name__ == "__main__":
    IMG_SIZE = 64
    save_dir = "/data/Datasets/300VW_Dataset_2015_12_14/extracted"
    data_dir = "/data/Datasets/300VW_Dataset_2015_12_14/original"
    dataset = sorted(glob(f"{data_dir}/*"))[:-2]
    
    train_data = dataset[:-3]
    test_data = dataset[-3:]

    start(f"{save_dir}/train", train_data)
    start(f"{save_dir}/test", test_data)