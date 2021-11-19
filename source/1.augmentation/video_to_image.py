import os
import cv2
from glob import glob

state = "train"
root = "/data/Datasets/Seeds/SPC/2021-11-11/videos2"
videos = glob(f"{root}/*.mp4")

for video in videos:
    folder_name = video.split('/')[-1].split('.')[0]

    save_dir = f"{root}/frames/{folder_name}"
    if not os.path.isdir(save_dir):
        os.makedirs(f"{root}/frames/{folder_name}")
    
    idx = 0
    cap = cv2.VideoCapture(video)
    while True:
        ret, frame = cap.read()

        if ret == False:
            break

        idx += 1
        if state == "train":
            cv2.imwrite(f"{save_dir}/{folder_name}_{idx}.jpg", frame)

        else:
            image = cv2.resize(frame, (640, 480))
            cv2.imwrite(f"/data/Datasets/testset/SPC/{folder_name}_{idx}.jpg", image)