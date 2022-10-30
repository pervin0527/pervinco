import os
import cv2
from glob import glob
from tqdm import tqdm

def split_frames(path):
    files = sorted(glob(f"{path}/*.mp4"))
    for index in tqdm(range(len(files))):
        file = files[index]
        name = file.split('/')[-1].split('.')[0]

        if not os.path.isdir(f"{save_path}/{name}"):
            os.makedirs(f"{save_path}/{name}")
        
        capture = cv2.VideoCapture(file)
        index = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
            cv2.imwrite(f"{save_path}/{name}/frame_{index:>06}.jpg", frame)
            index += 1


if __name__ == "__main__":
    video_path = "/data/Datasets/BR/videos"
    save_path = "/data/Datasets/BR/frames"
    split_frames(video_path)