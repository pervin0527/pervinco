from attr import attributes
import cv2
import numpy as np

def draw_landmarks(image, landmarks):
    sample_image = image.copy()
    sample_landmarks = list(landmarks.copy())

    for (x, y) in sample_landmarks:
        cv2.circle(sample_image, (int(x), int(y)), radius=1, color=(0, 0, 255), thickness=-1)

    sample_image = cv2.resize(sample_image, (640, 480))
    cv2.imshow("sample_image", sample_image)
    cv2.waitKey(0)


def check_data(data):
    image_path = data[0]
    labels = data[1:146]

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    landmarks = np.array(labels[:136], np.float32).reshape(-1, 2)
    attributes = np.array(labels[136:142], np.float32)
    euler_angles = np.array(labels[142:146], np.float32)
    
    print(image_path)
    draw_landmarks(image, landmarks * height)


if __name__ == "__main__":
    data_list = "/data/Datasets/WFLW/custom2/train_data_68pts/list.txt"
    f = open(data_list, "r")
    lines = f.readlines()

    for line in lines:
        line = line.strip().split()
        check_data(line)