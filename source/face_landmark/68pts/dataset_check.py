import cv2
import numpy as np

def data_preprocess(sample_data):
    data = sample_data.split(' ')
    
    image_path = data[0]
    labels = np.array(data[1:146], dtype=np.float32)
    # print(labels.shape)
    # print(labels)

    # landmarks = np.array(data[1:137], dtype=np.float32)
    # attributes = data[137:143]
    # yaw, pitch, roll = data[143:146]

    # print(landmarks)
    # print(attributes)
    # print(yaw, pitch, roll)

    landmarks = np.array(labels[0:136], dtype=np.float32)
    attributes = labels[136:142]
    yaw, pitch, roll = labels[142:146]

    print(landmarks)
    print(attributes)
    print(yaw, pitch, roll)

    image = cv2.imread(image_path)
    landmarks = landmarks.reshape(-1, 2)
    landmarks  = landmarks * image.shape[0]

    for (x, y) in landmarks:
        cv2.circle(image, (int(x), int(y)), radius=1, color=(255, 255, 0), thickness=-1)
        cv2.imshow("result", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    data_list = "/data/Datasets/TOTAL_FACE/train_data_68pts/list.txt"
    f = open(data_list, "r")
    data = f.readlines()
    
    WFLW = data[0]
    VW_300 = data[75001]

    data_preprocess(WFLW)
    data_preprocess(VW_300)

    f.close()