import cv2
import numpy as np

def data_preprocess(wflw, vw):
    wflw_data = wflw.split(' ')
    vw_data = vw.split(' ')
    
    wflw_image_path = wflw_data[0]
    wflw_labels = np.array(wflw_data[1:146], dtype=np.float32)
    vw_image_path = vw_data[0]
    vw_labels = np.array(vw_data[1:146], dtype=np.float32)
    # print(wflw_labels.shape)
    # print(wflw_labels)

    # landmarks = np.array(wflw_data[1:137], dtype=np.float32)
    # attributes = wflw_data[137:143]
    # yaw, pitch, roll = wflw_data[143:146]

    # print(landmarks)
    # print(attributes)
    # print(yaw, pitch, roll)

    wflw_landmarks = np.array(wflw_labels[0:136], dtype=np.float32)
    vw_landmarks = np.array(vw_labels[0:136], dtype=np.float32)
    wflw_attributes = wflw_labels[136:142]
    wflw_yaw, wflw_pitch, wflw_roll = wflw_labels[142:146]

    print(wflw_landmarks)
    print(wflw_attributes)
    print(wflw_yaw, wflw_pitch, wflw_roll)

    wflw_image = cv2.imread(wflw_image_path)
    wflw_landmarks = wflw_landmarks.reshape(-1, 2)
    wflw_landmarks  = wflw_landmarks * wflw_image.shape[0]

    vw_image = cv2.imread(vw_image_path)
    vw_landmarks = vw_landmarks.reshape(-1, 2)
    vw_landmarks = vw_landmarks * vw_image.shape[0]

    for (x1, y1), (x2, y2) in zip(wflw_landmarks, vw_landmarks):
        cv2.circle(wflw_image, (int(x1), int(y1)), radius=1, color=(255, 255, 0))
        cv2.circle(vw_image, (int(x2), int(y2)), radius=1, color=(0, 0, 255))

        wflw = cv2.resize(wflw_image, (640, 480))
        vw = cv2.resize(vw_image, (640, 480))

        cv2.imshow("wflw", wflw)
        cv2.imshow("vw300", vw)
        cv2.waitKey(0)


if __name__ == "__main__":
    data_list = "/data/Datasets/TOTAL_FACE/train_data_68pts/list.txt"
    f = open(data_list, "r")
    data = f.readlines()
    
    WFLW = data[0]
    VW_300 = data[75001]

    data_preprocess(WFLW, VW_300)
    f.close()