import cv2

video_dir = "/data/Datasets/Seeds/DMC/samples2/video/20211027_174324.mp4"
save_dir = "/data/Datasets/Seeds/DMC/samples2/video/frames"
cap = cv2.VideoCapture(video_dir)

filename = video_dir.split('/')[-1].split('.')[0]
print(filename)

idx = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    cv2.imwrite(f"{save_dir}/{filename}_{idx}.jpg", frame)
    idx += 1