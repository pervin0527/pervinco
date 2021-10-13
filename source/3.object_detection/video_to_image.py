import cv2

video_dir = "/data/Datasets/Seeds/DMC/samples/sample_video_1.mp4"
save_dir = "/data/Datasets/Seeds/DMC/samples/images"
cap = cv2.VideoCapture(video_dir)


idx = 0
while True:
    ret, frame = cap.read()

    if ret == False:
        break

    # frame = cv2.resize(frame, (1080, 1920))
    cv2.imwrite(f"{save_dir}/image_{idx}.jpg", frame)
    idx += 1