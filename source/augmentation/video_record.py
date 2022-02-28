import cv2

cap = cv2.VideoCapture(-1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
fps = 30

# fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
fcc = cv2.VideoWriter_fourcc(*'DIVX')

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
out = cv2.VideoWriter('/data/Datasets/SPC-Hanam/Pikachu.avi', fcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    
    if ret:
        cv2.imshow('divx', frame)
        out.write(frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()