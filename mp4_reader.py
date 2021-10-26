# this script is created for reading MP4 and showing video

import cv2

cap = cv2.VideoCapture('flying.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("1_ Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) == ord('q'):
        break

if ret is False:
    print("Can't receive frame (stream end?). Exiting ...")
    cap.release()
cv2.destroyAllWindows()

print(frame)