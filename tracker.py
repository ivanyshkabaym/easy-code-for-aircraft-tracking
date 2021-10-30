import numpy as np
import cv2


def corner_detector(img_gray):
    dst = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)
    dst = cv2.cornerHarris(src=dst, blockSize=5, ksize=3, k=0.2)

    indexes = np.argwhere(dst > 0.04 * dst.max())
    true_indexes, j = [], 0
    for i in range(0, len(indexes), 15):
        true_indexes.append([])
        true_indexes[j].append(list([indexes[i][1], indexes[i][0]]))
        j += 1
    return np.array(true_indexes).astype('float32')


def calculate_optical_flow(old_gray, frame_gray, angle_features):
    lk_params = dict(winSize=(70, 70), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                               10, 0.03))
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                                    frame_gray,
                                                    angle_features,
                                                    None,
                                                    **lk_params)
    return nextPts, status, err


cap = cv2.VideoCapture('flying.mp4')

while cap.isOpened():
    ret, old_frame = cap.read()
    if not ret:
        print("1_ Can't receive frame (stream end?). Exiting ...")
        break
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    angle_features = corner_detector(old_gray)
    mask = np.zeros_like(old_gray)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("1_ Can't receive frame (stream end?). Exiting ...")
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nextPts, status, err = calculate_optical_flow(old_gray,
                                                      frame_gray,
                                                      angle_features)
        if nextPts is not None:
            good_new = nextPts[status == 1]
            good_old = angle_features[status == 1]
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            x_new, y_new = new.ravel()
            x_old, y_old = old.ravel()
            frame = cv2.circle(frame, (int(x_new), int(y_new)),
                               4, (255, 0, 0), thickness=2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(20) == ord('q'):
            break
        old_gray = frame_gray.copy()
        angle_features = good_new.reshape(-1, 1, 2)

if ret is False:
    print("Can't receive frame (stream end?). Exiting ...")
    cap.release()
cv2.destroyAllWindows()