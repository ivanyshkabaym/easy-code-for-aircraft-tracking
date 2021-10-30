import cv2
import numpy as np


def corner_detector(img_col):
    img_gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    dst = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)
    dst = cv2.cornerHarris(src=dst, blockSize=5, ksize=3, k=0.2)

    indexes = np.argwhere(dst > 0.04 * dst.max())
    var_indexes = []
    for i in range(0, len(indexes), 15):
        var_indexes.append(indexes[i].astype('float32'))

    ##  We can use commented code if you want to choose the 'best' corners, using special criterias.

    # dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst, 0.05 * dst.max(), 255, 0)
    # dst = np.uint8(dst)
    #
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    # corners = cv2.cornerSubPix(dst, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return var_indexes


# cap = cv2.VideoCapture('flying.mp4')
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         print("1_ Can't receive frame (stream end?). Exiting ...")
#         break
#     true_indexes = corner_detector(frame)
#     for new in true_indexes:
#         x_new, y_new = new[1], new[0]
#         frame = cv2.circle(frame, (int(x_new), int(y_new)),
#                            3, (255, 0, 0), thickness=2)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(20) == ord('q'):
#         break
#
# if ret is False:
#     print("Can't receive frame (stream end?). Exiting ...")
#     cap.release()
# cv2.destroyAllWindows()


## If you want to experiment with corner detector for image, you can uncomment low code, having commented code for videostream

# img_col = cv2.imread('plain_vision.jpg')
# true_indexes = corner_detector(img_col)
# for new in true_indexes:
#     x_new, y_new = new[1], new[0]
#     img_col = cv2.circle(img_col, (int(x_new), int(y_new)),
#                     3, (255, 0, 0), thickness=2)
# cv2.imshow('frame', img_col)
# key = cv2.waitKey()

