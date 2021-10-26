import cv2
import matplotlib.pyplot as plt
import numpy as np


def display_img(img):
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


def corner_detector(img_col):
    img_gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    dst = cv2.GaussianBlur(img_gray, (5, 5), cv2.BORDER_DEFAULT)
    dst = cv2.cornerHarris(src=dst, blockSize=5, ksize=3, k=0.2)

    indexes = np.argwhere(dst > 0.04 * dst.max())
    var_indexes = []
    for i in range(0, len(indexes), 15):
        var_indexes.append(indexes[i].astype('float32'))

    ##  We can use commented code if you want to choose the 'best' corners, using criterias.

    # dst = cv2.dilate(dst, None)
    # ret, dst = cv2.threshold(dst, 0.05 * dst.max(), 255, 0)
    # dst = np.uint8(dst)
    #
    # ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    # corners = cv2.cornerSubPix(dst, np.float32(centroids), (5, 5), (-1, -1), criteria)

    return var_indexes


img_col = cv2.imread('plain_vision.jpg')
var_indexes = corner_detector(img_col)

for new in var_indexes:
    x_new, y_new = new[1], new[0]
    frame = cv2.circle(img_col, (int(x_new), int(y_new)),
                    3, (255, 0, 0), thickness=2)
display_img(img_col)