import cv2
import imutils
import os

import numpy as np

from config import Config
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

import ipdb


def mark_countor(
    image_path,
    blur_ksize=Config.BLUR_KERNEL_SIZE,
    sobel_ksize=Config.SOBEL_KERNEL_SIZE,
    bin_threshold_1st=Config.BINARY_THRESHOLD_1ST,
    bin_threshold_2nd=Config.BINARY_THRESHOLD_2ND,
    erode_iter=Config.ERODE_ITER,
    dilate_iter=Config.DILATE_ITER,
    gse_ksize=Config.GSE_KERNEL_SIZE
):

    image = cv2.imread(image_path)
    image = ContrastImage(image).AdjustContrast()

    # a trick for tmp image
    cv2.imwrite(image_path + '.jpg', image)
    img = cv2.imread(image_path + '.jpg')
    os.remove(image_path + '.jpg')

    # convert image to gray scale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply sobel filter
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=sobel_ksize)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=sobel_ksize)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # average blur and set a threshold
    blurred = cv2.blur(gradient, (blur_ksize, blur_ksize))
    (_, thresh) = cv2.threshold(blurred, bin_threshold_1st, bin_threshold_2nd, cv2.THRESH_BINARY)

    # to fill the binary image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gse_ksize, gse_ksize))
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # erosions and dilations
    # closed = cv2.erode(closed, None, iterations=erode_iter)
    # closed = cv2.dilate(closed, None, iterations=dilate_iter)
    # cv2.imshow('test',closed)
    #  cv2.waitKey(0)
    # ipdb.set_trace()

    # to find all object contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours individually
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    # sort the contours
    (cnts, _) = contours.sort_contours(cnts)
    pixelsPerMetric = None

    return (img, cnts, pixelsPerMetric)


def visual(image, cnt, pixelsPerMetric, width):

    # compute the rotated bounding box of the contour
    original = image.copy()
    rect = cv2.minAreaRect(cnt)
    rect = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    rect = np.array(rect, dtype="int")

    # order the contour as (top-left, top-right, bottom-right, bottom-left)

    rect = perspective.order_points(rect)
    cv2.drawContours(original, [rect.astype("int")], -1, (255, 0, 0), 1)

    # loop over the original points and draw them
    for (x, y) in rect:
        cv2.circle(original, (int(x), int(y)), 2, (255, 0, 0), -1)

    (topleft, topright, bottomright, bottomleft) = rect

    def mid(A, B):
        return ((A[0] + B[0]) / 2, (A[1] + B[1]) / 2)

    (topX, topY) = mid(topleft, topright)
    (bottomX, bottomY) = mid(bottomleft, bottomright)

    # compute the midpoint respectively
    (leftX, leftY) = mid(topleft, bottomleft)
    (rightX, rightY) = mid(topright, bottomright)

    # visual the middle-point
    cv2.circle(original, (int(topX), int(topY)), 2, (255, 0, 0), -1)
    cv2.circle(original, (int(bottomX), int(bottomY)), 2, (255, 0, 0), -1)
    cv2.circle(original, (int(leftX), int(leftY)), 2, (255, 0, 0), -1)
    cv2.circle(original, (int(rightX), int(rightY)), 2, (255, 0, 0), -1)

    # visual the boject lenth and width
    cv2.line(original, (int(topX), int(topY)), (int(bottomX), int(bottomY)), (255, 0, 0), 1)
    cv2.line(original, (int(leftX), int(leftY)), (int(rightX), int(rightY)), (255, 0, 0), 1)
    # compute the 2D EU distance between the midpoints
    dA = dist.euclidean((topX, topY), (bottomX, bottomY))
    dB = dist.euclidean((leftX, leftY), (rightX, rightY))

    # calculate the pixels per metric
    if pixelsPerMetric is None:
        pixelsPerMetric = dB / width
    # calculate the size
    lenA = dA / pixelsPerMetric
    lenB = dB / pixelsPerMetric

    # present
    cv2.putText(original, "{:.2f}cm".format(lenA), (int(topX - 15), int(topY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(original, "{:.2f}cm".format(lenB), (int(rightX + 10), int(rightY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    cv2.imshow("Image", original)
    cv2.waitKey(0)


class ContrastImage(object):
    
    # initialize the object
    def __init__(self,image):
        
        self.image = image
        self.height, self.width, self.depth = self.image.shape
        
        self.minrate = Config.CONTRAST_MIN_RATE
        self.maxrate = Config.CONTRAST_MAX_RATE

    # a method to adjust contrast ratio
    def AdjustContrast(self):
        
        ones = np.ones((self.height, self.width))

        
        def ComputeMinLevel(hist):
            sum = 0
            for i in range(256):
                sum += hist[i]
                if (sum >= (self.height * self.width * self.minrate * 0.01)):
                    return i

        def ComputeMaxLevel(hist):
            sum = 0
            for i in range(256):
                sum += hist[255 - i]
                if (sum >= (self.height * self.width * self.maxrate * 0.01)):
                    return 255 - i

        for num_layer in range(self.depth):
            
            hist, _ = np.histogram(self.image[:, :, num_layer].reshape(1, self.height * self.width), bins=list(range(257)))

            minlevel = ComputeMinLevel(hist)
            maxlevel = ComputeMaxLevel(hist)
            #minlevel = np.ceil(np.percentile(hist, self.minrate))
            #maxlevel = np.floor(np.percentile(hist, self.maxrate))
            assert maxlevel > minlevel, 'MaxLevel is smaller than MinLevel'
            min_matrix = minlevel * ones
            max_matrix = maxlevel * ones
            
            # obtain bool matrix
            minbool = self.image[:, :, num_layer] < min_matrix
            maxbool = self.image[:, :, num_layer] > max_matrix

            # matrix operation
            # convert elements less than minlevel to 0
            self.image[:, :, num_layer] = (ones - (ones * minbool)) * self.image[:, :, num_layer]
            # convert elements more than maxlevel to 255
            self.image[:, :, num_layer] = ones * ~maxbool * self.image[:, :, num_layer] + ones * maxbool * 255
            # apply normalization to elements between above 2 levels eg.(x-min/max-min)
            self.image[:, :, num_layer] = (self.image[:, :, num_layer] - (~minbool * ~maxbool) * minlevel) / ((~minbool * ~maxbool) * (maxlevel - minlevel) +ones * (1 - (~minbool * ~maxbool))) * ((~minbool * ~maxbool) * 254 + ones)
        
        return self.image