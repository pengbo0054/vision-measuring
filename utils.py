import cv2
import imutils

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

    image = cv2.imread(image_path, 1)
    image = CreateNewImg(image)
    cv2.imshow('test',image/255)
    cv2.waitKey(0)
    # convert image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply sobel filter
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=sobel_ksize)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=sobel_ksize)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # average blur and set a threshold
    blurred = cv2.blur(gradient, (blur_ksize, blur_ksize))
    (_, thresh) = cv2.threshold(blurred, bin_threshold_1st, bin_threshold_2nd, cv2.THRESH_BINARY)
    
    ## to fill the binary image
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gse_ksize, gse_ksize))
    #closed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    ## erosions and dilations
    #closed = cv2.erode(closed, None, iterations=erode_iter)
    #closed = cv2.dilate(closed, None, iterations=dilate_iter)
    #cv2.imshow('test',closed)
    #cv2.waitKey(0)
    #ipdb.set_trace()

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
    cv2.drawContours(original, [rect.astype("int")], -1, (0, 255, 0), 2)

    # loop over the original points and draw them
    for (x, y) in rect:
        cv2.circle(original, (int(x), int(y)), 5, (0, 0, 255), -1)

    # unpack the ordered bounding box, then compute the midpoint
    # between the top-left and top-right coordinates, followed by
    # the midpoint between bottom-left and bottom-right coordinates
    (topleft, topright, bottomright, bottomleft) = rect

    def mid(A, B):
        return ((A[0] + B[0])/2, (A[1] + B[1])/2)

    (topX, topY) = mid(topleft, topright)
    (bottomX, bottomY) = mid(bottomleft, bottomright)

    # compute the midpoint respectively
    (leftX, leftY) = mid(topleft, bottomleft)
    (rightX, rightY) = mid(topright, bottomright)

    # visual the middle-point
    cv2.circle(original, (int(topX), int(topY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(bottomX), int(bottomY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(leftX), int(leftY)), 5, (255, 0, 0), -1)
    cv2.circle(original, (int(rightX), int(rightY)), 5, (255, 0, 0), -1)

    # visual the boject lenth and width
    cv2.line(original, (int(topX), int(topY)), (int(bottomX), int(bottomY)),
        (255, 0, 255), 2)
    cv2.line(original, (int(leftX), int(leftY)), (int(rightX), int(rightY)),
        (255, 0, 255), 2)
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
    cv2.putText(original, "{:.2f}cm".format(lenA),
        (int(topX - 15), int(topY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
    cv2.putText(original, "{:.2f}cm".format(lenB),
        (int(rightX + 10), int(rightY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)

    cv2.imshow("Image", original)
    cv2.waitKey(0)


def ComputeHist(img):
    h,w = img.shape
    hist, bin_edge = np.histogram(img.reshape(1,w*h), bins=list(range(257)))
    return hist
    
def ComputeMinLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[i]
        if (sum >= (pnum * rate * 0.01)):
            return i
            
def ComputeMaxLevel(hist, rate, pnum):
    sum = 0
    for i in range(256):
        sum += hist[255-i]
        if (sum >= (pnum * rate * 0.01)):
            return 255-i
            
def LinearMap(minlevel, maxlevel):
    if (minlevel >= maxlevel):
        return []
    else:
        newmap = np.zeros(256)
        for i in range(256):
            if (i < minlevel):
                newmap[i] = 0
            elif (i > maxlevel):
                newmap[i] = 255
            else:
                newmap[i] = (i-minlevel)/(maxlevel-minlevel) * 255
        return newmap
        
def CreateNewImg(img):
    h,w,d = img.shape
    newimg = np.zeros([h,w,d])
    for i in range(d):
        imgmin = np.min(img[:,:,i])
        imgmax = np.max(img[:,:,i])
        imghist = ComputeHist(img[:,:,i])
        minlevel = ComputeMinLevel(imghist, 85, h*w)
        maxlevel = ComputeMaxLevel(imghist, 4, h*w)
        newmap = LinearMap(minlevel,maxlevel)
        # print(minlevel, maxlevel)
        if (newmap.size ==0 ):
            continue
        for j in range(h):
            newimg[j,:,i] = newmap[img[j,:, i]]
    return newimg