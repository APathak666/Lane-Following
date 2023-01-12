import cv2
import numpy as np

def thresholding(image):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   #convert to HSV space so that inRange() function can be used
    lowerWhite = np.array([80, 0, 0])
    upperWhite = np.array([255, 160, 255])  #find these values using trackbars/trial and error
    maskWhite = cv2.inRange(imageHSV, lowerWhite, upperWhite)   #mark values in range to 255, out of range to 0

    return maskWhite

def warp(image, points, width, height):
    sourcePoints = np.float32(points)
    destPoints = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    transMatrix = cv2.getPerspectiveTransform(sourcePoints, destPoints) #get transformation matrix
    warpedImage = cv2.warpPerspective(image, transMatrix, (width, height))  #apply transformation matrix to warp image

    return warpedImage

def drawPoints(img,points):
    for x in range (0, 4):
        cv2.circle(img,(int(points[x][0]), int(points[x][1])), 15, (0, 0, 255), cv2.FILLED) #draw points on image
    return img

def getHistogram(image, frac, region = 1):
    if region == 1:
        histValues = np.sum(image, axis = 0)    #sum pixels of entire vertical
    else:
        histValues = np.sum(image[image.shape[0]//region:, :], axis = 0)    #sum pixels only upto specified fraction

    maxVal = np.max(histValues) #find maximum value
    minThresh = frac*maxVal #threshold below which value will be considered noise (certain fraction of max val)

    indexArray = np.where(histValues >= minThresh)  #store indices values which fall above threshold
    center = int(np.average(indexArray))    #find center index
    imageHist = np.zeros((image.shape[0], image.shape[1], 3), np.uint8) #initialize image of 0s with same dimensions as image

    for x, intensity in enumerate(histValues):
        if intensity > minThresh:
            color = (255, 0, 255)   #if more than minimum threshold then mark
        else:
            color = (0, 0, 255) #otherwise leave unfilled

        cv2.line(imageHist, (x, image.shape[0]), (x, int(image.shape[0] - (intensity//255//region))), color, 1) #draw lines in region
        cv2.circle(imageHist, (center, image.shape[0]), 20, (0, 255, 255), cv2.FILLED)  #plot the center index point

    return center, imageHist

# def nothing(a):
#     pass

# def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
#     cv2.namedWindow("Trackbars")
#     cv2.resizeWindow("Trackbars", 360, 240)
#     cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, nothing)
#     cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, nothing)
#     cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, nothing)
#     cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, nothing)

# def valTrackbars(wT=480, hT=240):
#     widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
#     heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
#     widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
#     heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
#     points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
#                       (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
#     return points
