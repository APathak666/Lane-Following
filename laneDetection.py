import cv2
import utils

curveList = []
curveLen = 10

def getLaneCurve(image):
    imageCopy = image.copy()
    thresh = utils.thresholding(image)
    height, width, channels = image.shape
    # points = utils.valTrackbars()
    # print(points)
    points = [[102, 80], [378, 80], [20, 214], [460, 214]]
    imageWarp = utils.warp(thresh, points, width, height)
    imageWarpPoints = utils.drawPoints(imageCopy, points)

    center, histogram = utils.getHistogram(imageWarp, 0.5, 4)   #find true center
    dev, histogram = utils.getHistogram(imageWarp, 0.1) #find displaced center (with curvature)
    rawCurve = dev-center   #difference gives raw value of curvature
    curveList.append(rawCurve)

    #maintain fixed number of previous curvature values in list
    if len(curveList) > curveLen:
        curveList.pop(0)

    #moving averages to reflect more accurate curve value
    curve = int(sum(curveList)/len(curveList))
    curve /= 100

    #normalizing curve values in range [-1, 1]
    if curve > 1:
        curve = 1

    if curve < -1:
        curve = -1

    cv2.imshow('Threshold', thresh)
    cv2.imshow('Warp', imageWarp)
    cv2.imshow('Warp Points', imageWarpPoints)
    cv2.imshow('Histogram', histogram)

    return curve

if __name__ == '__main__':
    cap = cv2.VideoCapture('vid1.mp4')
    # initialTrackBarVals = [102, 80, 20, 214]
    # utils.initializeTrackbars(initialTrackBarVals)
    frameCounter = 0

    while True:
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:   #loop frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, frame = cap.read()
        frame = cv2.resize(frame, (480, 240))   #resize image
        curve = getLaneCurve(frame)
        print(curve)
        cv2.imshow('Video', frame)
        cv2.waitKey(1)
