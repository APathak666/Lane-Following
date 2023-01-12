import cv2
import numpy as np
import matplotlib.pyplot as plt

def cannyEdge(image):
    grayLane = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #convert to grayscale
    blurLane = cv2.GaussianBlur(grayLane, (5, 5), 0) #blur image
    cannyLane = cv2.Canny(blurLane, 50, 150) #apply Canny edge detection
    return cannyLane

def region_of_interest(image):
    height = image.shape[0] #store height of image
    polygon = np.array([
        [(200, height), (1100, height), (550, 250)]
        ])                  #define edge points of triangle mask
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygon, 255)    #fill mask
    masked_image = cv2.bitwise_and(image, mask) #apply mask and isolate region of interest
    return masked_image

def show_lines(image, lines):
    lined_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(lined_image, (x1, y1), (x2, y2), (255, 0, 0), 10)  #draw line based on defined coordinates
    return lined_image

def make_coordinates(image, line_params): #find line coordinates given slope, intercept, height (assume from bottom to 3/5th of image height)
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    return np.array([x1, y1, x2, y2])

def avg_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is None:
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  #fit coordinates to line, return parameters
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:   #append to left or right side based on slope
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    avg_left_fit = np.average(left_fit, axis = 0)   #find average of left params
    avg_right_fit = np.average(right_fit, axis = 0) #find average of right params
    left_line = make_coordinates(image, avg_left_fit)   #find left coordinates
    right_line = make_coordinates(image, avg_right_fit) #find right coordinates

    return np.array([left_line, right_line])

# image = cv2.imread('test_image.jpg')
# lane = np.copy(image)

cap = cv2.VideoCapture('test2.mp4')
while (cap.isOpened()):
    _, frame = cap.read()
    cannyLane = cannyEdge(frame)    #apply Canny Edge detection to frame
    cropped_image = region_of_interest(cannyLane)   #isolate road region
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) #find lines in image using polar Hough transform
    averaged_lines = avg_slope_intercept(frame, lines)  #calculate average left and right lane line value
    lined_image = show_lines(frame, averaged_lines) #create image of only road lines 
    combined_image = cv2.addWeighted(lined_image, 1, frame, 1, 1)   #superpose lined and regular image
    cv2.imshow('result', combined_image)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
# plt.imshow(cannyLane)
# plt.show()
