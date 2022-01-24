import cv2
import numpy as np

WIDTH, HEIGHT = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(10, 150)


def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            # Showing the contours on the image
            # cv2.drawContours(imgContour, cnt, -1, (255, 0, 0),3)
            peri = cv2.arcLength(cnt, True)
            # Approximation of corner points
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def imgProcess(img):  # sourcery skip: inline-immediately-returned-variable
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    kernel = np.ones((5, 5), np.uint8)
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=2)
    # The final thresholded image
    imgEroded = cv2.erode(imgDilate, kernel, iterations=1)
    return imgEroded


def reOrder(myPoints):  # For making warp work properly
    # Since the points have a redundant column
    myPoints = myPoints.reshape(4, 2)
    myPointsNew = np.zeros((4, 1, 2), np.int32)

    add = myPoints.sum(1)
    # print("ADD ", add)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]

    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


def getWarpPerspective(img, biggest):
    biggest = reOrder(biggest)
    # print(biggest.shape) # (4, 1, 2)
    print(biggest)
    pts1 = np.float32(biggest)  # Source points
    pts2 = np.float32([[0, 0], [WIDTH, 0], [0, HEIGHT], [
        WIDTH, HEIGHT]])  # Destination points
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOut = cv2.warpPerspective(img, matrix, (WIDTH, HEIGHT))

    # Cropping the image for better view by removing 20 px
    imgCropped = imgOut[20:imgOut.shape[0]-20, 20:imgOut.shape[1]-20]
    imgCropped = cv2.resize(imgCropped, (WIDTH, HEIGHT))
    return imgCropped


while True:
    success, img = cap.read()
    img = cv2.resize(img, (WIDTH, HEIGHT))  # Resizing the image
    imgThreshold = imgProcess(img)
    imgContour = img.copy()
    biggest = getContours(imgThreshold)
    # print(biggest)
    if biggest.size != 0:
        imgWarpped = getWarpPerspective(img, biggest)
    else:
        imgWarpped = img
    cv2.imshow("Result", imgWarpped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
