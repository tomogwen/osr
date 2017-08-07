import numpy as np
import cv2


def toArray(array2d):
    truth = ~np.all(array2d == 1, axis=1)
    array2d = array2d[truth]

    truth2 = ~np.all(array2d == 1, axis=0)
    array2d = array2d[:, truth2]
    mapArray = np.ones((28,28), np.int)
    scaledownX = 28.0 / float(array2d.shape[0])
    scaledownY = 28.0 / float(array2d.shape[1])

    for i in range(array2d.shape[0]):
        for j in range(array2d.shape[1]):
            if array2d[i][j] == 0:
                mapArray[int(i * scaledownX)][int(j * scaledownY)] = 0
    print mapArray


def processImage(frame):


    range2d = np.ones((frame.shape[0], frame.shape[1]), dtype=np.int)

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i][j][0] == 0xff or frame[i][j][1] == 0xff or frame[i][j][2] == 0xff:
                range2d[i][j] = 0
    toArray(range2d)


camera = cv2.VideoCapture(0)
while 1 == 1:
    (ret, frame) = camera.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 50, 150)

    im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    for cont in contours:
        approx = cv2.approxPolyDP(cont, 0.01*cv2.arcLength(cont, True), True)
        if len(approx) == 4:
            if cv2.contourArea(cont) > 15000:
                cv2.drawContours(frame, [cont], -1, (255, 255, 255), 3)

                mask = frame.copy()
                mask[mask > 0] = 0
                cv2.fillPoly(mask, [cont], 255)
                mask = np.logical_not(mask)
                frame[mask] = 0

                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, frame)
                cv2.threshold(frame, 150, 255, cv2.THRESH_BINARY, frame)
                # processImage(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        processImage(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
