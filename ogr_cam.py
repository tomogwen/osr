import numpy as np
import cv2

camera = cv2.VideoCapture(0)


def toArray(frame):
    print frame.shape[0]
    print frame.shape[1]


found = 0
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
                cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY, frame)


                toArray(frame)

    cv2.imshow('frame', frame)


    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

camera.release()