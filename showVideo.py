import numpy as np
import cv2

def showVideo():
    try:
        cap = cv2.VideoCapture(0)
    except:
        return
    cap.set(3, 480)
    cap.set(5, 320)

    while True:
        ret, frame = cap.read()

        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS_FULL)
        cv2.imshow('video', gray)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

showVideo()