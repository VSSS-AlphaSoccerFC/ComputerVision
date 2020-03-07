import cv2
import numpy as np

class Ball():
    def __init__(self, boundaries):
        self.x = None
        self.y = None
        self.boundaries = boundaries

    def find(self, img, color_code):
            kernel = np.ones(shape = (2,2), dtype = np.uint8)
            if color_code == 'h':
                selected_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                selected_boundaries = self.boundaries[0]

            elif color_code == 'r':
                selected_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                selected_boundaries = self.boundaries[1]

            elif color_code == 'l':
                selected_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                selected_boundaries = self.boundaries[2]
            
            lower = np.array(selected_boundaries[2][0], dtype = "uint8")
            upper = np.array(selected_boundaries[2][1], dtype = "uint8")
            
            mask = cv2.inRange(selected_img, lower, upper)
            result = cv2.bitwise_and(img, img, mask = mask)
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

            ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

            eroded = cv2.erode(thresh, kernel, iterations = 1)
            dilated = cv2.dilate(eroded, kernel, iterations = 3)

            contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                for i in contours:
                
                    conv_hull = cv2.convexHull(i)

                    top    = tuple(conv_hull[conv_hull[:,:,1].argmin()][0])
                    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0])
                    left   = tuple(conv_hull[conv_hull[:,:,0].argmin()][0])
                    right  = tuple(conv_hull[conv_hull[:,:,0].argmax()][0])

                    cX = (left[0] + right[0]) // 2
                    cY = (top[1] + bottom[1]) // 2

                    self.x = cX
                    self.y = cY
            
    def show(self, img):
        cv2.circle(img, (self.x, self.y), 2, (255,255,255), -1)