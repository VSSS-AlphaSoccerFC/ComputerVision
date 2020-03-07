import cv2
from threading import Thread
from RangeSelector import *
import numpy as np
from ExitWindow import *

def to255(num, val):
    x = (num*255)//val
    return x

def check_boundaries(boundaries):
    for i, color in enumerate(boundaries):
        for j, channels in enumerate(color):
            for k, val in enumerate(channels):
                if boundaries[i][j][k] < 0:
                    boundaries[i][j][k]=0
                elif boundaries[i][j][k] > 255:
                     boundaries[i][j][k]=255
                        
def color_matching(cap):

    hsv_boundaries = [[[0,0,0],[255,255,255]],  #AMARILLO
                     [[0,0,0],[255,255,255]],   #AZUL
                     [[0,0,0],[255,255,255]],   #NARANJA
                     [[0,0,0],[255,255,255]],   #ROJO
                     [[0,0,0],[255,255,255]],   #VERDE
                     [[0,0,0],[255,255,255]],   #MORADO
                     [[0,0,0],[255,255,255]],   #ROSA
                     [[0,0,0], [255,255,255]]]  #CANCHA

    rgb_boundaries = [[[0,0,0],[255,255,255]],  #AMARILLO
                     [[0,0,0],[255,255,255]],   #AZUL
                     [[0,0,0],[255,255,255]],   #NARANJA
                     [[0,0,0],[255,255,255]],   #ROJO
                     [[0,0,0],[255,255,255]],   #VERDE
                     [[0,0,0],[255,255,255]],   #MORADO
                     [[0,0,0],[255,255,255]],   #ROSA
                     [[0,0,0], [255,255,255]]]   #CANCHA

    lab_boundaries = [[[0,0,0],[255,255,255]],  #AMARILLO
                     [[0,0,0],[255,255,255]],   #AZUL
                     [[0,0,0],[255,255,255]],   #NARANJA
                     [[0,0,0],[255,255,255]],   #ROJO
                     [[0,0,0],[255,255,255]],   #VERDE
                     [[0,0,0],[255,255,255]],   #MORADO
                     [[0,0,0],[255,255,255]],   #ROSA
                     [[0,0,0], [255,255,255]]]  #CANCHA


    def position(event, x, y, flags, params):
        global cX, cY
        cX, cY = x, y

        if event == cv2.EVENT_LBUTTONDOWN:
            if code == 'h':
                hsv_boundaries[color][0][0] = hsv_img[y,x,0] - 8
                hsv_boundaries[color][1][0] = hsv_img[y,x,0] + 8
                hsv_boundaries[color][0][1] = hsv_img[y,x,1] - 40
                hsv_boundaries[color][1][1] = hsv_img[y,x,1] + 40
                hsv_boundaries[color][0][2] = hsv_img[y,x,2] - 40
                hsv_boundaries[color][1][2] = hsv_img[y,x,2] + 40

            elif code == 'r':
                rgb_boundaries[color][0][0] = rgb_img[y,x,0] - 30
                rgb_boundaries[color][1][0] = rgb_img[y,x,0] + 30
                rgb_boundaries[color][0][1] = rgb_img[y,x,1] - 30
                rgb_boundaries[color][1][1] = rgb_img[y,x,1] + 30
                rgb_boundaries[color][0][2] = rgb_img[y,x,2] - 30
                rgb_boundaries[color][1][2] = rgb_img[y,x,2] + 30

            elif code == 'l':
                lab_boundaries[color][0][0] = lab_img[y,x,0] - 100
                lab_boundaries[color][1][0] = lab_img[y,x,0] + 100
                lab_boundaries[color][0][1] = lab_img[y,x,1] - 10
                lab_boundaries[color][1][1] = lab_img[y,x,1] + 10
                lab_boundaries[color][0][2] = lab_img[y,x,2] - 10
                lab_boundaries[color][1][2] = lab_img[y,x,2] + 10


    cX = 0
    cY = 0
    code = 'h'
    color = 0
    cv2.namedWindow(winname = "Colors")
    cv2.setMouseCallback("Colors", position)

    while True:

        ret, frame = cap.read()
        external_contours = np.zeros_like(frame)
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lab_img = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        check_boundaries(hsv_boundaries)
        check_boundaries(rgb_boundaries)
        check_boundaries(lab_boundaries)

        k = cv2.waitKey(1)
        if k == 27:
            quit = exit_window()
            if quit:break
        elif k == ord("h"):
            code = 'h'        
        elif k == ord("r"):
            code = 'r'
        elif k == ord("l"):
            code = 'l'
        elif k >= 0 and chr(k).isdigit():
            if int(chr(k)) <= 7:
                color = int(chr(k))

        if color == 0:
            cv2.putText(frame, "Color: Amarillo", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[0][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[0][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[0][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[0][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[0][0], dtype = "uint8")
                upper = np.array(lab_boundaries[0][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 1:
            cv2.putText(frame, "Color: Azul", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[1][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[1][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[1][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[1][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[1][0], dtype = "uint8")
                upper = np.array(lab_boundaries[1][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 2:
            cv2.putText(frame, "Color: Naranja", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[2][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[2][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[2][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[2][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[2][0], dtype = "uint8")
                upper = np.array(lab_boundaries[2][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 3:
            cv2.putText(frame, "Color: Rojo", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[3][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[3][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[3][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[3][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[3][0], dtype = "uint8")
                upper = np.array(lab_boundaries[3][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 4:
            cv2.putText(frame, "Color: Verde", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[4][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[4][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[4][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[4][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[4][0], dtype = "uint8")
                upper = np.array(lab_boundaries[4][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 5:
            cv2.putText(frame, "Color: Morado", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[5][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[5][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[5][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[5][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[5][0], dtype = "uint8")
                upper = np.array(lab_boundaries[5][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 6:
            cv2.putText(frame, "Color: Rosa", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[6][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[6][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[6][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[6][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[6][0], dtype = "uint8")
                upper = np.array(lab_boundaries[6][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

        elif color == 7:
            frame = cv2.medianBlur(frame, 5)
            cv2.putText(frame, "Color: Cancha", org = (50, 200), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (255,255,255), thickness = 1)

            if code == 'h':
                cv2.putText(frame, f"H: {hsv_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"S: {hsv_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                cv2.putText(frame, f"V: {hsv_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = 255, thickness = 1)
                lower = np.array(hsv_boundaries[7][0], dtype = "uint8")
                upper = np.array(hsv_boundaries[7][1], dtype = "uint8")
                mask = cv2.inRange(hsv_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'r':
                cv2.putText(frame, f"R: {rgb_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"G: {rgb_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                cv2.putText(frame, f"B: {rgb_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0,0,255), thickness = 1)
                lower = np.array(rgb_boundaries[7][0], dtype = "uint8")
                upper = np.array(rgb_boundaries[7][1], dtype = "uint8")
                mask = cv2.inRange(rgb_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            elif code == 'l':
                cv2.putText(frame, f"L: {lab_img[cY,cX,0]}", org = (50, 50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"A: {lab_img[cY,cX,1]}", org = (50, 100), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                cv2.putText(frame, f"B: {lab_img[cY,cX,2]}", org = (50, 150), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (100,230,150), thickness = 1)
                lower = np.array(lab_boundaries[7][0], dtype = "uint8")
                upper = np.array(lab_boundaries[7][1], dtype = "uint8")
                mask = cv2.inRange(lab_img, lower, upper)
                result = cv2.bitwise_and(frame, frame, mask = mask)

            mask = cv2.medianBlur(mask, 5)

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            try: 

                max_lenght = cv2.arcLength(contours[0], True)
                max_contour = 0

                for i, contour in enumerate(contours):
                    
                    if hierarchy[0][i][3] == -1:          #Cheking the last element of each list in hierarchy - If it equals -1, thats means it is an external contour
                        lenght = cv2.arcLength(contours[i], True)
                        
                        if lenght > max_lenght:
                            max_lenght = lenght
                            max_contour = i

                cv2.drawContours(image = external_contours, contours = contours, contourIdx = max_contour, color = (255,255,255), thickness = 2)
        
                esquinas = [[10000,10000], [0,10000], [10000,0], [0,0]]
                conv_hull = cv2.convexHull(contours[max_contour])

                for j in range(4):
                    for i in conv_hull:
                        x, y = i.ravel()
                        if j == 0:
                            if (x +y) <= (esquinas[j][0] + esquinas[j][1]):
                                esquinas[j][0] = x
                                esquinas[j][1] = y
                        elif j == 1:
                            if y <= esquinas[j][1] and (x >= esquinas[j][0] or abs(x - esquinas[j][0]) <= 50):
                                    esquinas[j][0] = x
                                    esquinas[j][1] = y
                        elif j == 2:
                            if x <= esquinas[j][0] and (y >= esquinas[j][1] or abs(y - esquinas[j][1]) <= 50):
                                    esquinas[j][0] = x
                                    esquinas[j][1] = y
                        elif j == 3:
                            if (x +y) >= (esquinas[j][0] + esquinas[j][1]):
                                esquinas[j][0] = x
                                esquinas[j][1] = y

                cX = (esquinas[0][0] + esquinas[3][0]) // 2
                cY = (esquinas[1][1] + esquinas[2][1]) // 2
                    
                for [x, y] in esquinas:
                    cv2.circle(frame, (x,y), 5, (0,0,255), -1)

                cv2.line(frame, tuple(esquinas[0]), tuple(esquinas[3]), (255,0,0), 2)
                cv2.line(frame, tuple(esquinas[1]), tuple(esquinas[2]), (255,0,0), 2)
                cv2.circle(frame, (cX, cY), 10, (0,255,0), -1)
            
            except IndexError:
                pass


        cv2.imshow("Colors", frame)
        cv2.imshow("Mask", result)

    cv2.destroyAllWindows()
    return esquinas, hsv_boundaries, rgb_boundaries, lab_boundaries

if __name__ == '__main__':
   color_matching()
#thread1 = Thread(target=color_matching, args=(cap, values, color_code,))
#thread2 = Thread(target=range_selector, args=(color_code , values,))
#thread1.start()
#thread2.start()
#thread1.join()
#thread2.join()
