from Robot import Robot
from Ball import Ball
from ColorMatching import *
import cv2
import numpy as np
import socket
import threading
from object_pb2 import ObjectData
import time



cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

esquinas , *boundaries = color_matching(cap)

ball = Ball(boundaries)
robot = Robot('red', 'blue', boundaries)

socket = socket.socket()
socket.connect(('192.168.0.17', 4000))

#thread1 = Thread(target=, args=())
c = 0

while True:

    ret, frame = cap.read()

    robot.find(frame, 'l')
    robot.show(frame)

    ball.find(frame, 'l')
    ball.show(frame)

    if c % 5 == 0:

        robot_data = ObjectData()
        robot_data.kind = 1
        robot_data.id = 0
        robot_data.team = 1
        robot_data.x = int(robot.position[0])
        robot_data.y = int(robot.position[1])
        robot_data.yaw = int(robot.angle)
        socket.sendall(robot_data.SerializeToString())

    k = cv2.waitKey(1)
    if k == 27:
        break

    cv2.imshow('frame', frame)

    c += 1

cap.release()
cv2.destroyAllWindows()