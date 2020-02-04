#!/usr/bin/env python3

import cv2
import numpy as np
import datetime
import argparse
import imutils
from imutils.video import VideoStream

from utils import detector_utils as detector_utils

def Wbalance(img):
    width = img.shape[1]
    height = img.shape[0]
    dst = np.zeros(img.shape,img.dtype)

    #1.计算三通道灰度平均值
    imgB = img[:,:,0]
    imgG = img[:,:,1]
    imgR = img[:,:,2]
    bAve = cv2.mean(imgB)[0]
    gAve = cv2.mean(imgG)[0]
    rAve = cv2.mean(imgR)[0]
    aveGray = (int)(bAve + gAve + rAve)/3

    #2计算每个通道的增益系数
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve

    #3使用增益系数
    imgB = np.floor((imgB * bCoef)) #向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    #4将数组元素后处理
    maxB = np.max(imgB)
    minB = np.min(imgB)
    maxG = np.max(imgG)
    minG = np.min(imgG)
    maxR = np.max(imgR)
    minR = np.min(imgR)
    for i in range(0,height):
        for j in range(0,width):
            imgb = imgB[i, j]
            imgg = imgG[i, j]
            imgr = imgR[i, j]
            if imgb > 255:
                imgb = 255
            if imgg > 255:
                imgg = 255
            if imgr > 255:
                imgr = 255
            dst[i,j] = (imgb,imgg,imgr)
    return dst


ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()


if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.20

    # Get stream from webcam and set parameters)
    #vs = VideoStream().start()

    # max number of hands we want to detect/track
    num_hands_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)

    cap = cv2.VideoCapture(0)
    try:
        while True:
            # Read Frame and process
            ret, frame = cap.read()
            #img = Wbalance(frame)
            #print(frame)
            #frame = vs.read()
            frame = cv2.resize(frame, (320*2, 240*2))

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                #cv2.imshow('gray',img)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
