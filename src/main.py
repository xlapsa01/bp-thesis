import cv2 as cv
import numpy as np

import torch
from time import time
import argparse

parser = argparse.ArgumentParser(description="Todo")
parser.add_argument('--v', help="Path to the input video", required=True)
parser.add_argument('--s', help="Rescaling value for the input video \nDefault value is set to 1.0, meaning that the video is not rescaled", type=float, default=1.0)
parser.add_argument('--conf', help="Confidence value for the classifications \nDefault value is set to 0.5", type=float, default=0.5)
parser.add_argument('--ksize', help="Sets the kernel size used for erosion\ Default value is set to 1", type=int, default=1, choices=range(0,15))
parser.add_argument('--a', help="Sets the minimum detection area for object detection\ Default value is set to 3", type=int, default=3, choices=range(0,55))

args = parser.parse_args()

# Video Capture
capture = cv.VideoCapture(args.v)

def trackVal(x):
    pass

def trackKernel(x):
    pass

cv.namedWindow('Original')
cv.createTrackbar("Kernel Size", "Original", 1, 15, trackKernel)
cv.createTrackbar("Min Detection Area Threshold", "Original", 5, 50, trackVal)
cv.setTrackbarPos("Min Detection Area Threshold", "Original", args.a)
if args.ksize%2 == 0:
        args.ksize += 1
        cv.setTrackbarPos("Kernel Size", "Original", args.ksize)
else:
    cv.setTrackbarPos("Kernel Size", "Original", args.ksize)

# MOG2 Subtractor
mog2Subtractor = cv.createBackgroundSubtractorMOG2(200, 150, False)

color = (127, 0, 255)
model = torch.hub.load('ultralytics/yolov5','custom', '64aug.pt')
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(dev)


# Resize for Video, Image, LiveVideo
def rescaleFrame(frame, scale=1):
    """
    Function takes frame, changes the scale of it and returns it.
    :param frame: Frame used for rescaling.
    :param scale: Value of applied scale.
    :return: Rescaled frame.
    """
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def coordCalculations(rect, coords, ratio):
    """ 
    Function takes a intial bounding box coordinates, applying ratio of scale with model results, providing new values.
    :param rect: Contains initial bounding box coordinates.
    :param coords: Coordinates provided by the model inference.
    :param ratio: Ratio of the scale between 64x64 image and detected object.
    :return: Provides coordinates for bounding of the detected object within the original frame.
    """
    return_coords = [None]*4

    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]

    return_coords[0] = int(coords[0] / ratio[0] + x1)
    return_coords[1] = int(coords[1] / ratio[1] + y1)
    return_coords[2] = int(coords[2] / ratio[0] + return_coords[0])
    return_coords[3] = int(coords[3] / ratio[1] + return_coords[1])
    
    return return_coords


def plotBoxes(label, coord, conf, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param label: Contains labels predicted by model on the given frame.
    :param coord: Contains coordinates predicted by model on the given frame.
    :param conf: Contains confidance value predicted by model.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted onto it.
    """

    x1 = coord[0]
    y1 = coord[1]
    x2 = coord[2]
    y2 = coord[3]


    cv.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv.putText(frame, label + " " + str(conf.round(2)), (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return frame


def scoreFrame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        frame = frame[:, :, ::-1]
        frame = [frame]

        results = model(frame, size=64)
        df = results.pandas().xyxy[0]
        df = df[df.confidence > args.conf]
        coords = []
        for i in range(len(df.xmin.values)):
            coord = [df.xmin.values[i], df.ymin.values[i], df.xmax.values[i], df.ymax.values[i]]
            coords.append(coord)

        confidence = df.confidence.values
        labels = [df.name.values][0]
        return labels, coords, confidence


def objectDetection(frame, scale):
    """
    This function takes a frame and applies backgroundsubtraction method, followed by morphological operations.
    In order to produce detected objects.
    :param frame: Frame for object detection.
    :param scale: Rescale value of the frame. 
    :return: Function returns array of coordinates of the detected objects, 
    with actual cropped out objectes and their count. The function also returns a rescaled frame.
    """
    Original = rescaleFrame(frame, scale)

    # Get the foreground masks using all of the subtractors
    mogMask = mog2Subtractor.apply(Original)

    kernel_size = cv.getTrackbarPos("Kernel Size", "Original")

    if kernel_size%2 == 0:
        kernel_size += 1
        cv.setTrackbarPos("Kernel Size", "Original", kernel_size)

    kernel = (kernel_size, kernel_size)

    erosion = cv.morphologyEx(mogMask, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel))
    dilation = cv.dilate(erosion, (7, 7), iterations=3)
    morphed = cv.morphologyEx(dilation, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (25,25)))

    contours, hierarchies = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)

    rect_list = []
    crop_list = []
    cnt = 0

    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    for i in range(len(contours)):
        x1 = int(boundRect[i][0])
        x2 = int(boundRect[i][0]+boundRect[i][2])
        y1 = int(boundRect[i][1])
        y2 = int(boundRect[i][1]+boundRect[i][3])
        MIN = cv.getTrackbarPos('Min Detection Area Threshold', 'Original')

        area = (x2- x1) * (y2 -y1)
        if area > MIN:
            cropped = Original[y1:y2, x1:x2]
            rect_list.append([x1,y1,x2,y2])
            crop_list.append(cropped)
            cnt += 1

    return rect_list, crop_list, cnt, Original

while True:
    start_time = time()

    # Return Value and the current frame
    ret, frame = capture.read()

    # Check if a current frame actually exist
    if not ret:
        break

    rect_list, crop_list, cnt, Original = objectDetection(frame, args.s)

    for i in range(cnt):
        im = crop_list.pop(0)
        rect = rect_list.pop(0)
        ratio = [64 / im.shape[1], 64 / im.shape[0]]

        im = cv.resize(im, (64, 64), interpolation = cv.INTER_AREA)
        lbls, coords, conf = scoreFrame(im)
        if len(lbls) > 0:
            for idx, lbl in enumerate(lbls):
                output = coordCalculations(rect, coords[idx], ratio)
                plotBoxes(lbl, output, conf[idx], Original)

    end_time = time()
    fps = 1/np.round(end_time - start_time, 2)
    cv.putText(Original, f'FPS: {int(fps)}', (20,70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv.imshow('Original', Original)
    
    if cv.waitKey(20) & 0xFF==ord('d'):
        break

capture.release()
cv.destroyAllWindows()