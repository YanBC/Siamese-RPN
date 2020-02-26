from net.tracker import SiamRPNTracker_cpu, SiamRPNTracker
import cv2 as cv
import argparse
from ast import literal_eval
import numpy as np


def get_options():
    p = argparse.ArgumentParser()
    p.add_argument('model', help='path to model file')
    p.add_argument('video', help='path to video file')
    p.add_argument('init_boxes', help='initial object boxes as a list (think of it as a 2D array), each element in the form of [left, top, right, bottom]')
    opts = p.parse_args()
    return opts


def get_tracker(modelPath):
    try:
        tracker = SiamRPNTracker(modelPath)
    except:
        print('CUDA unavailable...\nUsing CPU version')
        tracker = SiamRPNTracker_cpu(modelPath)
    return tracker


def write_text(image, text, left, top, right, bottom, color=(0, 255, 0)):
    ff = image.copy()
    cv.rectangle(ff, (left, top), (right, bottom), color, 3)

    if text:
        labelSize, baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(ff, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (0, 0, 255), cv.FILLED)
        cv.putText(ff, text, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 2)   

    return ff


def get_colors(num):
    assert num < 27
    condidates = [0, 127, 255]

    colors = []
    for i in range(1, num + 1):
        a = i // 9
        b = (i - a * 9) // 3
        c = i % 3
        colors.append((condidates[a], condidates[b], condidates[c]))

    return colors


def minmax2ltwh(boxes):
    transformed = []
    for box in boxes:
        left, top, right, bottom = box
        width = right - left
        height = bottom - top
        transformed.append([left, top, width, height])
    return transformed


if __name__ == '__main__':
    opts = get_options()
    modelPath = opts.model
    init_boxes = literal_eval(opts.init_boxes)
    init_boxes = minmax2ltwh(init_boxes)
    srcPath = opts.video
    videoName = 'res_' + srcPath.split('/')[-1]
    desPath = f'./{videoName}'

    num_objects = len(init_boxes)
    trackers = [get_tracker(modelPath) for _ in range(num_objects)]
    srcVideo = cv.VideoCapture(srcPath)
    desVideo = cv.VideoWriter(desPath, cv.VideoWriter_fourcc(*'XVID'), srcVideo.get(cv.CAP_PROP_FPS), (round(srcVideo.get(cv.CAP_PROP_FRAME_WIDTH)),round(srcVideo.get(cv.CAP_PROP_FRAME_HEIGHT))))

    uninitialize = [True for _ in range(num_objects)]
    box_colors = get_colors(num_objects)
    while cv.waitKey(1) < 0:
        hasFrame, frame = srcVideo.read()

        if not hasFrame:
            print('Finish Processing')
            print('Exiting ...')
            break

        canvas = frame.copy()
        for index in range(num_objects):
            if uninitialize[index]:
                trackers[index].init(frame, init_boxes[index])
                uninitialize[index] = False
            else:
                bbox, score = trackers[index].update(frame)

                left = int(bbox[0] - bbox[2] / 2)
                top = int(bbox[1] - bbox[3] / 2)
                right = int(bbox[0] + bbox[2] / 2)
                bottom = int(bbox[1] + bbox[3] / 2)

                canvas = write_text(canvas, 'object_'+str(index), left, top, right, bottom, color=box_colors[index])
            
        desVideo.write(canvas.astype(np.uint8))

