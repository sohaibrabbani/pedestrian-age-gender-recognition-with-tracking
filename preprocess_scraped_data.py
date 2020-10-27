#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

warnings.filterwarnings('ignore')

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    folder = "/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/scraped_images/google"
    out_folder = "/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/processed_images"
    filenames = os.listdir(folder)
    images = load_images_from_folder(folder)
    j=-1
    num = 0
    # writeVideo_flag = True
    # asyncVideo_flag = False
    #
    # file_path = 'video.webm'
    # if asyncVideo_flag:
    #     video_capture = VideoCaptureAsync(file_path)
    # else:
    #     video_capture = cv2.VideoCapture(file_path)
    #
    # if asyncVideo_flag:
    #     video_capture.start()
    #
    # if writeVideo_flag:
    #     if asyncVideo_flag:
    #         w = int(video_capture.cap.get(3))
    #         h = int(video_capture.cap.get(4))
    #     else:
    #         w = int(video_capture.get(3))
    #         h = int(video_capture.get(4))
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
    #     frame_index = -1

    # fps = 0.0
    # fps_imutils = imutils.video.FPS().start()

    for frame in images:
        j = j + 1
        # ret, frame = video_capture.read()  # frame shape 640*480*3
        # if ret != True:
        #     break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        out_filename = 1
        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            # crop_img = frame.crop([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            w = int(bbox[2])
            h = int(bbox[3])
            y = int(int(bbox[1]))
            x = int(int(bbox[0]))
            crop_img = frame[y:h, x:w]
            print(str(out_filename) + filenames[j])
            cv2.imwrite(out_folder + "/" + str(out_filename) + filenames[j], crop_img)
            num = num + 1
            out_filename = out_filename + 1
            # if len(classes) > 0:
            #     cls = det.cls
            #     cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
            #                 1e-3 * frame.shape[0], (0, 255, 0), 1)

            #crop_img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

            #res_txt = demo_par(model_par, valid_transform, crop_img)

            #draw.rectangle(xy=person_bbox[:-1], outline='red', width=1)

            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            # cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
            #             1e-3 * frame.shape[0], (0, 255, 0), 1)
            # font = ImageFont.truetype(
            #     '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/arial.ttf',
            #     size=10)

            # positive_cnt = 1
            # for txt in res_txt:
            #     if 'personal' in txt:
            #         #draw.text((x1, y1 + 20 * positive_cnt), txt, (255, 0, 0), font=font)
            #         cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1]) + 20 * positive_cnt), 0,
            #                     1e-3 * frame.shape[0], (0, 255, 0), 1)
            #         positive_cnt += 1

        # cv2.imshow('', frame)


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
