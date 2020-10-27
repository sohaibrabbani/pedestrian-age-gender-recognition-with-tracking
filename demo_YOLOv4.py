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
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

import torch
import torchvision.transforms as T
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50
import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw

warnings.filterwarnings('ignore')

pa100k_values = ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Front', 'Side', 'Back']
# values = ['accessoryHat', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'hairLong', 'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt', 'lowerBodyTrousers', 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker', 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'personalMale']
values = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

def model_init_par():
    # model
    backbone = resnet50()
    classifier = BaseClassifier(nattr=len(values))
    model = FeatClassifier(backbone, classifier)

    # load
    checkpoint = torch.load(
        '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/exp_result/custom/custom/img_model/ckpt_max.pth')
    # unfolded load
    # state_dict = checkpoint['state_dicts']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # one-liner load
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model).cuda()
    #     model.load_state_dict(checkpoint['state_dicts'])
    # else:
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dicts'].items()})
    # cuda eval
    model.cuda()
    model.eval()

    # valid_transform
    height, width = 256, 192
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])
    return model, valid_transform

def demo_par(model, valid_transform, img):
    # load one image
    img_trans = valid_transform(img)
    imgs = torch.unsqueeze(img_trans, dim=0)
    imgs = imgs.cuda()
    valid_logits = model(imgs)
    valid_probs = torch.sigmoid(valid_logits)
    score = valid_probs.data.cpu().numpy()

    # show the score in the image
    txt_res = []
    txt = ""
    for idx in range(len(values)):
        if score[0, idx] >= 0.5:
            temp = '%s: %.2f ' % (values[idx], score[0, idx])
            txt += temp
            txt_res.append(txt)
    return txt_res

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    # model_filename = 'model_data/mars-small128.pb'
    # encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    # metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)



    tracking = False
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = 'video.webm'
    if asyncVideo_flag:
        video_capture = VideoCaptureAsync(file_path)
    else:
        video_capture = cv2.VideoCapture(file_path)

    if asyncVideo_flag:
        video_capture.start()

    if writeVideo_flag:
        if asyncVideo_flag:
            w = int(video_capture.cap.get(3))
            h = int(video_capture.cap.get(4))
        else:
            w = int(video_capture.get(3))
            h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_custom_yolo.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    model_par, valid_transform = model_init_par()
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break

        t1 = time.time()

        image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
        boxes, confidence, classes = yolo.detect_image(image)

        if tracking:
            # features = encoder(frame, boxes)

            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % round(det.confidence * 100, 2) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)
            crop_img = image.crop([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
            # res_txt = demo_par(model_par, valid_transform, crop_img)
            res_txt = ''
            # draw.rectangle(xy=person_bbox[:-1], outline='red', width=1)

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
            cv2.putText(frame, "ID: " + str(res_txt), (int(bbox[0]), int(bbox[1])), 0,
                        1e-3 * frame.shape[0], (0, 255, 0), 1)
            font = ImageFont.truetype(
                '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/arial.ttf',
                size=10)

        if tracking:
            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                #crop_img = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
                crop_img = image.crop([int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])
                #res_txt = demo_par(model_par, valid_transform, crop_img)

                #draw.rectangle(xy=person_bbox[:-1], outline='red', width=1)

                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                            1e-3 * frame.shape[0], (0, 255, 0), 1)
                font = ImageFont.truetype(
                    '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/arial.ttf',
                    size=10)
                # positive_cnt = 1
                # for txt in res_txt:
                #     if 'personal' in txt:
                #         #draw.text((x1, y1 + 20 * positive_cnt), txt, (255, 0, 0), font=font)
                #         cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1]) + 20 * positive_cnt), 0,
                #                     1e-3 * frame.shape[0], (0, 255, 0), 1)
                #         positive_cnt += 1

        cv2.imshow('', frame)

        if writeVideo_flag:  # and not asyncVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1

        fps_imutils.update()

        if not asyncVideo_flag:
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % (fps))

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fps_imutils.stop()
    print('imutils FPS: {}'.format(fps_imutils.fps()))

    if asyncVideo_flag:
        video_capture.stop()
    else:
        video_capture.release()

    if writeVideo_flag:
        out.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
