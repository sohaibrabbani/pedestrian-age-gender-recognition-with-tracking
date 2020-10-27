# coding: utf-8
# -----------------------------------------
# Writed by wduo. github.com/wduo
# -----------------------------------------
import os
import glob
from timeit import time

import cv2
import imutils.video
from tqdm import tqdm
from PIL import Image, ImageFont, ImageDraw
import pdb

import torch
import torchvision.transforms as T

from mmdet.apis import init_detector, inference_detector
import mmcv

from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50

# values = ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Front', 'Side', 'Back']
# values = ['AgeOver60', 'Age18-60', 'AgeLess18', 'Female']
from videocaptureasync import VideoCaptureAsync

pa100k_values = ['Hat', 'Glasses', 'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice', 'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'AgeOver60', 'Age18-60', 'AgeLess18', 'Female', 'Front', 'Side', 'Back']
# values = ['accessoryHat', 'accessoryMuffler', 'accessoryNothing', 'accessorySunglasses', 'hairLong', 'upperBodyCasual', 'upperBodyFormal', 'upperBodyJacket', 'upperBodyLogo', 'upperBodyPlaid', 'upperBodyShortSleeve', 'upperBodyThinStripes', 'upperBodyTshirt', 'upperBodyOther', 'upperBodyVNeck', 'lowerBodyCasual', 'lowerBodyFormal', 'lowerBodyJeans', 'lowerBodyShorts', 'lowerBodyShortSkirt', 'lowerBodyTrousers', 'footwearLeatherShoes', 'footwearSandals', 'footwearShoes', 'footwearSneaker', 'carryingBackpack', 'carryingOther', 'carryingMessengerBag', 'carryingNothing', 'carryingPlasticBags', 'personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'personalMale']
values = ['Age1-16', 'Age17-30', 'Age31-45', 'Age46-60', 'Female', 'Male']

# values = ['personalLess30', 'personalLess45', 'personalLess60', 'personalLarger60', 'personalMale']

def model_init_mmdet():
    # mmdet v2.x
    config_file = '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco.py'
    checkpoint_file = '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/mmdetection/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_1x_coco_20200203-cff10310.pth'
    # config_file = 'configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_1x_coco.py'
    # checkpoint_file = '/mmdet_ckpt/cascade_rcnn_x101_64x4d_fpn_1x_coco_20200515_075702-43ce6a30.pth'
    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    return model


def demo_mmdet(model, img_path):
    img = mmcv.imread(img_path)
    result = inference_detector(model, img)
    return result


def model_init_par():
    # model
    backbone = resnet50()
    classifier = BaseClassifier(nattr=6)
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
    return txt

if __name__ == '__main__':

    # model
    model_mmdet = model_init_mmdet()
    model_par, valid_transform = model_init_par()

    # imgs
    root_dir = '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/test_data/imgs/'
    img_paths = glob.glob(os.path.join(root_dir, '**', '*.[pj][np][g]'), recursive=True)
    writeVideo_flag = True
    asyncVideo_flag = False

    file_path = '/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/videos/abc.mp4'
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
        out = cv2.VideoWriter('output_custom.avi', fourcc, 30, (w, h))
        frame_index = -1

    fps = 0.0
    fps_imutils = imutils.video.FPS().start()
    # L
    # for img_path in tqdm(img_paths):
    count = 0
    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

        # mmdet
        img = Image.fromarray(frame[..., ::-1])  # bgr to rgb

        result = demo_mmdet(model_mmdet, frame)
        person_bboxes = result[0]
        person_bboxes_list = []
        for ii in range(person_bboxes.shape[0]):
            x1, y1, x2, y2, score = person_bboxes[ii][0], person_bboxes[ii][1], \
                                    person_bboxes[ii][2], person_bboxes[ii][3], person_bboxes[ii][4]
            if score > 0.6:
                person_bboxes_list.append([x1, y1, x2, y2, score])

        # par
        # img = Image.open(img_path).convert('RGB')
        draw = ImageDraw.Draw(img)
        for person_bbox in person_bboxes_list:
            #
            x1, y1, x2, y2, score = person_bbox
            crop_img = img.crop(person_bbox[:-1])
            # res_txt = demo_par(model_par, valid_transform, crop_img)
            #
            draw.rectangle(xy=person_bbox[:-1], outline='red')
            font = ImageFont.truetype('/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/arial.ttf', size=25)

            # if not any("Female" in s for s in res_txt):
            #     res_txt.append("Male")

            # draw.text((x1, y1 + 5), " ".join([item for item in res_txt]), (255, 0, 0), font=font)
            # draw.text((x1, y1 + 5), res_txt, (255, 0, 0), font=font)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            # cv2.putText(frame, res_txt, (int(x1), int(y1)), fontFace=0,
            #             fontScale=int(1e-3 * frame.shape[0]), color=(0, 255, 0), thickness=2)

            # print(img_path, res_txt)
            # positive_cnt = 0
            # for txt in res_txt:
            #     if 'top:' in txt or 'mask:' in txt:
            #         draw.text((x1, y1 + 20 * positive_cnt), txt, (255, 0, 0), font=font)
            #         positive_cnt += 1
        #
        # if person_bboxes_list is not None:
        #     img.save(os.path.join('/home/sohaibrabbani/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition/test_data/res/', str(count) + '.jpg'))
        #     count += 1

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

        # pdb.set_trace()

    pass