import paddle
import math
import os
import sys
import cv2
sys.path.append('../')
from vision.ssd.config.fd_config import define_img_size
input_img_size = 320
define_img_size(input_img_size)
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd
from vision.ssd.mb_tiny_fd import create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd_predictor
label_path = '../models/voc-model-labels.txt'
net_type = 'RFB'
class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = 'cuda:0'
candidate_size = 800
threshold = 0.1
val_image_root = (
    '/pic/linzai/1080Ti/home_linzai/PycharmProjects/insightface/RetinaFace/data/retinaface/val'
    )
val_result_txt_save_root = './widerface_evaluation/'
if net_type == 'slim':
    model_path = '../models/pretrained/version-slim-320.pdiparams'
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=
        candidate_size, device=test_device)
elif net_type == 'RFB':
    model_path = '../models/pretrained/version-RFB-320.pdiparams'
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=
        test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=
        candidate_size, device=test_device)
else:
    print('The net type is wrong!')
    sys.exit(1)
net.load(model_path)
counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue
        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = predictor.predict(im, candidate_size / 2,
            threshold)
        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root,
            event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        fout = open(os.path.join(val_result_txt_save_root, event_name, 
            file_name.split('.')[0] + '.txt'), 'w')
        fout.write(file_name.split('.')[0] + '\n')
        fout.write(str(boxes.size(0)) + '\n')
        for i in range(boxes.size(0)):
            bbox = boxes[i, :]
            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.
                floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(
                bbox[3] - bbox[1]), probs[i] if probs[i] <= 1 else 1) + '\n')
        fout.close()
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))
