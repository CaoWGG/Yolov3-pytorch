from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import os
import torch.utils.data as data
import cv2
import json


class COCO(data.Dataset):
    class_name = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    _valid_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
        14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
        37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
        58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
        82, 84, 85, 86, 87, 88, 89, 90]
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}

    def __init__(self, data_dir,split,net_info,augment=True):
        super(COCO, self).__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'instances_{}2017.json').format(split)
        self.split = split
        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = []
        for img_id in self.coco.getImgIds():
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            if len(ann_ids) > 0:
                self.images.append(img_id)
        print(len(self.images),len(self.coco.getImgIds()))
        self.num_samples = len(self.images)
        self.input_h = net_info[0]['width']
        self.input_w = net_info[0]['height']
        self.input_size = max(self.input_h,self.input_w)
        self.input_size = 608
        self.augment=augment
        self.max_bbox = 90
        self.strides = []
        for ind, block in enumerate(net_info):
            if block['type'] != 'yolo':
                continue
            self.anchors = np.array(eval(block['anchors'])).reshape(int(block['num']), 2)
            self.num_classes = int(block['classes'])
            self.jitter = float(block['jitter'])
            self.strides.append(block['stride'])
        self.strides = np.array(self.strides)
        if not self.augment:
            self.jitter = 0
        print('Loaded {} {} samples'.format(split, self.num_samples))


    def __len__(self):
        return self.num_samples


    def _coco_box_to_bbox(self, box):
        bbox = [box[0], box[1], box[0] + box[2], box[1] + box[3]]
        return bbox

    def get_image_bboxes(self,index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        img = cv2.imread(img_path)
        bboexs=[]
        for i,ann in enumerate(anns):
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            bboexs.append([*bbox,cls_id])
        bboexs = np.array(bboexs)
        return img,bboexs

    def get_image_name(self,img_id):
        return os.path.join(self.img_dir,self.coco.loadImgs(ids=[self.images[img_id]])[0]['file_name']).strip()


    def _to_float(self, x):
        return float("{:.2f}".format(x))