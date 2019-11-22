from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import os
import torch.utils.data as data
import cv2
import numpy as np


class PascalVOC(data.Dataset):

    class_name = ['__background__', "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
                       "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                       "train", "tvmonitor"]
    _valid_ids = np.arange(1, 21, dtype=np.int32)
    cat_ids = {v: i for i, v in enumerate(_valid_ids)}

    def __init__(self, data_dir, split,net_info,augment=True):
        super(PascalVOC, self).__init__()
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'images')
        _ann_name = {'train': 'trainval0712', 'val': 'test2007'}
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'pascal_{}.json').format(_ann_name[split])
        self.split = split
        self.augment = augment
        self.data_rng = np.random.RandomState(123)
        self.eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self.eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        print('==> initializing pascal {} data.'.format(_ann_name[split]))
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
        self.input_size = 544
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
        bboexs = np.array(bboexs,dtype=np.float32)
        return img,bboexs

    def get_image_name(self,img_id):
        return os.path.join(self.img_dir,self.coco.loadImgs(ids=[self.images[img_id]])[0]['file_name']).strip()