from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import cv2
import os.path as osp
import numpy as np
class Darkdata(data.Dataset):
    class_name = ['__background__'] + list(map(str.strip, open(osp.join(osp.dirname(__file__),'../../cfg/yolo.names')).readlines()))

    def __init__(self, data_dir, split, net_info, augment=True):
        super(Darkdata, self).__init__()
        self.split = split
        file_txt = osp.join(data_dir,'%s.txt'%split)
        self.images = open(file_txt).readlines()
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

    def get_image_bboxes(self,index):
        file_name = self.images[index].strip()
        label_name = file_name.replace('images','labels').replace('.jpg','.txt')
        with open(label_name) as f:
            lines = f.readlines()
        img = cv2.imread(file_name)
        h,w,c = img.shape
        bboexs=[]
        for i,ann in enumerate(lines):
            cls_id,bbox = self._dark_to_box(ann,h,w)
            bboexs.append([*bbox, cls_id])
        return img, bboexs

    def _dark_to_box(self,ann,im_h,im_w):
        cls,x,y,w,h= list(map(float,ann.strip().split(' ')))
        cls = int(cls)
        x1,y1,x2,y2 = (x - w/2)*im_w,(y - h/2)*im_h,(x + w/2)*im_w,(y + h/2)*im_h
        return cls,(x1,y1,x2,y2)

    def get_image_name(self,img_id):
        return self.images[img_id].strip()