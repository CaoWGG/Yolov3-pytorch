from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.utils.data as data
from lib.utils import cal_wh_iou_np
from lib.data_aug import *
import cv2


class Yolo(data.Dataset):
    def __getitem__(self, index_imgsize):
        if type(index_imgsize) == list or type(index_imgsize) == tuple:
            index, imgsize = index_imgsize
        else:
            index,imgsize = index_imgsize,self.input_size
        img, bboxes = self.get_image_bboxes(index)
        height, width = img.shape[0], img.shape[1]

        if  self.augment:
            augment_hsv(img,hgain=0.0103,sgain=0.691,vgain=0.433)

        ### centernet aug method
        s = max(img.shape[0], img.shape[1]) * 1.0
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.augment:
            sf = 0.4 # sclae
            cf = 0.1 # shift
            c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        flipped = False
        if np.random.random() < 0.5  and self.augment:
            flipped = True
            img = img[:, ::-1, :]
            c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [imgsize, imgsize])
        inp = cv2.warpAffine(img, trans_input,
                             (imgsize, imgsize),
                             flags=cv2.INTER_LINEAR,borderValue = (128,128,128))

        ## darknet aug method
        # dw ,dh = self.jitter * width , self.jitter * height
        # new_ar = (width + np.random.uniform(-dw,dw)) / (height + np.random.uniform(-dh,dh))
        # sclae = 1
        # if new_ar < 1:
        #     new_h = sclae * imgsize
        #     new_w = new_ar * new_h
        # else:
        #     new_w = sclae * imgsize
        #     new_h = new_w/new_ar
        # dx , dy =  (np.random.uniform(0,imgsize-new_w), np.random.uniform(0,imgsize-new_h)) if self.augment else \
        #     ((imgsize-new_w)/2,(imgsize-new_h)/2)
        #
        #
        # src = np.array( [[0,0],[0,height],[width,0]] ,dtype= np.float32)
        # dst = np.array( [[dx,dy],[dx,new_h+dy],[new_w+dx,dy]] , dtype=np.float32 )
        # M = cv2.getAffineTransform(src,dst)
        # inp = cv2.warpAffine(img, M, (imgsize, imgsize),borderValue = (128,128,128))


        gt = np.copy(bboxes)
        if flipped:
            bboxes[:, [0, 2]] = width - bboxes[:, [2, 0]]

        # bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * new_w / width + dx
        # bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * new_h / height + dy



        np.random.shuffle(bboxes)
        num_objs = len(bboxes)
        labels =[]
        for k in range(num_objs):
            bbox,class_id = bboxes[k][0:4],bboxes[k][-1]
            bbox[:2] = affine_transform(bbox[:2], trans_input)
            bbox[2:] = affine_transform(bbox[2:], trans_input)
            bbox = np.clip(bbox, 0, imgsize - 1)
            bboxes[k][0:4] = bbox[...]
            bbox_xywh = np.concatenate(
                [(bbox[2:] + bbox[:2]) * 0.5 , bbox[2:] - bbox[:2]], axis=-1)
            bbox_xywh = bbox_xywh /imgsize
            w,h = bbox_xywh[2:]
            if (w > 0.001) and ( h > 0.001):
                anchor_iou = cal_wh_iou_np(bbox_xywh[2:][np.newaxis, :], self.anchors/imgsize)
                best_achor = np.argmax(anchor_iou)
                labels.append([0, class_id, *bbox_xywh, best_achor])

        # #  debug
        # for x,y,x1,y1,c in bboxes.astype(np.int):
        #     cv2.rectangle(inp,(int(x),int(y)),(int(x1),int(y1)),(255,0,0),3)
        # cv2.imshow('',inp)
        # cv2.waitKey(0)

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = (inp.astype(np.float32) / 255.)
        inp = np.transpose(inp, [2, 0, 1])
        return inp, labels, index ,gt
