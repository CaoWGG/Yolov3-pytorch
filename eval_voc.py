import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
from dataset import get_dataset
from model.darknet import DarkNet
import os.path as osp
import shutil
from tqdm import tqdm
import torch
from lib.utils import post_process
import numpy as np
import cv2

class Eval():

    def __init__(self,cfg='cfg/yolov3.cfg',weights='weights/model49.pth',data_name='dark',
                 data_root = '/data/yoloCao/DataSet/coco2014'):
        self.model = DarkNet(cfg)
        if weights.endswith('.weights'):
            self.model.load_weights(weights)
        elif weights.endswith('.pth') or weights.endswith('.backup'):
            self.model.load_state_dict(torch.load(weights, map_location='cpu'))
        else:
            raise Exception("error")
        self.loader = get_dataset(data_name)(data_root, 'val', self.model.net_info,augment=False)
        #self.loadImgs = self.loader.dataset.coco.loadImgs
        self.class_name = self.loader.class_name
        self.net_shape=[416,416]
        self.model.eval()
        self.model.cuda()



    def run_det(self):
        predict_path = './mAP/predicted'
        gt_path = './mAP/ground-truth'
        if os.path.exists(predict_path):
            shutil.rmtree(predict_path)
        if os.path.exists(gt_path):
            shutil.rmtree(gt_path)
        os.mkdir(predict_path)
        os.mkdir(gt_path)

        for i,(inp, bb,img_id,gt_boxes) in enumerate(tqdm(self.loader)):
            netshape = inp.shape[1:]
            image = np.expand_dims(inp, 0)
            img = torch.from_numpy(image).float().cuda()
            out = self.model(img)
            img_name = self.loader.get_image_name(img_id)
            image = cv2.imread(img_name)
            bboxes = post_process(out, image.shape[:2], netshape, score_threshold=0.001, iou_threshold=0.45, method='nms')
            gt_path_res = osp.join(gt_path,str(i)+'.txt')
            pred_path_res = osp.join(predict_path,str(i)+'.txt')
            with open(pred_path_res,'w') as f:
                for x, y, x1, y1, p ,cls_id in bboxes:
                    f.write('%s %.4f %.4f %.4f %.4f %.4f'%(self.loader.class_name[int(cls_id) + 1],p,x,y,x1,y1) + '\n')
            with open(gt_path_res, 'w') as f:
                for x, y, x1, y1,cls_id in gt_boxes:
                    f.write('%s %.4f %.4f %.4f %.4f'%(self.loader.class_name[int(cls_id) + 1], x, y, x1, y1) + '\n')

if __name__ == '__main__':
    Eval(cfg='cfg/yolov3-voc.cfg',
         weights='weights/backup/model.backup',
         data_name='pascal',
         data_root='/data/yoloCao/DataSet/VOC').run_det()