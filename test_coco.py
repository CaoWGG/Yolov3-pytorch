from lib.utils import post_process
import os
import cv2
import time
from dataset import get_dataset
from model.darknet import DarkNet
import torch
import numpy as np


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')
model = DarkNet('cfg/yolov3.coco.cfg')
#model.load_weights('weights/pretrain/yolov3.weights')
model.load_state_dict(torch.load('weights/coco_backup/model_final.pth'))

model = model.cuda()
model = model.eval()
#data = get_dataset('dark')('/data/yoloCao/DataSet/coco2014', 'val', model.net_info)
#data = get_dataset('pascal')('/data/yoloCao/DataSet/VOC','val', model.net_info,augment=False)
data = get_dataset('coco')('/data/yoloCao/DataSet/coco', 'val', model.net_info,augment=False)
for inp, bb,img_id,_ in data:
    netshape = inp.shape[1:]
    img = np.expand_dims(inp, 0)
    img = torch.from_numpy(img).float().cuda()
    start = time.time()
    out = model(img)
    img_name = data.get_image_name(img_id)
    image = cv2.imread(img_name)
    #image = cv2.resize(image,(416,416))
    out = post_process(out, image.shape[:2],netshape,score_threshold=0.1, iou_threshold=0.45, method='nms')
    print()
    for det in out:
        x, y, x1, y1, conf, cls = det
        print('%s  : %0.4f' % (data.class_name[int(cls)+1],
                               conf))
        # cv2.circle(image, (int((x + x1) / 2), int((y + y1) / 2)), 3, (255, 0, 0), 3)
        cv2.rectangle(image, (int(x), int(y)), (int(x1), int(y1)), (255, 0, 0), 3)
    cv2.imshow('', image)
    if cv2.waitKey(0) & 0xff == 27:
        break
cv2.destroyAllWindows()