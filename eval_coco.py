from lib.utils import post_process
import os
import cv2
import time
from dataset import get_dataset
from model.darknet import DarkNet
import torch
import numpy as np
from pycocotools.cocoeval import COCOeval
import json
from tqdm import tqdm
#
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
model = DarkNet('cfg/yolov3.coco.cfg')
model.load_state_dict(torch.load('weights/coco_backup/model_final.pth'))

model = model.cuda()
model = model.eval()
data = get_dataset('coco')('/data/yoloCao/DataSet/coco', 'val', model.net_info,augment=False)
detections = []
for inp, bb,img_id,_ in tqdm(data):
    netshape = inp.shape[1:]
    img = np.expand_dims(inp, 0)
    img = torch.from_numpy(img).float().cuda()
    start = time.time()
    out = model(img)
    img_name = data.get_image_name(img_id)
    image = cv2.imread(img_name)
    out = post_process(out, image.shape[:2],netshape,score_threshold=0.001, iou_threshold=0.45, method='nms')
    for det in out:
        x, y, x1, y1, conf, cls = det
        detection = {
            "image_id": int(data.images[img_id]),
            "category_id": int(data._valid_ids[int(cls)]),
            "bbox": [x,y,x1-x,y1-y],
            "score": float("{:.2f}".format(conf))
        }
        detections.append(detection)
json.dump(detections, open('mAP/results.json', 'w'))
coco_dets = data.coco.loadRes('mAP/results.json')
coco_eval = COCOeval(data.coco, coco_dets, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
##
# Average Precision(AP) @ [IoU = 0.50:0.95 | area = all | maxDets = 100] = 0.343
# Average Precision(AP) @ [IoU = 0.50 | area = all | maxDets = 100] = 0.572
# Average Precision(AP) @ [IoU = 0.75 | area = all | maxDets = 100] = 0.365
# Average Precision(AP) @ [IoU = 0.50:0.95 | area = small | maxDets = 100] = 0.181
# Average Precision(AP) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100] = 0.377
# Average Precision(AP) @ [IoU = 0.50:0.95 | area = large | maxDets = 100] = 0.451
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 1] = 0.282
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 10] = 0.435
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = all | maxDets = 100] = 0.462
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = small | maxDets = 100] = 0.296
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = medium | maxDets = 100] = 0.490
# Average Recall(AR) @ [IoU = 0.50:0.95 | area = large | maxDets = 100] = 0.573