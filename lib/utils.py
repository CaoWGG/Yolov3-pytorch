import torch
import numpy as np
from torch.utils.data import DataLoader,Sampler,RandomSampler,SequentialSampler

def cal_iou_torch(boxes1, boxes2 ,xywh = True):
    if xywh:
        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                          boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                          boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    inter_max_xy = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_min_xy = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter = torch.max((inter_max_xy - inter_min_xy), torch.zeros_like(inter_max_xy))
    inter_area = inter[..., 0] * inter[..., 1]
    union = area1 + area2 - inter_area
    ious = 1.0 * inter_area / union
    return ious

def cal_iou_np(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def cal_wh_iou_torch(wh1,wh2):
    wh1_area= wh1[...,0] * wh1[...,1]
    wh2_area = wh2[..., 0] * wh2[..., 1]
    inter_section =  torch.min(wh1,wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = (wh1_area + wh2_area +1e-16)- inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def cal_wh_iou_np(wh1,wh2):
    wh1_area= wh1[...,0] * wh1[...,1]
    wh2_area = wh2[..., 0] * wh2[..., 1]
    inter_section = np.minimum(wh1,wh2)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = (wh1_area + wh2_area +1e-16)- inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def nms(bboxes, score_threshold, iou_threshold, sigma=0.3, method='soft-nms'):
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = cal_iou_np(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            assert method in ['nms', 'soft-nms']
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return best_bboxes


def post_process(det,image_shape=None,net_shape=(416,416),score_threshold=0.5, iou_threshold=0.5, sigma=0.3,method='soft-nms'):
    det= torch.cat(det,1)
    class_conf,class_index=torch.max(det[:, :, 5:], dim=-1, keepdim=True)
    pred_xywh = det[:,:,:4]
    class_conf = det[:, :, 4:5]*class_conf
    mask = class_conf > score_threshold
    class_conf = class_conf[mask].cpu().detach().numpy()
    class_index = class_index[mask].cpu().detach().numpy()
    pred_xywh = pred_xywh[mask[:,:,0]].cpu().detach().numpy()
    pred_xyxy = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    orgh, orgw = image_shape
    w,h = net_shape
    ar = orgw / orgh
    if ar < 1:
        new_h = h
        new_w = ar * new_h
    else:
        new_w = w
        new_h = new_w/ar
    pred_xyxy[:, [0, 2]] = (pred_xyxy[:, [0, 2]] * w - (w - new_w)/2)*(orgw/new_w)
    pred_xyxy[:, [1, 3]] = (pred_xyxy[:, [1, 3]] * h - (h - new_h)/2)*(orgh/new_h)
    pred_xyxy = np.concatenate([np.maximum(pred_xyxy[:, :2], [0, 0]),
                                np.minimum(pred_xyxy[:, 2:], [orgw - 1, orgh - 1])], axis=-1)
    pred_xyxy = np.concatenate([pred_xyxy,class_conf[:,np.newaxis],class_index[:,np.newaxis]],axis=-1)
    best_bboxes = nms(pred_xyxy,score_threshold,iou_threshold,sigma,method)
    return best_bboxes


def set_lr(opt,lr):
    for param_group in opt.param_groups:
        param_group['lr'] = lr
    return lr


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last,multiscale_step=None,img_sizes = None):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        if multiscale_step is not None and multiscale_step < 1 :
            raise ValueError("multiscale_step should be > 0, but got "
                             "multiscale_step={}".format(multiscale_step))
        if multiscale_step is not None and img_sizes is None:
            raise ValueError("img_sizes must a list, but got img_sizes={} ".format(img_sizes))
        self.multiscale_step = multiscale_step
        self.img_sizes = img_sizes

    def __iter__(self):
        num_batch = 0
        batch = []
        size = 416
        for idx in self.sampler:
            batch.append([idx,size])
            if len(batch) == self.batch_size:
                yield batch
                num_batch+=1
                batch = []
                if self.multiscale_step and num_batch % self.multiscale_step == 0 :
                    size = np.random.choice(self.img_sizes)
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

def collate_fn(batch):
    imgs, targets,index,_ = list(zip(*batch))
    labels = []
    for i, boxes in enumerate(targets):
        if len(boxes)>0:
            boxes = np.array(boxes)
            boxes[:, 0] = i
            labels.append(boxes)
    targets = np.concatenate(labels, 0)
    targets = torch.from_numpy(targets).float()
    imgs = torch.stack([torch.from_numpy(img) for img in imgs])
    return imgs, targets
