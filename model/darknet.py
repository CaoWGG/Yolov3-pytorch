import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function,once_differentiable
from lib.cfg import parse_cfg
from collections import OrderedDict
from lib.utils import cal_iou_torch
import math

SUPPORTED_LAYERS = ['shorcut','convolutional','route',
                    'yolo','my_yolo','upsample']


def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class ConvWS2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


class route(nn.Module):
    def __init__(self, axis=1):
        super(route, self).__init__()
        self.axis = axis

    def forward(self, *x):
        if len(x) ==1:
            return x[0]
        else:
            return torch.cat(x, self.axis)


class upsample(nn.Module):
    def __init__(self):
        super(upsample,self).__init__()

    def forward(self, x):
        return F.interpolate(x, scale_factor=2,mode='nearest')

class shorcut(nn.Module):
    def __init__(self, block ):
        super(shorcut, self).__init__()
        activation = block['activation']
        if activation=='leaky':
            self.act = nn.LeakyReLU(0.1)
        elif activation == 'linear' :
            self.act = None

    def forward(self, *inputs):
        x = inputs[0] + inputs[1]
        if self.act:
            x = self.act(x)
        return x

class max_pool(nn.Module):

    def __init__(self,block):
        super(max_pool,self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=int(block['size']),stride=int(block['stride']),padding=0)
    def forward(self, x):
        x = self.pool(x)
        return x

class convolutional(nn.Module):

    def __init__(self, block):
        super(convolutional, self).__init__()
        in_channels=block['in_channel']
        out_channels = block['filters']
        kernel_size = block['size']
        stride = block['stride']
        pad=block['pad']
        batch_normalize = block['batch_normalize']
        activation = block['activation']
        padding = 0
        if pad: padding= kernel_size//2

        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                              kernel_size=kernel_size,stride=stride,
                              padding=padding,bias=not batch_normalize)
        if batch_normalize:
            ##  updata slow
            self.bn = nn.BatchNorm2d(out_channels,momentum=0.01,eps=1e-5)
        else:
            self.bn = None

        if activation=='leaky':
            self.act = nn.LeakyReLU(0.1,inplace=True)

        elif activation == 'linear' :
            self.act = None

        self.init_wight()

    def forward(self, x):

        x=self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act:
            x = self.act(x)

        return x

    def init_wight(self):

        if self.bn:
            nn.init.constant_(self.bn.weight,1)
            nn.init.constant_(self.bn.bias,0)
        else:
            nn.init.constant_(self.conv.bias,0)


class yolo_op(Function):

    @staticmethod
    def forward(ctx, x , grid_x ,grid_y, anchors_w, anchors_h,anchors,classes,min_mask,max_mask,ignore_thresh,target):

        cls_acc, conf_obj, conf_noobj, mean_iou, iou50, iou75, total_loss = 0, 0, 0, 0, 0, 0, 0
        B,C,G,G = x.size()
        anchor_num = anchors.size(0)
        classes = classes.item()
        pred = x.view(B,anchor_num,classes+5,G,G).permute([0,1,3,4,2])

        ####  decode for compute iou and grad
        pred_dx = torch.sigmoid(pred[..., 0])
        pred_dy = torch.sigmoid(pred[..., 1])
        pred_dw = pred[..., 2]
        pred_dh = pred[..., 3]
        pred_x = (pred_dx+ grid_x)/G
        pred_y = (pred_dy+ grid_y)/G
        pred_w = torch.exp(pred_dw) * anchors_w/G
        pred_h = torch.exp(pred_dh) * anchors_h/G
        pred_conf = torch.sigmoid(pred[...,4])
        pred_prob = torch.sigmoid(pred[...,5:])
        pred_xywh = torch.stack([pred_x, pred_y, pred_w, pred_h], -1)

        ## compute iou
        iou_scores = torch.zeros_like(pred_conf)
        all_b = target[:, 0].long().t()
        iou = cal_iou_torch(pred_xywh[all_b, :, :, :, :4],
                            target[:, 2:6][:, np.newaxis, np.newaxis, np.newaxis, :])
        for i in range(B):
            b_mask = all_b == i
            if b_mask.sum() > 0:
                b_iou = torch.max(iou[b_mask], 0)[0]
                iou_scores[i] = b_iou

        ## compute grad
        grad_x = torch.zeros_like(pred_dx)
        grad_y = torch.zeros_like(pred_dy)
        grad_w = torch.zeros_like(pred_dw)
        grad_h = torch.zeros_like(pred_dh)
        grad_conf = torch.zeros_like(pred_conf)
        grad_class = torch.zeros_like(pred_prob)


        grad_conf[...]  = pred_conf[...] - 0
        ingore_mask = iou_scores > ignore_thresh
        grad_conf[ingore_mask] = 0

        best_anchor = target[:, -1]
        best_mask = (best_anchor >= min_mask) * (best_anchor <= max_mask)
        count = best_mask.sum()
        if count > 0:
            target = target[best_mask]
            best_n = (target[:, -1] - min_mask).long()
            gxy = target[:, 2:4]*G
            gwh = target[:, 4:6]
            b, target_labels = target[:, :2].long().t()  # target = [b_id, cls, x, y, w, h, best_acnhor]
            gx, gy = gxy.t()
            gw, gh = gwh.t()
            gi, gj = gxy.long().t()
            scale = 2-gwh[:,0]*gwh[:,1]
            grad_x[b, best_n, gj, gi] = scale*(pred_dx[b,best_n,gj,gi] - (gx - gx.floor()))
            grad_y[b, best_n, gj, gi] = scale*(pred_dy[b,best_n,gj,gi] - (gy - gy.floor()))
            grad_w[b, best_n, gj, gi] = scale*(pred_dw[b,best_n,gj,gi] - torch.log(gw*G / anchors[best_n][:, 0] + 1e-16))
            grad_h[b, best_n, gj, gi] = scale*(pred_dh[b,best_n,gj,gi] - torch.log(gh*G / anchors[best_n][:, 1] + 1e-16))
            grad_conf[b, best_n, gj, gi] = pred_conf[b, best_n, gj, gi] -1

            grad_class[b,best_n,gj,gi] = pred_prob[b,best_n,gj,gi] -0
            grad_class[b, best_n, gj, gi, target_labels] = pred_prob[b, best_n, gj, gi, target_labels] - 1


        ### mertic
            cls_acc = (pred_prob[b, best_n, gj, gi].argmax(-1) == target_labels).float().mean().item()
            conf_obj = pred_conf[b,best_n,gj,gi].mean().item()
            conf_noobj = pred_conf.mean().item()
            iou = iou_scores[b,best_n,gj,gi]
            mean_iou = iou.mean().item()
            iou50 = (iou>0.5).float().mean().item()
            iou75 = (iou>0.75).float().mean().item()

        print('grid: %d, Avg IOU: %0.4f, Class: %0.4f, Obj: %0.4f, No Obj: %0.4f, .5R: %0.4f, .75R: %0.4f, count: %d' % (
                G,mean_iou, cls_acc, conf_obj, conf_noobj,iou50, iou75,count.item()))

        ##  constrcut grad for backward
        grad_xywhc = torch.stack([grad_x,grad_y,grad_w,grad_h,grad_conf],-1)
        grad = torch.cat([grad_xywhc,grad_class],-1).permute([0,1,4,2,3]).contiguous().view([B,C,G,G])
        ctx.save_for_backward(grad)

        ## compute loss for show
        loss = torch.pow(grad, 2).sum()

        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        grad, = ctx.saved_variables

        return grad,None,None,None,None,None,None,None,None,None,None

def yolo_grad(x, grid_x ,grid_y, anchors_w, anchors_h,anchors,classes,min_mask,max_mask,ignore_thresh,target):

    return yolo_op.apply(x,grid_x ,grid_y, anchors_w,anchors_h,anchors,classes,min_mask,max_mask,ignore_thresh,target)

class yolo_layer(nn.Module):

    def __init__(self, block,net):
        super(yolo_layer, self).__init__()
        self.stride = block['stride']
        mask = list(eval(block['mask']))
        anchors = np.array(eval(block['anchors'])).reshape(int(block['num']),2)[mask]/self.stride
        self.classes = int(block['classes'])
        self.anchor_num = len(anchors)
        self.w ,self.h= block['width'],block['height']
        self.x = torch.arange(self.w)[np.newaxis, :].expand([self.w, self.h])[np.newaxis,np.newaxis,:,:].float()
        self.y = torch.arange(self.h)[:, np.newaxis].expand([self.w, self.h])[np.newaxis,np.newaxis,:,:].float()
        self.anchors_w = torch.from_numpy(anchors[:,0])[np.newaxis,:,np.newaxis,np.newaxis].float()
        self.anchors_h = torch.from_numpy(anchors[:,1])[np.newaxis,:,np.newaxis,np.newaxis].float()
        self.anchors = torch.from_numpy(anchors).float()
        self.min_mask = torch.tensor(min(mask)).float()
        self.max_mask = torch.tensor(max(mask)).float()
        self.ingore_thresh = torch.tensor(float(block['ignore_thresh'])).float()
        self.classes_tensor = torch.tensor(self.classes).long()

    def compute_grid(self,h,w):
        self.w,self.h=w,h
        self.x = torch.arange(self.w)[np.newaxis, :].expand([self.w, self.h])[np.newaxis, np.newaxis, :, :].float()
        self.y = torch.arange(self.h)[:, np.newaxis].expand([self.w, self.h])[np.newaxis, np.newaxis, :, :].float()
        self.x = self.fn(self.x)
        self.y = self.fn(self.y)

    def _apply(self, fn):
        self.x = fn(self.x)
        self.y = fn(self.y)
        self.anchors_w = fn(self.anchors_w)
        self.anchors_h = fn(self.anchors_h)
        self.anchors = fn(self.anchors)
        self.classes_tensor = fn(self.classes_tensor)
        self.min_mask = fn(self.min_mask)
        self.max_mask = fn(self.max_mask)
        self.ingore_thresh = fn(self.ingore_thresh)
        self.fn=fn

    def forward(self, x,label=None):
        B,C,H,W = x.size()
        if H != self.h :
            self.compute_grid(H,W)
        if label is not None:
            loss = yolo_grad(x,self.x,self.y,self.anchors_w,self.anchors_h,self.anchors,self.classes_tensor,
                              self.min_mask,self.max_mask,self.ingore_thresh,label)
            return loss
        else:
            pred = x.view(B,self.anchor_num,self.classes+5,H,W).permute([0,1,3,4,2])
            pred_dx = torch.sigmoid(pred[..., 0])
            pred_dy = torch.sigmoid(pred[..., 1])
            pred_dw = pred[..., 2]
            pred_dh = pred[..., 3]
            pred_x = (pred_dx+ self.x)/W
            pred_y = (pred_dy+ self.y)/H
            pred_w = torch.exp(pred_dw) * self.anchors_w / W
            pred_h = torch.exp(pred_dh) * self.anchors_h / H
            pred_conf = torch.sigmoid(pred[...,4])
            pred_prob = torch.sigmoid(pred[...,5:])
            pred_xywh = torch.stack([pred_x, pred_y, pred_w, pred_h, pred_conf], -1)
            pred_bbox = torch.cat([pred_xywh, pred_prob], -1)
            pred_bbox = pred_bbox.view([B,-1,self.classes+5])
            return pred_bbox


class DarkNet(nn.Module):
    def __init__(self, cfgfile):
        super(DarkNet,self).__init__()
        self.net_info,self.need_save = parse_cfg(cfgfile)
        self.models = self.create_network(self.net_info)
        for name, model in self.models.items():
            self.add_module(name, model)

    def forward(self,x,label=None):

        temp_save = OrderedDict()
        ret = []
        for ind, block in enumerate(self.net_info):
            ind = str(ind)
            type = block['type']
            if type == 'net' :
                continue

            elif type == 'convolutional':
                x = self.models[ind](x)

            elif type == 'route':
                index = block['from']
                x = [temp_save[str(fid)] for fid in index]
                x = self.models[ind](*x)
                for fid in index:
                    del temp_save[str(fid)]

            elif type == 'shortcut':
                index = str(block['from'])
                x = [x,temp_save[index]]
                x = self.models[ind](*x)
                del temp_save[index]

            elif type == 'upsample':
                x = self.models[ind](x)

            elif type == 'maxpool':
                x = self.models[ind](x)

            elif type == 'yolo':
                res = self.models[ind](x,label)
                ret.append(res)

            if int(ind) in self.need_save:
                temp_save[ind] = x

        return ret

    def create_network(self,net_info):
        models = OrderedDict()
        for ind,block in enumerate(net_info):
            ind = str(ind)
            type = block['type']
            if type == 'net' :
                continue
            elif type == 'convolutional':
                models[ind] = convolutional(block)
            elif type == 'route':
                models[ind] = route()
            elif type == 'shortcut':
                models[ind] = shorcut(block)
            elif type == 'upsample':
                models[ind] = upsample()
            elif type == 'yolo':
                models[ind] = yolo_layer(block,net_info[0])
            elif type == 'maxpool':
                models[ind] = max_pool(block)
            else:
                raise Exception("Not Support")

        return models

    def load_weights(self,weights):
        fp = open(weights, 'rb')
        header = np.fromfile(fp, count=5, dtype=np.int32)
        buf = np.fromfile(fp, dtype=np.float32)
        buf_len = len(buf)
        point = 0
        for ind, block in enumerate(self.net_info):
            if point == buf_len:
                print(ind-1)
                break
            ind = str(ind)
            type = block['type']
            if type == 'convolutional':
                in_channels = block['in_channel']
                out_channels = block['filters']
                kernel_size = block['size']
                batch_normalize = block['batch_normalize']
                numb = out_channels
                numw = in_channels*out_channels*kernel_size*kernel_size
                bias = buf[point:point + numb].copy();point += numb
                if batch_normalize:
                    scale = buf[point:point + numb].copy();point += numb
                    mean = buf[point:point + numb].copy();point += numb
                    var = buf[point:point + numb].copy();point += numb
                weight = buf[point:point + numw].copy();point += numw
                weight = torch.from_numpy(weight).view_as(self.models[ind].conv.weight)
                self.models[ind].conv.weight.data.copy_(weight)
                if batch_normalize:
                    scale = torch.from_numpy(scale).view_as(self.models[ind].bn.weight)
                    self.models[ind].bn.weight.data.copy_(scale)
                    bias = torch.from_numpy(bias).view_as(self.models[ind].bn.bias)
                    self.models[ind].bn.bias.data.copy_(bias)
                    mean  = torch.from_numpy(mean).view_as(self.models[ind].bn.running_mean)
                    self.models[ind].bn.running_mean.data.copy_(mean)
                    var = torch.from_numpy(var).view_as(self.models[ind].bn.running_var)
                    self.models[ind].bn.running_var.data.copy_(var)
                else:
                    bias = torch.from_numpy(bias).view_as(self.models[ind].conv.bias)
                    self.models[ind].conv.bias.data.copy_(bias)