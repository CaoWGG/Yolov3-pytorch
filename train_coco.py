from model.darknet import DarkNet
from torch.utils.data import DataLoader
from lib.utils import BatchSampler,SequentialSampler,RandomSampler,collate_fn
import os
from torch.optim import SGD, Adam
import torch
from dataset import get_dataset
from lib.utils import set_lr
from joblib import cpu_count
import numpy as np
import logging
from tensorboardX import SummaryWriter
import random
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.benchmark= False  ## input size is not fixed


os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

logging.basicConfig(filename='log/train_coco.log',
                    format='%(filename)s %(asctime)s\t%(message)s',
                    level=logging.DEBUG, datefmt='%Y-%m-%d %I:%M:%S', filemode='w')

cfg = 'cfg/yolov3.coco.cfg'
weights = 'weights/pretrain/darknet53.conv.74'

lr = 1e-3
burn_in = 1000
max_epoch = 140
batch_size = 64
subvid = 8
one_batch = batch_size // subvid
dataset_name = 'coco'
dataset_root = '/data/yoloCao/DataSet/coco'
lr_step = [90,120]
start_epoch = 0
num_iter = 0

writer = SummaryWriter(logdir='log')
avg_loss,avg_loss_x,avg_loss_y,avg_loss_w,avg_loss_h,avg_loss_c,avg_loss_cls = -1,-1,-1,-1,-1,-1,-1


model = DarkNet(cfg)
model.load_weights(weights=weights)
#model.load_state_dict(torch.load(weights))

dataset_train = get_dataset(dataset_name)(dataset_root,'train',model.net_info,augment=True)
loader_train = DataLoader(dataset=dataset_train,
                          batch_sampler=BatchSampler(RandomSampler(dataset_train),
                                         batch_size=batch_size,
                                         drop_last=True,
                                         multiscale_step=10,
                                         img_sizes=list(range(320, 608 + 1, 32))),
                          num_workers=cpu_count(),
                          collate_fn=collate_fn)
log_freq = len(loader_train)//10

dataset_val = get_dataset(dataset_name)(dataset_root,'val',model.net_info,augment=False)
loader_val =DataLoader(dataset=dataset_val,
                      batch_sampler=BatchSampler(SequentialSampler(dataset_val),
                                     batch_size=batch_size//subvid,
                                     drop_last=False),
                      num_workers=cpu_count(),
                      collate_fn=collate_fn)
model = model.cuda()
opt = SGD([{'params':filter(lambda x:len(x.size()) == 4 ,model.parameters()),'weight_decay':0.0005*batch_size },
           {'params': filter(lambda x:len(x.size())<4,model.parameters())}],
          lr=lr, momentum= 0.9 ,nesterov=True)

for epoch in range(start_epoch,max_epoch,1):
    model = model.train()
    if epoch in lr_step:
        now_lr = set_lr(opt, lr * (0.1**(lr_step.index(epoch)+1))/batch_size)*batch_size
    for b_id, batch in enumerate(loader_train):
        num_iter= num_iter + 1
        if num_iter <= burn_in:
            now_lr = set_lr(opt, lr * pow(num_iter / burn_in, 4) / batch_size)*batch_size

        img, label = batch
        l_b = label[:,0]
        train_loss,loss_x,loss_y,loss_w,loss_h,loss_c,loss_cls = 0,0,0,0,0,0,0
        opt.zero_grad()
        for i in range(0,batch_size,one_batch):
            bmask = (i<=l_b)*(l_b<(i+one_batch))
            input = img[i:i+one_batch].cuda().float()
            input_label = label[bmask].cuda().float()
            input_label[:,0]-=i
            loss = model(input, input_label)
            loss_sum = (sum(loss))/3
            loss_sum.backward()
            train_loss += (loss_sum.item())
        opt.step()

        train_loss/=batch_size
        if avg_loss < 0 :
            avg_loss = train_loss
        avg_loss = avg_loss*0.9 + train_loss*0.1


        writer.add_scalar('train_loss',train_loss,num_iter)
        writer.add_scalar('train_avg_loss', avg_loss, num_iter)
        writer.add_scalar('lr',now_lr,num_iter)
        if num_iter % log_freq == 0:
            logging.info('%d/%d loss: %0.4f(%0.4f), rate : %0.8f\n' % (epoch,num_iter,train_loss,avg_loss,now_lr))
        print('%d/%d loss: %0.4f(%0.4f), rate : %0.8f\n' % (epoch,num_iter,train_loss,avg_loss,now_lr))
        if num_iter % 100 ==0 :
            torch.save(model.state_dict(), 'weights/coco_backup/model.backup')
        if num_iter % 10000==0 or (num_iter <1000 and num_iter%100==0) :
            torch.save(model.state_dict(), 'weights/coco_backup/model%d.pth'%num_iter)

    model = model.eval()
    test_loss,loss_x,loss_y,loss_w,loss_h,loss_c,loss_cls = 0,0,0,0,0,0,0
    for bid,batch in enumerate(loader_val):
        with torch.no_grad():
            img, label = batch
            img, label = img.cuda().float(), label.cuda().float()
            loss = model(img, label)
            test_loss += (sum(loss)).item()/(3*batch_size//subvid)
    test_loss/=len(loader_val)
    writer.add_scalar('test_loss', test_loss)
    logging.info('%d loss: %0.4f\n' % (epoch,test_loss))
torch.save(model.state_dict(), 'weights/coco_backup/model_final.pth')
