import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.enet import ENet
from models.bisenetv2 import BiSeNetV2
#from models.icnet import ICNet
from config import cfg
from loading_data import loading_data
from utils.utils import *
from timer import Timer
from utils.stream_metrics import StreamSegMetrics
import pdb
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.args import get_parser


exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()
train_metric = StreamSegMetrics(2, "train")
val_metric = StreamSegMetrics(2, "val")

def main():
    # TODO Create a skeleton OOP

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')

    if torch.cuda.is_available():
        device = "cuda"
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    torch.backends.cudnn.benchmark = True

    net = []   
    
    if cfg.MODEL == "enet":
        if cfg.TRAIN.STAGE=='all':
            net = ENet(only_encode=False)
            if cfg.TRAIN.PRETRAINED_ENCODER != '':
                encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
                del encoder_weight['classifier.bias']
                del encoder_weight['classifier.weight']
                # pdb.set_trace()
                net.encoder.load_state_dict(encoder_weight)
        elif cfg.TRAIN.STAGE =='encoder':
            net = ENet(only_encode=True)
    else:   
        net = BiSeNetV2(cfg.DATA.NUM_CLASSES, pretrained=True)

    net=net.to(device)

    net.train()
    criterion = torch.nn.BCEWithLogitsLoss().to(device)# Binary Classification
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 
    validate(val_loader, net, criterion, optimizer, -1, restore_transform, device)
    print("Starting training..")
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        print(f"Epoch {epoch}/{cfg.TRAIN.MAX_EPOCH}")
        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, epoch, device)
        _t['train time'].toc(average=False)
        print('training time of one epoch: {:.2f}s'.format(_t['train time'].diff))
        _t['val time'].tic()
        validate(val_loader, net, criterion, optimizer, epoch, restore_transform, device)
        _t['val time'].toc(average=False)
        print('val time of one epoch: {:.2f}s'.format(_t['val time'].diff))


def train(train_loader, net, criterion, optimizer, epoch, device="cpu"):
    train_metric.reset()
    for inputs, labels in tqdm(train_loader, ascii=True):
        #inputs, labels = data
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)
        optimizer.zero_grad()

        if cfg.MODEL == "enet":
            outputs = net(inputs)
        else:
            outputs = net(inputs, test=False)[0]

        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        #print(scores(labels, outputs, 1))
        out_metr = outputs.detach()
        out_metr[out_metr > 0.5] = 1
        out_metr[out_metr<=0.5] = 0
        train_metric.update(labels.cpu().numpy(), out_metr.cpu().numpy())
    print(train_metric.get_results())


def update_metric(metric, outputs, labels):
        """
        Update the evaluation metric with the model outputs and labels.

        Args:
            `metric`: Metric object to be updated.\n
            `outputs`: Model outputs.\n
            `labels`: True labels.

        """
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction[prediction>0.5] = 1
        prediction[prediction<=0.5] = 0
        prediction = prediction.cpu().numpy()

def validate(val_loader, net, criterion, optimizer, epoch, restore, device):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    val_metric.reset()
    with torch.no_grad():
        for vi, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs).to(device)
            labels = Variable(labels).to(device)
            if cfg.MODEL == "enet":
                outputs = net(inputs)
            else:
                outputs = net(inputs, test=True)[0]
            #for binary classification
            outputs[outputs>0.5] = 1
            outputs[outputs<=0.5] = 0
            #print(outputs)
            #print(labels)
            val_metric.update(labels.cpu().numpy(), outputs.cpu().numpy())
        
        iou_ += calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], 2)
    mean_iu = iou_/len(val_loader)   

    print('[mean iu %.4f]' % (mean_iu)) 
    print(val_metric.get_results())
    net.train()
    criterion.to(device)


if __name__ == '__main__':
    main()








