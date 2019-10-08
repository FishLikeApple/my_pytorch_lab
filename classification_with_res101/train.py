import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader 
import numpy as np
import torchvision
from torchvision import models, transforms
import os
from dataset import SteelDataset
import time
from tqdm import tqdm

#get input args 
import argparse
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--train_dataset', default='./data/train_images', type=str, help='config file path')
parser.add_argument('--list_train', default='./data/train.csv', type=str)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--new_checkpoint_path', default='', type=str)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--epoch_start', default=0, type=int)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_class', default=5, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--clearing_steps', default=12, type=int)
args = parser.parse_args()

#define some hyperparameters
num_classes = 2

#define the model and so on
model = models.resnet101(pretrained=True)
model.fc=nn.Linear(model.fc.in_features, num_classes)
model = model.cuda()
train_dataset = SteelDataset(root_dataset = args.train_dataset, list_data = args.list_train, phase='train')
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss 

def train(data_loader):
    model.train()
    total_loss = 0
    accumulation_steps = 32 // args.batch_size
    optimizer.zero_grad()
    for idx, (img, cls) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        cls = cls.cuda()
        outputs = torch.argmax(model(img))
        print(cls)
        print(outputs)
        loss = loss_fn(outputs, cls)
        (loss/accumulation_steps).backward()
        clipping_value = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        if (idx + 1 ) % accumulation_steps == 0:
            optimizer.step() 
            optimizer.zero_grad()
        total_loss += loss.item()
        
        # delete caches
        del img, cls, outputs, loss
        torch.cuda.empty_cache()
            
    return total_loss/len(data_loader)
  
for epoch in range(args.epoch_start, args.epoch_start+args.num_epoch):
    start_time = time.time()
    loss_train = train(train_loader)
    print('[TRAIN] Epoch: {}| Loss: {}| Time: {}'.format(epoch, loss_train, time.time()-start_time))
    state = {
    "status": 'not used',
    "epoch": epoch,
    "arch": 'res101',
    "state_dict": model.state_dict()
    }
    torch.save(state, '{}{}_checkpoint_{}.pth'.format(args.new_checkpoint_path, arch, epoch))
