import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader 
import numpy as np
import torchvision
from torchvision import transforms
import os
from dataset import CloudDataset
import time
from tqdm import tqdm

#get input args 
import argparse
parser = argparse.ArgumentParser(description='classification')
parser.add_argument('--train_dataset', default='./data/train_images', type=str, help='config file path')
parser.add_argument('--list_train', default='./data/train.csv', type=str)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--new_checkpoint_path', default='', type=str)
parser.add_argument('--lr', default=5e-4, type=float)
parser.add_argument('--epoch_start', default=0, type=int)
parser.add_argument('--num_epoch', default=200, type=int)
parser.add_argument('--num_class', default=4, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--n_folds', default=0, type=int)
parser.add_argument('--clearing_steps', default=12, type=int)
args = parser.parse_args()

#define the models, optimizers and datasets
if args.n_folds <= 0:
    train_dataset = CloudDataset(root_dataset = args.train_dataset, list_data = args.list_train, phase='train')
    train_loaders = [DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)]
    #valid_dataset = SteelDataset(root_dataset = args.train_dataset, list_data = args.list_train, phase='valid')
    valid_loaders = [] #[DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)]
    models = [torchvision.models.resnet101(pretrained=True)]
    models[-1].fc=nn.Linear(models[-1].fc.in_features, args.num_class)
    models[-1].add_module('output', nn.Sigmoid())
    models[-1] = models[-1].cuda()
    if args.checkpoint != None:
        models[-1].load_state_dict(torch.load(args.checkpoint+'_'+str(0))['state_dict']) 
    optimizers = [optim.Adam(models[-1].parameters(), lr=args.lr)]
else:
    train_loaders = []
    valid_loaders = []
    models = []
    optimizers = []
    for i in range(args.n_folds):
        train_dataset = CloudDataset(root_dataset = args.train_dataset, list_data = args.list_train, 
                                     phase='train', fold_i=i, n_folds=args.n_folds)
        train_loaders.append(DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers))
        valid_dataset = CloudDataset(root_dataset = args.train_dataset, list_data = args.list_train, 
                                     phase='valid', fold_i=i, n_folds=args.n_folds)
        valid_loaders.append(DataLoader(valid_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers))
        models.append(torchvision.models.resnet101(pretrained=True))
        models[-1].fc=nn.Linear(models[-1].fc.in_features, args.num_class)
        models[-1].add_module('output', nn.Sigmoid())
        models[-1] = models[-1].cuda()
        if args.checkpoint != None:
            models[-1].load_state_dict(torch.load(args.checkpoint+'_'+str(i))['state_dict'])
        optimizers.append(optim.Adam(models[-1].parameters(), lr=args.lr))
        
loss_fn = nn.functional.binary_cross_entropy_with_logits

def train(data_loader, model, optimizer):
    model.train()
    total_loss = 0
    accumulation_steps = 32 // args.batch_size
    optimizer.zero_grad()
    for idx, (img, cls, _) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        cls = cls.cuda()
        outputs = model(img)
        loss = loss_fn(outputs, cls)
        (loss/accumulation_steps).backward()
        clipping_value = 1.0
        nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        if (idx + 1 ) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            #with open(args.new_checkpoint_path+'logs.txt', 'a') as f:
            print('idx:'+str(idx)+'  last_loss:'+str(loss))
        total_loss += loss.item()
        
        # delete caches
        del img, cls, outputs, loss
        torch.cuda.empty_cache()
            
    return total_loss/len(data_loader)

def valid(data_loader, model):
    model.eval()
    num_correct = 0
    accumulation_steps = 32 // args.batch_size
    for idx, (img, cls, _) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        cls = cls.cuda()
        outputs = model(img)
        
        preds = outputs.argmax(dim=1)
        num_correct += torch.eq(preds, cls).sum().float().item()
        
        # delete caches
        del img, cls, outputs, preds
        torch.cuda.empty_cache()
            
    return num_correct/len(data_loader)
  
for epoch in range(args.epoch_start, args.epoch_start+args.num_epoch):
    start_time = time.time()
    corrections_train = []
    
    for i in range(len(train_loaders)):
        loss_train = train(train_loaders[i], models[i], optimizers[i])
        if len(valid_loaders) > 0:
            corrections_train.append(valid(valid_loaders[i], models[i]))
        print('[TRAIN] Epoch: {}| Fold: {}| Loss: {}| Time: {}'.format(epoch, i, loss_train, time.time()-start_time))
        state = {
        "status": 'not used',
        "epoch": epoch,
        "arch": 'res101_' + str(i),
        "state_dict": models[i].state_dict()
        }
        torch.save(state, '{}{}_checkpoint_{}.pth_{}'.format(args.new_checkpoint_path, 'res101', epoch, i))
    
    if len(valid_loaders) > 0:
        print('correction:'+str(sum(corrections_train)/len(corrections_train)))
     
