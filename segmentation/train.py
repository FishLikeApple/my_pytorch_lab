import os
import cv2
import collections
import time 
import tqdm
from PIL import Image
from functools import partial
train_on_gpu = True

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import albumentations as albu
from albumentations import torch as AT

from catalyst.data import Augmentor
from catalyst.dl import utils
from catalyst.data.reader import ImageReader, ScalarReader, ReaderCompose, LambdaReader
from catalyst.dl.runner import SupervisedRunner
from catalyst.contrib.models.segmentation import Unet
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback

import segmentation_models_pytorch as smp

from args import *
from helper_functions_and_classes import *
from dataset import *

ACTIVATION = None
model = smp.PSPNet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=4, 
    activation=ACTIVATION,
)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

train = pd.read_csv(f'{path}/train.csv')
train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
n_train = len(os.listdir(f'{path}/train_images'))
n_test = len(os.listdir(f'{path}/test_images'))
id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)

train_dataset = CloudDataset(df=train, datatype='train', img_ids=train_ids, transforms = get_training_augmentation(), 
                             preprocessing=get_preprocessing(preprocessing_fn))
valid_dataset = CloudDataset(df=train, datatype='valid', img_ids=valid_ids, transforms = get_validation_augmentation(), 
                             preprocessing=get_preprocessing(preprocessing_fn))
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=num_workers)

loaders = {
    "train": train_loader,
    "valid": valid_loader
}

# model, criterion, optimizer
optimizer = torch.optim.Adam([
    {'params': model.decoder.parameters(), 'lr': 1e-2}, 
    {'params': model.encoder.parameters(), 'lr': 1e-3},  
])
scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
runner = SupervisedRunner()

def custom_train(model, criterion, optimizer, data_loader):
      
    model.train()
    total_loss = 0
    loss_sum = 0
    accumulation_steps = 32 // bs
    optimizer.zero_grad()
    for idx, (img, segm) in enumerate(tqdm(data_loader)):
        img = img.cuda()
        segm = segm.cuda()
        outputs = model(img)
        loss = criterion(outputs, segm)
        (loss/accumulation_steps).backward()
        clipping_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
        loss_sum += loss.item()
        if (idx + 1 ) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print('loss:'+str(loss_sum/accumulation_steps))
            loss_sum = 0
        total_loss += loss.item()
        
        # delete caches
        del img, segm, outputs, loss
        torch.cuda.empty_cache()
            
    return total_loss/len(data_loader)

def evaluate(model, data_loader):
    
    meter = Metric(mode=args.mode)
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (img, segm) in enumerate():
            img = img.cuda() 
            segm = segm.cuda() 
            outputs = model(img) 
            loss = criterion(outputs, segm)
            del img
            del segm
            outputs = outputs.detach().cpu()
            segm = segm.detach().cpu() 
            meter.update(segm, outputs) 
            total_loss += loss.item()
        
        dices, iou = meter.get_metrics() 
        dice, dice_neg, dice_pos = dices 
        torch.cuda.empty_cache() 
        return total_loss/len(data_loader), iou, dice, dice_neg, dice_pos

def train(model, criterion, optimizer, scheduler, loaders, callbacks, logdir, num_epochs, verbose):
    """train function using gradient accumulating"""
    
    for i in range(num_epochs):
        bset_loss = 99999999
        custom_train(model, criterion, optimizer, loaders["train"])
        loss = evaluate(model, loaders["valid"])
        
        if bset_loss >= loss:
            torch.save(model.state_dict(), f"{output_logdir}/checkpoints/best.pth")
            bset_loss = loss
    
#runner.train(
train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
    logdir=logdir,
    num_epochs=num_epochs,
    verbose=True
)

utils.plot_metrics(
    logdir=logdir, 
    # specify which metrics we want to plot
    metrics=["loss", "dice", 'lr', '_base/lr']
)
