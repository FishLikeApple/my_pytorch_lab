import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader 
import numpy as np
import torchvision
from torchvision import transforms
import os
from dataset import SteelDataset
import time
from tqdm import tqdm
import pandas as pd

#get input args 
import argparse
parser = argparse.ArgumentParser(description='Semantic Segmentation')
parser.add_argument('--test_dataset', default='./data/test_images', type=str, help='config file path')
parser.add_argument('--list_test', default='./data/test.csv', type=str)
parser.add_argument('--batch_size', default=None, type=int)
parser.add_argument('--checkpoint', default=None, type=str)
parser.add_argument('--num_class', default=2, type=int)
parser.add_argument('--num_workers', default=1, type=int)
parser.add_argument('--n_folds', default=0, type=int)
parser.add_argument('--clearing_steps', default=12, type=int)
args = parser.parse_args()

#define some hyperparameters
num_classes = args.num_class

#define the models, optimizers and datasets
test_dataset = SteelDataset(root_dataset = args.test_dataset, list_data = args.list_test, phase='test')
test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers=args.num_workers)
if args.n_folds <= 0:
    #valid_dataset = SteelDataset(root_dataset = args.train_dataset, list_data = args.list_train, phase='valid')
    models = [torchvision.models.resnet101(pretrained=True)]
    models[-1].fc=nn.Linear(models[-1].fc.in_features, num_classes)
    models[-1] = models[-1].cuda()
    if args.checkpoint != None:
        models[-1].load_state_dict(torch.load(args.checkpoint+'_'+str(0))['state_dict']) 
else:
    models = []
    for i in range(args.n_folds): 
        models.append(torchvision.models.resnet101(pretrained=True))
        models[-1].fc=nn.Linear(models[-1].fc.in_features, num_classes)
        models[-1] = models[-1].cuda()
        if args.checkpoint != None:
            models[-1].load_state_dict(torch.load(args.checkpoint+'_'+str(i))['state_dict'])
               
submission = pd.read_csv(args.list_test)    
    
def test(data_loader, models):
    for model in models: 
        model.eval()
    for img, segm, img_id in tqdm(data_loader):
        img = img.cuda()
        
        output = models[-1](img).cpu().detach().numpy()
        for model in models[1:]:
            output += model(img).cpu().detach().numpy()
        print(output)
        a = 1/0
        submission.loc[submission['ImageId_ClassId']==img_id[0]+'_'+str(type), 'EncodedPixels'] = rle
            
    submission.to_csv(args.submission)
                
test(test_loader, models)     
