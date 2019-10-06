import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os

#define the model
model = models.resnet101(pretrained=True)
model.fc=nn.Linear(model.fc.in_features, 2)
