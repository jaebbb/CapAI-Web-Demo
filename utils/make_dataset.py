import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset
import random
import torch.nn.functional as F
import os
import cv2
from PIL import Image
from glob import glob

class CapAI(Dataset):
    def __init__(self, route_list, transform = None, Path = 'data/testset/'):
        self.transform = transform
        if self.transform == None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(384),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
                )

        self.dataset = []
        for i in route_list:
            self.dataset.extend(glob(f"{Path}{i}/*.jpg"))     

        self.label = {'06.stomach':0, '07.intestineSS':1, '08.intestineSF':2, '09.intestineL':3}

            
    def __len__(self):
        return len(self.dataset)
    
    
    def check_label(self, pth):
        if '06.stomach' in pth:
            return 0
        elif '07.intestineSS' in pth:
            return 1
        elif '08.intestineSF' in pth:
            return 2
        elif '09.intestineL' in pth:
            return 3
        
    

    
    
    def __getitem__(self,index):
        img = Image.open(self.dataset[index]).convert('RGB')
        label = self.check_label(self.dataset[index])

        if self.transform:
            img = self.transform(img)

        return (img,label)
