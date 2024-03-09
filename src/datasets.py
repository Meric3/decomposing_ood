import os 
import scipy.fftpack as fp
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
import torch

# lsun
from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.lsun import LSUNClass

from torch.utils.data import Dataset

import pickle


import shutil
from PIL import Image
import codecs
from typing import Any, Callable, List, Iterable, Optional, TypeVar, Dict, IO, Tuple, Iterator
import string
import zipfile
import bz2
import gzip

from os.path import join



    
class DatasetBuilder(object):
    def __init__(self, args, **kwargs):
        self.args = args
        self.im_size = args.im_size
        self.centercrop = args.centercrop
        
    def __call__(self, optional_transform=None):
        transform = self._get_transform(optional_transform=optional_transform)
        transform_c = self._get_transform_c(optional_transform=optional_transform)
        datasets_ = dict()

        datasets_['cifar10'] = datasets.CIFAR10(root=self.args.dataset_path['cifar10'], train=False, transform=transform, download=False)
        datasets_['svhn'] = datasets.SVHN(root=self.args.dataset_path['svhn'], split='test', transform=transform_c, download=False)

          
        return datasets_

    def _get_transform(self, **kwargs):
        transform = []
        optional_transform = kwargs['optional_transform']
        transform.extend([
                transforms.Resize(self.im_size),
                transforms.CenterCrop(self.centercrop),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
     

        return transforms.Compose(transform)
    
    def _get_transform_c(self, **kwargs):
        transform = []
        transform.extend([
                transforms.Resize(self.im_size),
                transforms.CenterCrop(self.centercrop),
                transforms.ToTensor(),
                transforms.Normalize((-2.5, -2.5, -2.5), (4.5, 4.5, 4.5))
            ])  
        
        return transforms.Compose(transform)

    
def psd(x, batch_size):
    F1 = fp.fft2(x[:,0].cpu().detach().numpy())
    F1 = fp.fftshift(F1)
    (w, h) = x[0,0].shape
    half_w, half = int(w/2), int(h/2)
    
    psd_list = []
    for idx in range(batch_size):
        psd_ = []
        for i in range(half_w):
            psd_.append(abs(F1[idx, half-i:half+1+i:1, half-i:half+1+i:1]).sum())
            F1[idx, half-i:half+1+i:1,half-i:half+1+i:1] = 0
        psd_ = np.asarray(psd_)
        psd_ = (psd_ - psd_.min()) / (psd_.max() - psd_.min()) 
        psd_list.append(np.asarray(psd_))
    
    return torch.FloatTensor(psd_list)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)





