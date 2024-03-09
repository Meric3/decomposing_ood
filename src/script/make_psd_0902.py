#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings(action='ignore') 


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.environ['WANDB_SILENT']="true"

import sys 
sys.path.append('/home/mjh319/workspace/flow/flows_ood/')

from datasets import DatasetBuilder, psd 
from models import ResNet, ResidualBlock
from util import FlowLoss, clip_grad_norm, get_loss_vals, FlowLossList,get_radius,init_center_c

import argparse
from datetime import timezone, timedelta, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
import numpy as np

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tqdm




parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='Temp')
parser.add_argument('--im_size', default=32, type=int)
parser.add_argument('--centercrop', default=32, type=int)
parser.add_argument('--batch_size', default=256, type=int)


args = parser.parse_args()

dataset_path = dict()
dataset_path['celeba'] ='~/data/celebA/'
dataset_path['imagenet'] ='~/data/ood/'
dataset_path['cifar10'] ='~/data/pytorch/'
dataset_path['cifar100'] ='~/data/pytorch/'
dataset_path['svhn'] ='~/data/svhn/'
dataset_path['cub'] ='~/data/csi/'
dataset_path['dtd'] ='~/data/csi/'
dataset_path['imagenet_resize'] ='~/data/csi/'
dataset_path['mvtec'] ='~/data/mvtec_anomaly_detection/'
dataset_path['lsun'] = '/home/mjh319/data/lsun'

dataset_path['lsun_resize'] = '~/data/csi/'
dataset_path['lsun_fix'] = '~/data/csi/' 
dataset_path['imagenet_fix'] = '~/data/csi/'
dataset_path['stanford_dogs'] = '~/data/csi/'
dataset_path['flowers102'] ='~/data/csi/'
dataset_path['places365'] ='~/data/csi/'
dataset_path['food_101'] ='~/data/csi/' 
dataset_path['caltech_256'] = '~/data/csi/'
dataset_path['pets'] ='~/data/csi/'

dataset_path['tiny_imagenet'] ='~/data/tiny_imagenet/'
args.dataset_path = dataset_path



print("im size {}  center crop {}".format(args.im_size, args.centercrop))


dataset_builder = DatasetBuilder(args)
dataset = dataset_builder()


dataset_loader = dict()
for i in dataset:
    print(i)
    dataset_loader[i] = torch.utils.data.DataLoader(dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    
psd_dict = dict()
import pickle

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=int(args.im_size/2)))])

import tqdm
for dataset_ in dataset_loader:
    psd_path = "/home/mjh319/saves/ood/psd_f/"
    torch.cuda.empty_cache()    
    print(dataset_)
    loader = tqdm.tqdm(dataset_loader[dataset_])
    psds_ = []


    for batch_idx, (x, y) in enumerate(loader):
        psds = psd(x, x.shape[0]).numpy()
        psds_.append(psds)

    psd_dict[dataset_] = np.asarray(psds_[:-1]).reshape(-1, psds.shape[1])
    transform_ = pipeline.fit(psd_dict[dataset_])
    psd_path = psd_path +str(dataset_)+"_"+str(args.im_size)+"_"+str(args.centercrop)
    
    with open(psd_path, 'wb') as file:    # james.p 파일을 바이너리 쓰기 모드(wb)로 열기
        pickle.dump(transform_, file)
    del transform_
        








