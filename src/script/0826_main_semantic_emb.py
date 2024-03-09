#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings(action='ignore') 
warnings.simplefilter(action='ignore', category=DeprecationWarning)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
os.environ['WANDB_SILENT']="true"

import sys 
sys.path.append('/home/mjh319/workspace/flow/flows_ood/')

from datasets import DatasetBuilder, psd 
from models import ResNet, ResidualBlock, VAE, ClusterAssignment
from util import FlowLoss, clip_grad_norm, get_loss_vals, FlowLossList,get_radius,init_center_c
import pickle

# Train 수정중.. 
# from train import train_8 as train
# from train import train_9 as train
from train import train_15 as train

import argparse
from datetime import timezone, timedelta, datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import tqdm
import wandb



parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='Temp')
parser.add_argument('--project_name', type = str, default='temp_project')
parser.add_argument("--local_save", default=True, action="store_true")
parser.add_argument('--local_path', type = str, default='/home/mjh319/saves/')
parser.add_argument("--wandb", default=False, action="store_true")

parser.add_argument('--im_size', default=224, type=int)
parser.add_argument('--centercrop', default=224, type=int)

parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_epochs', default=350, type=int)
parser.add_argument('--max_grad_norm', default=10., type=float)
parser.add_argument('--save', default='none', type=str)

parser.add_argument('--lr_emb', default=1e-6, type=float)
parser.add_argument('--lr_psd', default=1e-6, type=float)
parser.add_argument('--lr_semantic', default=1e-6, type=float)
parser.add_argument('--lr_joint', default=1e-6, type=float)
parser.add_argument('--weight_decay', default=5e-5, type=float)

parser.add_argument('--class_ratio', default=1.0, type=float)
parser.add_argument('--psd_ratio', default=1.0, type=float)
parser.add_argument('--seman_ratio', default=1.0, type=float)
parser.add_argument('--joint_ratio', default=1.0, type=float)
parser.add_argument('--joint2_ratio', default=1.0, type=float)

parser.add_argument("--emb_train", default=False, action="store_true")
parser.add_argument("--psd_train", default=False, action="store_true")
parser.add_argument("--seman_train", default=False, action="store_true")
parser.add_argument("--joint_train", default=False, action="store_true")

parser.add_argument('--name', type = str, default='celeba')
parser.add_argument('--train_data_name', type = str, default='cifar100')
parser.add_argument('--noise_data_name', type = str, default='imagenet')
parser.add_argument('--nu', default=0.1, type=float)
parser.add_argument('--list_len', default=10, type=int)
parser.add_argument('--dist_normal', default=1., type=float)

parser.add_argument('--psd_feature_dim', default=10, type=int)
parser.add_argument('--semantic_feature_dim', default=64, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epoch_freq', default=30, type=int)
parser.add_argument('--range_of_radius', default=1., type=float)
parser.add_argument('--sum_loss', default=1., type=float)
parser.add_argument('--assign_loss', default=1., type=float)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)

dataset_path = dict()
dataset_path['celeba'] ='~/data/celebA/'
dataset_path['imagenet'] ='~/data/csi/'
dataset_path['cifar10'] ='~/data/pytorch/'
dataset_path['cifar100'] ='~/data/pytorch/'
dataset_path['svhn'] ='~/data/svhn/'
dataset_path['cub'] ='~/data/csi/'
dataset_path['dtd'] ='~/data/csi/'
dataset_path['imagenet_resize'] ='~/data/csi/'
dataset_path['mvtec'] ='~/data/mvtec_anomaly_detection/'
dataset_path['lsun'] = '/home/mjh319/data/lsun/'

dataset_path['lsun_resize'] = '~/data/csi/'
dataset_path['lsun_fix'] = '~/data/csi/' 
dataset_path['imagenet_fix'] = '~/data/csi/'
dataset_path['stanford_dogs'] = '~/data/csi/'
dataset_path['flowers102'] ='~/data/csi/'
dataset_path['places365'] ='~/data/csi/'
dataset_path['food_101'] ='~/data/csi/' 
dataset_path['caltech_256'] = '~/data/csi/'
dataset_path['pets'] ='~/data/csi/'
args.dataset_path = dataset_path


tz = timezone(timedelta(hours=9))
now = datetime.now(tz)

today = '%s_%s'%(now.month, now.day)
second = '_%s:%s:%s'%(now.hour, now.minute, now.second)

print("Today {} ".format(today + second))

exp = args.exp
gpu = "(g{})_".format(args.gpu)
project_name = args.project_name
run_name = exp + gpu + today + second

log_dir_path = args.local_path

if args.local_save == True:
    log_dir_path = Path(log_dir_path, today + second)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    print(log_dir_path)
    
if args.wandb == True:
    wandb.init(project=project_name, reinit=True)
    wandb.run.name = run_name
    wandb.config.update(args)
    print("wnandb")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_builder = DatasetBuilder(args)
dataset = dataset_builder()

dataset_loader = dict()
for i in dataset:
    print(i)
    dataset_loader[i] = torch.utils.data.DataLoader(dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
eval_loader = dict()
for i in dataset:
    eval_loader[i] = torch.utils.data.DataLoader(dataset[i], batch_size=1, shuffle=True, num_workers=4, pin_memory=True)




# psd_path = "/home/mjh319/saves/ood/psd/"
# psd_path = psd_path +str(args.train_data_name)+"_"+str(args.im_size)+"_"+str(args.centercrop)

# with open(psd_path, 'rb') as file:
#     pca_psd = pickle.load(file)

emb_model = ResNet(ResidualBlock, [2, 2, 2], num_classes=100).to(device)
# emb_model = VAE(image_channels=3)
emb_model = emb_model.to(device)


# # args.semantic_feature_dim = 2*args.semantic_feature_dim

D = int(args.psd_feature_dim)
D = int(D)
prior_psd = distributions.MultivariateNormal(torch.zeros(int(args.psd_feature_dim)).cuda(),
                                             torch.eye(int(args.psd_feature_dim)).cuda())
# import pdb;pdb.set_trace()

D = args.semantic_feature_dim
D = int(D)
prior_semantic = distributions.MultivariateNormal(torch.zeros(int(args.semantic_feature_dim)).cuda(),
                                             torch.eye(int(args.semantic_feature_dim)).cuda())



loss_semantic = FlowLoss(prior_semantic)
flowlosslist_semantic = FlowLossList(prior_semantic)
loss_psd = FlowLoss(prior_psd)
flowlosslist_psd = FlowLossList(prior_psd)
# loss_fn3 = FlowLoss(prior3)
mse_loss = nn.MSELoss()
entropy_fn = nn.CrossEntropyLoss()


from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.realnvp import RealNVPTabular_MaskCheckerboard

loader = dataset_loader[args.train_data_name]
noise_dataset = args.noise_data_name

# emb_model.load_state_dict(torch.load("/home/mjh319/workspace/flow/flows_ood/experiments/train_flows/vae.ckpt"))
# emb_model.load_state_dict(torch.load("/home/mjh319/saves/(8_18)(12:47:47)c0818_04_0_15/_emb_net_62.ckpt"))
# emb_model.load_state_dict(torch.load("/home/mjh319/saves/(8_20)(8:51:34)c0820_04_0_15/_emb_net_62.ckpt"))

# /home/mjh319/saves/(8_20)(8:51:34)c0820

# for param in emb_model.parameters():
#     param.requires_grad = False




# cluster_centers = torch.load( "/home/mjh319/workspace/flow/flows_ood/experiments/train_flows/cluster_centers.ckpt")

c_list = []
c_list_tensor = []
for i in range( args.list_len):
    t = torch.normal(0.0, args.dist_normal, size=(1,512))
    t = t / torch.norm(t)
    t = args.range_of_radius*t
    c_list.append(t.to(device))
    c_list_tensor.append(t.numpy())
    
c_list_tensor = np.asarray(c_list_tensor)
c_list_tensor = torch.Tensor(c_list_tensor).to(device)

    
R_list = []
for i in range(args.list_len):
    R_list.append(torch.tensor(0.0, device=device))     

assignment = ClusterAssignment(
    cluster_number = args.list_len,
    embedding_dimension = 512,
    alpha=1.0,
    cluster_centers = c_list_tensor.squeeze()
) 


assignment = assignment.to(device)



        
print("train {}, noise {}".format(args.train_data_name, args.noise_data_name))

# args.lr_semantic = 5e-6
semantic_net = RealNVPTabular_MaskCheckerboard(num_coupling_layers=64, in_dim=args.semantic_feature_dim, num_layers=64, hidden_dim=256)
semantic_net = semantic_net.to(device)


# print("semantic_net contains {} parameters".format(sum([p.numel() for p in semantic_net.parameters()])))

# semantic_net.train()
for param in semantic_net.parameters():
    param.requires_grad = False

param_groups = [
#                 {'params':semantic_net.parameters(),'lr':args.lr_semantic, 'weight_decay':args.weight_decay},
                {'params':emb_model.parameters(),'lr':0.0001, 'weight_decay':args.weight_decay}
               ]
optimizer = optim.Adam(param_groups)


# basemodel no load 
nu = args.nu
 

# scores_a = -iid_score
# scores_b = t
# compute_auc_for_scores(scores_a, scores_b)
from sklearn.metrics import roc_auc_score, average_precision_score
def compute_auc_for_scores(scores_a, scores_b):
    auc = roc_auc_score(
        np.concatenate((np.zeros_like(scores_a),
                       np.ones_like(scores_b)),
                      axis=0),
        np.concatenate((scores_a,
                       scores_b,),
                      axis=0))
    return auc

for epoch in range(args.num_epochs):
    epoch_losses = 0
    epoch_losses_nll = 0
    epoch_losses_noise = 0
    epoch_losses_seman = 0
    epoch_losses_sv = 0
    correct = 0
    
    losses_ = train(args, epoch, dataset_loader, loader, emb_model, loss_semantic, optimizer, semantic_net, noise_dataset, nu, c_list, R_list, args.max_grad_norm, device, wandb, assignment)
    

    save_path_emb = Path(log_dir_path, '_emb_net_'+str(epoch)+'.ckpt')
    torch.save(emb_model.state_dict(), save_path_emb)
#     save_path_seman = Path(log_dir_path, '_seman_net_'+str(epoch)+'.ckpt')
#     torch.save(semantic_net.state_dict(), save_path_seman)    
    

    if epoch % 500 ==0 and epoch != 0:
        save_path_emb = Path(log_dir_path, '_emb_net_'+str(epoch)+'.ckpt')
        torch.save(emb_model.state_dict(), save_path_emb)
#         save_path_seman = Path(log_dir_path, '_seman_net_'+str(epoch)+'.ckpt')
#         torch.save(semantic_net.state_dict(), save_path_seman)
        
        iid_score = get_loss_vals(flowlosslist_semantic, eval_loader[args.train_data_name], emb_model, semantic_net, device, obtion_='semantic')
        iid_score[~np.isfinite(iid_score)] = 300000
        for loader__ in dataset_loader:  

            t = get_loss_vals(flowlosslist_semantic, eval_loader[loader__], emb_model, semantic_net, device, obtion_='semantic')
            t[~np.isfinite(t)] = 300000           
            print("[{}]auc[{:.2f}]mean[{:.2f}]std[{:.2f}]min[{:.2f}]max[{:.2f}]".format(loader__, compute_auc_for_scores(-iid_score, -t)*100,\
                                                                      np.mean(t), np.std(t), t.min(), t.max()))        
            if args.wandb == True:
                wandb.log({loader__ + " loss": np.mean(t)})    
        
        
        
        
        
    if args.wandb == True:   
        for tag, parm in emb_model.named_parameters():
            if "layer1.0.conv1.weight" in tag or "layer3.0.conv1.weight" in tag:
                wandb.log({"emb_"+tag: wandb.Histogram(parm.cpu().detach().numpy())})
        for tag, parm in semantic_net.named_parameters():
            if "body.0.st_net.0.weight" in tag or "body.60.st_net.0.weight" in tag :
                wandb.log({"seman_"+tag: wandb.Histogram(parm.cpu().detach().numpy())})
#             for tag, parm in joint_net.named_parameters():
#                 if "body.0.st_net.0.weight" in tag or "body.60.st_net.0.weight" in tag :
#                     wandb.log({"joint_"+tag: wandb.Histogram(parm.cpu().detach().numpy())})
#             for tag, parm in psd_net.named_parameters():
#                 if "body.0.st_net.0.weight" in tag or "body.60.st_net.0.weight" in tag :
#                     wandb.log({"psd_"+tag: wandb.Histogram(parm.cpu().detach().numpy())})







if args.wandb == True:
    wandb.finish()






















