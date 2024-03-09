import logging
import argparse
import pickle

import numpy as np
import torch
from torch import distributions
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from datasets import DatasetBuilder, psd 
from models import ResNet, ResidualBlock, VAE, ClusterAssignment
from util import FlowLoss, clip_grad_norm,  FlowLossList,get_radius,init_center_c, compute_auc_for_scores, get_loss_vals_combine

from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.realnvp import RealNVPTabular_MaskCheckerboard

logging.basicConfig(format='%(message)s', level=logging.INFO)
device = torch.device('cpu')



parser = argparse.ArgumentParser()

parser.add_argument('--exp', type = str, default='Temp')
parser.add_argument('--project_name', type = str, default='temp_project')
parser.add_argument("--local_save", default=True, action="store_true")
parser.add_argument('--local_path', type = str, default='/home/mjh319/saves/ood/combine/')
parser.add_argument("--wandb", default=False, action="store_true")

parser.add_argument('--im_size', default=112, type=int)
parser.add_argument('--centercrop', default=112, type=int)

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
parser.add_argument('--train_data_name', type = str, default='cifar10')
parser.add_argument('--nu', default=0.1, type=float)
parser.add_argument('--list_len', default=10, type=int)
parser.add_argument('--dist_normal', default=1, type=float)

parser.add_argument('--psd_feature_dim', default=56, type=int)
parser.add_argument('--semantic_feature_dim', default=56, type=int)

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epoch_freq', default=30, type=int)
parser.add_argument('--select', default=2, type=int)


args = parser.parse_args()

dataset_path = dict()
dataset_path['celeba'] ='~/data/celebA/'
dataset_path['imagenet'] ='~/data/ood/'
dataset_path['cifar10'] ="/data/CIFAR/cifar-10"
dataset_path['cifar100'] ='~/data/pytorch/'
dataset_path['svhn'] ="/code/svhn"
dataset_path['cub'] ='~/data/csi/'
dataset_path['dtd'] ='~/data/csi/'
dataset_path['imagenet_resize'] ='~/data/csi/'
dataset_path['tiny_imagenet'] ='~/data/tiny_imagenet/'
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
args.dataset_path = dataset_path


dataset_builder = DatasetBuilder(args)
dataset = dataset_builder()

eval_loader = dict()
for i in dataset:
    eval_loader[i] = torch.utils.data.DataLoader(dataset[i], batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)

psd_path = "/code/saves/psd_f/"
base_path = "/code/saves/backbone/" + str(args.train_data_name)
psd_path = psd_path +str(args.train_data_name)+"_"+str(args.im_size)+"_"+str(args.centercrop)
with open(psd_path, 'rb') as file:
    transform_psd = pickle.load(file)



path = base_path+"/seman_pca.pkl"
with open(path, 'rb') as file:
    transform_seman = pickle.load(file)

emb_model = ResNet(ResidualBlock, [2, 2, 2], num_classes=100, size = args.im_size).to(device)
emb_model = emb_model.to(device)
emb_model.load_state_dict(torch.load(base_path+"/emb_net.ckpt", map_location=torch.device('cpu')))

prior_psd = distributions.MultivariateNormal(torch.zeros(int(args.psd_feature_dim)),
                                             torch.eye(int(args.psd_feature_dim)))
prior_semantic = distributions.MultivariateNormal(torch.zeros(int(args.semantic_feature_dim)),
                                             torch.eye(int(args.semantic_feature_dim)))



loss_semantic = FlowLoss(prior_semantic)
flowlosslist_semantic = FlowLossList(prior_semantic)
loss_psd = FlowLoss(prior_psd)
flowlosslist_psd = FlowLossList(prior_psd)

c_list = []
c_list_tensor = []
for i in range( args.list_len):
    t = torch.normal(0.0, 1.0, size=(1,512))
    t = t / torch.norm(t)
    t = 3.0*t
    c_list.append(t.to(device))
    c_list_tensor.append(t.numpy())
    
c_list_tensor = np.asarray(c_list_tensor)
c_list_tensor = torch.Tensor(c_list_tensor).to(device)

    
R_list = []
for i in range(args.list_len):
    R_list.append(torch.tensor(0.0, device=device))     

assignment = ClusterAssignment(
    cluster_number = 10,
    embedding_dimension = 512,
    alpha=1.0,
    cluster_centers = c_list_tensor.squeeze()
) 


assignment = assignment.to(device)

psd_net = RealNVPTabular_MaskCheckerboard(num_coupling_layers=8, in_dim=args.psd_feature_dim, num_layers=8, hidden_dim=16)
psd_net = psd_net.to(device)
semantic_net = RealNVPTabular_MaskCheckerboard(num_coupling_layers=8, in_dim=args.semantic_feature_dim, num_layers=8, hidden_dim=16)
semantic_net = semantic_net.to(device)


psd_net.load_state_dict(torch.load(base_path+"/psd_net.ckpt", map_location=torch.device('cpu')))
semantic_net.load_state_dict(torch.load(base_path+"/seman_net.ckpt", map_location=torch.device('cpu')))




iid_score, iid_score_psd, iid_score_seman,iid_2,iid_4,iid_6,iid_8  = get_loss_vals_combine(flowlosslist_psd, flowlosslist_semantic, 
                                  eval_loader[args.train_data_name], emb_model,
                                  psd_net, semantic_net, device,transform_psd,  transform_seman)

iid_score[~np.isfinite(iid_score)] = 300000
iid_score_psd[~np.isfinite(iid_score_psd)] = 300000
iid_score_seman[~np.isfinite(iid_score_seman)] = 300000


for loader__ in eval_loader:  
    if loader__ != args.train_data_name:
        comb, psd, seman,_2,_4,_6,_8 = get_loss_vals_combine(flowlosslist_psd, flowlosslist_semantic, 
                                  eval_loader[loader__], emb_model, psd_net, 
                                  semantic_net, device, transform_psd, transform_seman)
        comb[~np.isfinite(comb)] = 300000     
        psd[~np.isfinite(psd)] = 300000      
        seman[~np.isfinite(seman)] = 300000      
        print("psd[{}]auc[{:.2f}]mean[{:.2f}]std[{:.2f}]min[{:.2f}]max[{:.2f}]".\
              format(loader__, compute_auc_for_scores(-iid_score_psd, -psd)*100,\
                                                                      np.mean(psd), np.std(psd), psd.min(), psd.max())) 
        print("seman[{}]auc[{:.2f}]mean[{:.2f}]std[{:.2f}]min[{:.2f}]max[{:.2f}]".\
              format(loader__, compute_auc_for_scores(-iid_score_seman, -seman)*100,\
                                                                      np.mean(seman), np.std(seman), seman.min(), seman.max()))      
        print("comb[{}]auc[{:.2f}]mean[{:.2f}]std[{:.2f}]min[{:.2f}]max[{:.2f}]".\
              format(loader__, compute_auc_for_scores(-iid_score, -comb)*100,\
                                                                      np.mean(comb), np.std(comb), comb.min(), comb.max()))        


logging.info("\n\n Finish")