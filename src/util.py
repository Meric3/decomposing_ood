import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from datasets import psd 
from sklearn.metrics import roc_auc_score, average_precision_score
import tqdm
class FlowLoss(nn.Module):
    """Get the NLL loss for a RealNVP model.

    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.

    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """

    def __init__(self, prior, k=256):
        super().__init__()
        self.k = k
        self.prior = prior

    def forward(self, z, sldj, y=None, mean=True, vector=None):
        z = z.reshape((z.shape[0], -1))
        if y is not None:
            prior_ll = self.prior.log_prob(z, y)
        else:
            prior_ll = self.prior.log_prob(z)
        corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:]) 

        ll = corrected_prior_ll + sldj
        nll = -ll.mean() if mean else -ll
        
        if vector is not None:
            return nll, -ll

        return nll

class FlowLossList(nn.Module):
    def __init__(self, prior, k=256):
        super().__init__()
        self.k = k
        self.prior = prior

    def forward(self, z, sldj):
        z = z.reshape((z.shape[0], -1))
        prior_ll = self.prior.log_prob(z)
        corrected_prior_ll = prior_ll - np.log(self.k) * np.prod(z.size()[1:])
        ll = corrected_prior_ll + sldj
        return ll
    

def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)

def get_loss_vals(loss_fn, loader, emb_net, net, device, transform_=None, testmode=True, obtion_ = None):
    if testmode:
        net.eval()
    else:
        net.train()
    loss_vals = []
    
    with torch.no_grad():
        
        for  i, (images, labels)  in enumerate(loader):  
            images = images.to(device)               
            if obtion_ == "psd":
                x_psd = psd(images, images.shape[0]).numpy()
                try:
                    x_psd = torch.from_numpy(transform_.transform(x_psd)).to(device)
                except:
                    continue
                z = net(x_psd)
                sldj = net.logdet()
            elif obtion_ == "semantic":
                x_embeddings = emb_net.embed(images)
                if transform_ != None :
                    t = transform_.transform(x_embeddings.cpu().numpy())
                    x_embeddings = torch.from_numpy(t).to(device)
                z = net(x_embeddings)
                sldj = net.logdet()
            elif obtion_ == "joint":
                x_embeddings = emb_net.embed(images) 
                z = net(torch.hstack([x_psd, x_embeddings]))
                sldj = net.logdet()
            else:
                print('obtion')
                break

            losses = loss_fn(z, sldj=sldj)
            loss_vals.extend([loss.item() for loss in losses])
            
#             if i > 200:
#                 break
           
    return np.array(loss_vals)

def get_loss_vals_combine(loss_psd, loss_seman, loader, emb_net, psd_net, semantic_net, device, transform_psd, transform_seman, testmode=True):

    psd_net.eval()
    semantic_net.eval()

    loss_vals_combine = []
    loss_vals_psd = []
    loss_vals_seman = []
    loss_vlas_2 = []
    loss_vlas_4 = []
    loss_vlas_6 = []
    loss_vlas_8 = []
    train_loader = tqdm.tqdm(loader)
    with torch.no_grad():
        
        for  i, (images, labels)  in enumerate(train_loader):  
            images = images.to(device) 
            x_psd = psd(images, images.shape[0]).numpy()
            x_psd = torch.from_numpy(transform_psd.transform(x_psd)).to(device)
#             import pdb; pdb.set_trace()
            z_ = psd_net(x_psd)
            sldj_ = psd_net.logdet()
            losses_psd = loss_psd(z_, sldj=sldj_)
            
            x_embeddings = emb_net.embed(images)
            t = transform_seman.transform(x_embeddings.cpu().detach().numpy())
            x_embeddings = torch.from_numpy(t).to(device)
            
            z = semantic_net(x_embeddings)
            sldj = semantic_net.logdet()
            losses_seman = loss_seman(z, sldj=sldj)

            loss_vals_combine.extend([0.5*loss.item()+0.5*loss2.item() for loss,loss2 in zip(losses_psd, losses_seman)])
            loss_vals_psd.extend([1.*loss.item() for loss in losses_psd])
            loss_vals_seman.extend([1.*loss.item() for loss in losses_seman])
            loss_vlas_2.extend([0.2*loss.item()+0.8*loss2.item() for loss,loss2 in zip(losses_psd, losses_seman)])
            loss_vlas_4.extend([0.4*loss.item()+0.6*loss2.item() for loss,loss2 in zip(losses_psd, losses_seman)])
            loss_vlas_6.extend([0.6*loss.item()+0.4*loss2.item() for loss,loss2 in zip(losses_psd, losses_seman)])
            loss_vlas_8.extend([0.8*loss.item()+0.2*loss2.item() for loss,loss2 in zip(losses_psd, losses_seman)])
            
            if i > 3000:
                break
            
          
    return np.array(loss_vals_combine) , np.array(loss_vals_seman), np.array(loss_vals_psd),\
            np.array(loss_vlas_2),np.array(loss_vlas_4),np.array(loss_vlas_6),np.array(loss_vlas_8)


def get_loss_vals_const(loss_fn, loader,  net, device, testmode=True, obtion_ = None):
    if testmode:
        net.eval()
    else:
        net.train()
    loss_vals = []
    
    with torch.no_grad():
        
        for  i, (images, labels)  in enumerate(loader):  
            images = images.to(device)   

            
            if obtion_ == "psd":
                z = net(images)
                sldj = net.logdet()

            losses = loss_fn(z, sldj=sldj)
            loss_vals.extend([loss.item() for loss in losses])

    return np.array(loss_vals)


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    
    return np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)

def init_center_c( train_loader, net, device,eps=0.1):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(64, device=device)

    net.eval()
    with torch.no_grad():
        for data in train_loader:
            # get the inputs of the batch
            inputs, _ = data
            inputs = inputs.to(device)
#             import pdb; pdb.set_trace()
            outputs = net.embed(inputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)

    c /= n_samples

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c[(abs(c) < eps) & (c < 0)] = -eps
    c[(abs(c) < eps) & (c > 0)] = eps

    return c



def compute_auc_for_scores(scores_a, scores_b):
    auc = roc_auc_score(
        np.concatenate((np.zeros_like(scores_a),
                       np.ones_like(scores_b)),
                      axis=0),
        np.concatenate((scores_a,
                       scores_b,),
                      axis=0))
    return auc