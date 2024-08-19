import sys
sys.path.append('.')
from models.model_loader import load_model
import torch
from loguru import logger
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from Centers import generate_new_centers, generate_old_centers
import copy
import os

class HashProxy(nn.Module):
    def __init__(self, temp):
        super(HashProxy, self).__init__()
        self.temp = temp

    def forward(self, X, P, L):
        X = F.normalize(X, p = 2, dim = -1)
        P = F.normalize(P, p = 2, dim = -1)
        
        D = F.linear(X, P) / self.temp

        L /= torch.sum(L, dim=1, keepdim=True).expand_as(L) # onehot label smooth

        xent_loss = torch.mean(torch.sum(-L * F.log_softmax(D, -1), -1))
        return xent_loss

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]#depth = LATENT_SIZE
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD_loss(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def KDLoss(logits, labels, temperature=2.0):
    assert not labels.requires_grad, "output from teacher(old task model) should not contain gradients"
    # Compute the log of softmax values
    outputs = torch.log_softmax(logits/temperature,dim=1)
    labels  = torch.softmax(labels/temperature,dim=1)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

def NSM(feature, weight):
    norms = torch.norm(feature, p=2, dim=-1, keepdim=True)
    nfeat = torch.div(feature, norms)

    norms_c = torch.norm(weight, p=2, dim=-1, keepdim=True)
    ncenters = torch.div(weight, norms_c)
    logits = torch.matmul(nfeat, torch.transpose(ncenters, 0, 1))

    return logits

class BDIH(object):
    def __init__(self, config):
        self.config = config
        self.arch = config.arch
        self.code_length = config.code_length
        self.device = config.device
        
        self.num_class_list = config.num_class_list
        self.valid_length_list = config.valid_length_list
        # generate centers
        self.hashcenters = generate_old_centers( 
                    K=self.code_length,
                    sub_K=self.valid_length_list[0],
                    num_class=self.num_class_list[0],
                    padding_value=1,
        ).to(self.device)
        self.hashcenters.requires_grad_(False)
        
        self.model = load_model(self.arch, self.code_length).to(self.device)
        self.lr = config.lr
        self.multi_lr = 0.05
        # params_list = [{'params': self.model.feature_layers.parameters(), 'lr': self.multi_lr*self.lr}, # 0.05*(args.lr)
        #            {'params': self.model.hash_layer.parameters()}]
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config.lr,
            weight_decay=1e-5,
        )
        self.max_iters = config.method_parameters.max_iters
        self.scheduler = CosineAnnealingLR(self.optimizer, self.max_iters, 1e-7)

        self.lambda_q = config.method_parameters.lambda_q
        self.nf_ratio = config.method_parameters.nf_ratio
        self.lambda_kd = config.lambda_kd


        self.session_id = 0

        self.hash_loss = HashProxy(temp=0.2)
        
    def update(self, session_id, old_model):
        self.session_id = session_id
        self.old_model = old_model.to(self.device)
        self.old_model.eval()
        self.model = copy.deepcopy(old_model)

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, self.max_iters, 1e-7)
        self.old_centers = self.hashcenters.cpu()
        valid_length = self.valid_length_list[session_id]
        self.new_centers = generate_new_centers(self.old_centers[:, :valid_length], valid_length, self.num_class_list[session_id])

        if valid_length<self.code_length:
            append_code = torch.ones([self.num_class_list[session_id], self.code_length-valid_length]) * 1
            self.new_centers = torch.cat((self.new_centers, append_code), 1)
        self.hashcenters = torch.vstack((self.old_centers, self.new_centers)).to(self.device)
        self.hashcenters.requires_grad_(False)

    def train_iter(self, train_dataloader, iter, query_dataloader=None, retrieval_dataloader=None):
        
        for batch, (data, targets, index) in enumerate(train_dataloader):

            data, targets, index = data.to(self.device), targets.to(self.device), index.to(self.device)

            oh_targets = F.one_hot(targets, self.config.num_classes).float()

            f = self.model(data)

            if self.config.nf:
                #* add noise to code
                frozen_idx = list(range(self.valid_length_list[self.session_id], self.code_length))
                neg_bits = -1 * torch.ones(f.shape[0], len(frozen_idx))
                zero_bits = torch.zeros_like(neg_bits)
                p = torch.rand_like(neg_bits)
                noise_bits = torch.zeros_like(f).cpu()
                noise_bits[:, frozen_idx] = torch.where(p<=self.nf_ratio, neg_bits, zero_bits)
                f = f + noise_bits.detach().to(self.device)

            #* loss on short codes
            center_loss = self.hash_loss(f, self.hashcenters, oh_targets[:, :self.hashcenters.shape[0]]) 
            Q_loss = torch.mean((torch.abs(f) - 1.0) ** 2)

            loss = center_loss + self.lambda_q * Q_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        logger.debug(
            '[iter:{}][loss:{:.2f}][center loss:{:.2f}][Q loss:{:.2f}]'.format(
                iter + 1, loss, center_loss,Q_loss))
        return self.model
    

    def inc_train_iter(self, train_dataloader, iter, query_dataloader=None, retrieval_dataloader=None):
        
        for batch, (data, targets, index) in enumerate(train_dataloader):

            data, targets, index = data.to(self.device), targets.to(self.device), index.to(self.device)
 
            oh_targets = F.one_hot(targets, self.config.num_classes).float()

            f = self.model(data)

            if self.config.nf:
                #* add noise to code
                frozen_idx = list(range(self.valid_length_list[self.session_id], self.code_length))
                neg_bits = -1 * torch.ones(f.shape[0], len(frozen_idx))
                zero_bits = torch.zeros_like(neg_bits)
                p = torch.rand_like(neg_bits)
                noise_bits = torch.zeros_like(f).cpu()
                noise_bits[:, frozen_idx] = torch.where(p<=self.nf_ratio, neg_bits, zero_bits)
                f = f + noise_bits.detach().to(self.device)

            #* loss on short codes
            center_loss = self.hash_loss(f, self.hashcenters, oh_targets[:, :self.hashcenters.shape[0]]) 
            Q_loss = torch.mean((torch.abs(f) - 1.0) ** 2)
    
            #* kd loss
            kd_loss = 0.0
            f_old = self.old_model(data)
            old_codes_norm = F.normalize(f_old, p=2, dim=-1)
            hashcenters_norm = F.normalize(self.hashcenters, p=2, dim=-1)
            old_logits = F.softmax(F.linear(old_codes_norm, hashcenters_norm)/0.2, -1)
            confidence = torch.diag(old_logits @ oh_targets[:, :self.hashcenters.shape[0]].float().t()) # Fold对于每个样本分类到其gt hashcenter上的概率
            kd_loss = self.CKD_Loss(f, f_old.detach(), confidence) 


            loss = center_loss + self.lambda_q * Q_loss \
                    + self.lambda_kd * kd_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        logger.debug(
            '[iter:{}][loss:{:.2f}][center loss:{:.2f}][Q loss:{:.2f}][kd loss:{:.2f}]'.format(
                iter + 1, loss, center_loss, Q_loss, kd_loss))
        return self.model
      

    def CKD_Loss(self, codes, old_codes, confidence):
        d = nn.PairwiseDistance(p=2)
        loss = d(old_codes, codes)
        loss = confidence * loss
        return loss.mean()
