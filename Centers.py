# generate orthogonal  K' bits of K bits

import random
from itertools import combinations

import numpy as np
import torch
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
from scipy.special import comb  # calculate combination

def judge(code_length):
    result = code_length & code_length-1
    if result==0:
        return True
    else:
        return False

def generate_via_hadamard(code_length, num_class):
    ha_d = hadamard(code_length)   # hadamard matrix 
    ha_2d = np.concatenate((ha_d, -ha_d), 0)  # can be used as targets for 2*d hash bit


    if num_class<=code_length:
        hash_targets = torch.from_numpy(ha_d[0:num_class]).float()
        print('hash centers shape: {}'. format(hash_targets.shape))
    elif num_class>code_length:
        hash_targets = torch.from_numpy(ha_2d[0:num_class]).float()
        print('hash centers shape: {}'. format(hash_targets.shape))

    return hash_targets

def generate_via_bernouli(code_length, num_class):
    hash_targets = []
    a = []  # for sampling the 0.5*code_length 
    b = []  # for calculate the combinations of 51 num_class

    for i in range(0, code_length):
        a.append(i)

    for i in range(0, num_class):
        b.append(i)
        
    for j in range(10000):
        hash_targets = torch.zeros([num_class, code_length])
        for i in range(num_class):
            ones = torch.ones(code_length)
            sa = random.sample(a, round(code_length/2))
            ones[sa] = -1
            hash_targets[i]=ones
        com_num = int(comb(num_class, 2))# C(n, 2)
        c = np.zeros(com_num)

        for i in range(com_num):
            i_1 = list(combinations(b, 2))[i][0]
            i_2 = list(combinations(b, 2))[i][1]
            TF = torch.sum(hash_targets[i_1]!=hash_targets[i_2])
            c[i]=TF

        if np.mean(c)>=int(code_length / 2):  # guarantee the hash center are far away from each other in Hamming space, 20 can be set as 18 for fast convergence
            print(min(c))
            # print("stop! we find suitable hash centers")
            break

    return hash_targets

    
def generate_old_centers(K, sub_K, num_class, padding_value=0, padding_mode='append'):
    if padding_mode == 'append':
        if sub_K * 2 >= num_class and judge(sub_K):
            sub_Kbit_centers = generate_via_hadamard(code_length=sub_K, num_class=num_class)   
        else:
            sub_Kbit_centers = generate_via_bernouli(code_length=sub_K, num_class=num_class)
        append_code = torch.ones([num_class, K-sub_K]) * padding_value
        Kbit_centers = torch.cat((sub_Kbit_centers, append_code), 1)
        return Kbit_centers

def generate_new_centers(old_centers, code_length, num_inc):

    def hamming_dist(a, b):
        code_length = a.shape[0]
        return (code_length - a @ b.t()) / (2 * code_length)
    
    o = torch.zeros(num_inc, code_length)
    i = 0
    count = 0
    threshold = 0.6
    N_old = old_centers.shape[0]
    while i < num_inc:
        prob = torch.ones(code_length) * 0.5
        c = torch.bernoulli(prob) * 2.0 - 1.0
        nobreak = True

        for j in range(N_old + i):
            if j < N_old:
                if hamming_dist(c, old_centers[j]) < threshold:
                    if i >= 0:
                        i -= 1
                    nobreak=False
                    break
            else:
                if hamming_dist(c, o[j-N_old]) < threshold:
                    if i >= 0:
                        i -= 1
                    nobreak=False
                    break

        if nobreak:
            o[i] = c
        else:
            count += 1
        if count >= 10000:
            count = 0
            threshold -= 0.05
            if threshold < 0.2:
                print('Cannot find target')
        i += 1
    hashcenters = o[torch.randperm(num_inc)]
    return hashcenters
