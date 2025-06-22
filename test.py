### test
import torch
import torch.nn as nn
from model import Q_net, P_net, D_net_gauss, weight_estimate
from Dataset import Dataload
from scipy.io import savemat
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import argparse
device = torch.device("cuda:0")



parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'data.mat')
parser.add_argument('--batchsize', type = int, default = 32, help = 'training numbers every iteration')
parser.add_argument('--dataroot', type = str, default = 'data')
opt = parser.parse_args()


node = 10
Q_source = Q_net(204, 1024, node).to(device)
Q_target = Q_net(207, 1024, node).to(device)
P_source = P_net(node, 1024, 204).to(device)
P_target = P_net(node, 1024, 207).to(device)
epoch = 300
Q_source.eval()
Q_target.eval()
P_source.eval()
P_target.eval()
Q_source.load_state_dict(torch.load("model/encoder_source_epoch_{0}.pth".format(epoch)))
Q_target.load_state_dict(torch.load("model/encoder_target_epoch_{0}.pth".format(epoch)))
P_source.load_state_dict(torch.load("model/decoder_source_epoch_{0}.pth".format(epoch)))
P_target.load_state_dict(torch.load("model/decoder_target_epoch_{0}.pth".format(epoch)))
source_ = np.zeros((10000, 204))
target_ = np.zeros((10000, 207))
count = 0
db = Dataload(opt.dataroot,  "data31.mat", 'data32.mat', False)
sample = DataLoader(db, batch_size = opt.batchsize, shuffle = False, pin_memory = True)
with torch.no_grad():
    for step, (source, target) in enumerate(sample):
        source = source.type(torch.float32).to(device)
        target = target.type(torch.float32).to(device)
        z_sample = Q_source(source)
        X_sample = P_source(z_sample)
        source_[count : count + X_sample.shape[0], :] = X_sample.cpu().numpy()
        z_sample = Q_target(target)
        X_sample = P_target(z_sample)
        target_[count : count + X_sample.shape[0], :] = X_sample.cpu().numpy()
        count += X_sample.shape[0]

#savemat('result/source/source_result_{0}.mat'.format(i), {"out": np.reshape(source_, [100, 100, 204])})
savemat('result/target/target_result_{0}.mat'.format(i), {"out": np.reshape(target_, [100, 100, 207])})