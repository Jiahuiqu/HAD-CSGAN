import torch
import torch.nn as nn
from model import Q_net, P_net, D_net_gauss, weight_estimate
from Dataset import Dataload
import torch.nn.functional as F
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from scipy.io import savemat
device = torch.device("cuda:0")

"""parameter set"""
parser = argparse.ArgumentParser()
parser.add_argument('--data', type = str, default = 'data.mat')
parser.add_argument('--batchsize', type = int, default = 32, help = 'training numbers every iteration')
parser.add_argument('--dataroot', type = str, default = 'data')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--weight_decay', type = float, default = 1e-5, help = 'weight decay')
parser.add_argument('--epoch', type = int, default = 300, help = "the number of iteration")
parser.add_argument('--show_interview', type = int, default = 5, help = 'show frequency')

opt = parser.parse_args()

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

random_seed(1)


db = Dataload(opt.dataroot,  "data31.mat", 'data32.mat', True)
sample = DataLoader(db, batch_size = 32, shuffle = True, pin_memory = True)
len_sample = len(sample)

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=2, fix_sigma=None):
    """计算Gram核矩阵
    source: sample_size_1 * feature_size 的数据
    target: sample_size_2 * feature_size 的数据
    kernel_mul: 这个概念不太清楚，感觉也是为了计算每个核的bandwith
    kernel_num: 表示的是多核的数量
    fix_sigma: 表示是否使用固定的标准差
        return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
                        矩阵，表达形式:
                        [   K_ss K_st
                            K_ts K_tt ]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)  # 合并在一起

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)  # 计算高斯核中的|x-y|

    # 计算多核中每个核的bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    # 高斯核的公式，exp(-|x-y|/bandwith)
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
                  bandwidth_temp in bandwidth_list]

    return sum(kernel_val)  # 将多个核合并在一起

def random_sample(mean, var, pro, sample_num, K):
    count = sample_num
    sample = np.random.normal(mean[0], np.sqrt(var[0]), size = (int(sample_num * pro[0]), 1))
    count -= int(sample_num * pro[0])
    for k in range(1, K):
        if k == K - 1:
            sample = np.vstack([sample, np.random.normal(mean[k], np.sqrt(var[k]), size=(count, 1))])
        else:
            sample = np.vstack([sample, np.random.normal(mean[k], np.sqrt(var[k]), size = (int(sample_num * pro[k]), 1))])
            count -= int(sample_num * pro[k])
    return sample

def mmd(source, target, kernel_mul=2.0, kernel_num=2, fix_sigma=None):
    n = int(source.size()[0])
    m = int(target.size()[0])

    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()
    return loss

def adjust_learning_rate(lr, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def SAM(output, HS):
    data1 = torch.sum(output * HS, 1)
    data2 = torch.sqrt(torch.sum(output ** 2, 1)) * torch.sqrt(torch.sum(HS ** 2, 1))
    sam_loss = torch.acos((data1 / data2)).view(-1).mean().type(torch.float32)
    return sam_loss

def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
node = 10
Q_source = Q_net(204, 1024, node).to(device)
Q_target = Q_net(207, 1024, node).to(device)
P_source = P_net(node, 1024, 204).to(device)
P_target = P_net(node, 1024, 207).to(device)
D_source_gauss = D_net_gauss(node, 64).to(device)
D_target_gauss = D_net_gauss(node, 64).to(device)
Param_estimate = weight_estimate(207 + node, 512, 9).to(device)

gen_lr = 0.0001
reg_lr = 0.0001
P_source_decoder = optim.Adam(P_source.parameters(), lr = gen_lr, betas=(0.5, 0.9))
P_target_decoder = optim.Adam(P_target.parameters(), lr = gen_lr, betas=(0.5, 0.9))
Q_source_encoder = optim.Adam(Q_source.parameters(), lr = gen_lr, betas=(0.5, 0.9))
Q_target_encoder = optim.Adam(Q_target.parameters(), lr = gen_lr, betas=(0.5, 0.9))
Q_source_generator = optim.Adam(Q_source.parameters(), lr = reg_lr, betas=(0.5, 0.9))
Q_target_generator = optim.Adam(Q_target.parameters(), lr = reg_lr, betas=(0.5, 0.9))
Param = optim.Adam(Param_estimate.parameters(), lr = reg_lr, betas=(0.5, 0.9))
D_source_gauss_solver = optim.Adam(D_source_gauss.parameters(), lr = reg_lr, betas=(0.5, 0.9))
D_target_gauss_solver = optim.Adam(D_target_gauss.parameters(), lr = reg_lr, betas=(0.5, 0.9))
Q_alignment = optim.Adam(Q_target.parameters(), lr = reg_lr, betas=(0.5, 0.9))

best_loss = 10
criteon = nn.L1Loss()


for epoch in range(opt.epoch):
    TINY = 1e-15
    Q_source.train()
    Q_target.train()
    P_source.train()
    P_target.train()
    D_source_gauss.train()
    D_target_gauss.train()
    Param_estimate.train()

    if (epoch + 1) % 150 == 0:
        gen_lr = gen_lr / 10
        reg_lr = reg_lr / 10
        adjust_learning_rate(gen_lr, Q_source_encoder)
        adjust_learning_rate(gen_lr, Q_target_encoder)
        adjust_learning_rate(gen_lr, P_source_decoder)
        adjust_learning_rate(gen_lr, P_target_decoder)
        adjust_learning_rate(reg_lr, Q_source_generator)
        adjust_learning_rate(reg_lr, Q_target_generator)
        adjust_learning_rate(reg_lr, Param)
        adjust_learning_rate(reg_lr, D_source_gauss_solver)
        adjust_learning_rate(reg_lr, D_target_gauss_solver)
        adjust_learning_rate(reg_lr, Q_alignment)

    for step, (source, target) in enumerate(sample):
        source = source.type(torch.float32).to(device)
        target = target.type(torch.float32).to(device)

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()

        source_sample = Q_source(source)
        source_recon_sample = P_source(source_sample)
        recon_loss = criteon(source_recon_sample + TINY, source + TINY) + 0.1 * SAM(source_recon_sample + TINY, source + TINY)

        recon_loss.backward()
        P_source_decoder.step()
        Q_source_encoder.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()

        if step % 20 == 0:
            Q_source.eval()
            #z_real_gauss = (torch.randn(source.shape[0], 5) * 1).to(device)
            real = random_sample(mean = [0.08986611, 0.25775553, 0.23853643, 0.23515718, 0.42317983, 0.23096549,
                                       0.52180721, 0.28268092, 0.14412658],
                                 var = [0.00076042, 0.00160875, 0.00242589, 0.00271563, 0.00526027, 0.00207626,
                                      0.00435866, 0.00281696, 0.00127437],
                                 pro = [0.12603793, 0.10324363, 0.12649997, 0.11775449, 0.09746581, 0.12663226,
                                      0.06090274, 0.11372481, 0.12773836],
                                 sample_num = int(source.shape[0] * node),
                                 K = 9)

            real = np.reshape(real, [source.shape[0], node])
            z_real_gauss = torch.from_numpy(real).type(torch.float32).to(device)
            z_fake_gauss = Q_source(source)
            D_real_gauss = D_source_gauss(z_real_gauss).mean()
            D_fake_gauss = D_source_gauss(z_fake_gauss.detach()).mean()
            gradient_penalty_P = compute_gradient_penalty(D_source_gauss, z_real_gauss, z_fake_gauss)
            D_gauss_loss = D_fake_gauss - D_real_gauss + 10 * gradient_penalty_P

            D_gauss_loss.backward()
            D_source_gauss_solver.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()


        Q_source.train()
        z_fake_gauss = Q_source(source)

        D_fake_gauss = -1 * D_source_gauss(z_fake_gauss).mean()
        G_gauss_loss = D_fake_gauss
        #G_gauss_loss = -torch.mean(torch.log(D_fake_gauss + TINY))

        G_gauss_loss.backward()
        Q_source_generator.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()

        Q_source.eval()
        Q_target.train()
        source_sample = Q_source(source)
        target_sample = Q_target(target)
        loss = 0.5 * mmd(source_sample, target_sample)
        loss.backward()
        Q_alignment.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()

        Q_target.train()
        P_target.train()
        Param_estimate.train()
        z = Q_target(target)
        target_recon_sample = P_target(z)
        recon_loss = criteon(target_recon_sample + TINY, target + TINY) + 0.1 * SAM(target_recon_sample + TINY, target + TINY)
        recon_loss.backward()
        P_target_decoder.step()
        Q_target_encoder.step()
        Param.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()

        if step % 20 == 0:
            Q_target.eval()
            P_target.eval()
            Param_estimate.eval()

            latent_feature = Q_target(target)
            error = target - P_target(latent_feature)

            weight = Param_estimate(torch.cat([latent_feature, error], dim = 1))
            weight = torch.mean(weight, dim = 0).detach().cpu().numpy()
            real = random_sample(mean = [0.08986611, 0.25775553, 0.23853643, 0.23515718, 0.42317983, 0.23096549,
                                       0.52180721, 0.28268092, 0.14412658],
                                 var = [0.00076042, 0.00160875, 0.00242589, 0.00271563, 0.00526027, 0.00207626,
                                      0.00435866, 0.00281696, 0.00127437],
                                 pro = weight.tolist(),
                                 sample_num = int(target.shape[0] * node),
                                 K = 9)

            real = np.reshape(real, [target.shape[0], node])
            z_real_gauss = torch.from_numpy(real).type(torch.float32).to(device)
            z_fake_gauss = Q_target(target)
            D_real_gauss = D_target_gauss(z_real_gauss).mean()
            D_fake_gauss = D_target_gauss(z_fake_gauss.detach()).mean()
            gradient_penalty_P = compute_gradient_penalty(D_target_gauss, z_real_gauss, z_fake_gauss)
            D_gauss_loss = D_fake_gauss - D_real_gauss + 10 * gradient_penalty_P

            D_gauss_loss.backward()
            D_target_gauss_solver.step()

        P_source.zero_grad()
        P_target.zero_grad()
        Q_source.zero_grad()
        Q_target.zero_grad()
        D_source_gauss.zero_grad()
        D_target_gauss.zero_grad()
        Param.zero_grad()
        Q_alignment.zero_grad()


        Q_target.train()
        z_fake_gauss = Q_target(target)

        D_fake_gauss = -1 * D_target_gauss(z_fake_gauss).mean()
        G_gauss_loss = D_fake_gauss

        G_gauss_loss.backward()
        Q_target_generator.step()
        #
        #
        print('Epoch-{}; D_loss_gauss: {:.4}; G_gauss: {:.4}; recon_loss: {:.4}'.format(epoch,
                                                                                       D_gauss_loss.item(),
                                                                                       G_gauss_loss.item(),
                                                                                       recon_loss.item()))

    torch.save(Q_source.state_dict(), "{0}/encoder_source_epoch_{1}.pth".format("model", epoch + 1))
    torch.save(Q_target.state_dict(), "{0}/encoder_target_epoch_{1}.pth".format("model", epoch + 1))
    torch.save(P_source.state_dict(), "{0}/decoder_source_epoch_{1}.pth".format("model", epoch + 1))
    torch.save(P_target.state_dict(), "{0}/decoder_target_epoch_{1}.pth".format("model", epoch + 1))




