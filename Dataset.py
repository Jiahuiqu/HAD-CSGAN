import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io import loadmat
from sklearn.mixture import GaussianMixture
from scipy.io import loadmat, savemat

class Dataload(Dataset):
    def __init__(self, data_path, source, target, mode):
        super(Dataset, self).__init__()
        self.mode = mode
        self.source = loadmat(os.path.join(data_path, source))['data']
        self.label = loadmat(os.path.join(data_path, source))['map']
        self.target = loadmat(os.path.join(data_path, target))['data']
        self.source = np.reshape(self.source, [-1, 204])
        self.label = np.reshape(self.label, [-1])
        self.index = np.where(self.label == 0)[0] if self.mode else range(self.source.shape[0])
        self.target = np.reshape(self.target, [-1, 207])
        self.target_index = np.random.choice(np.arange(0, 10000), len(self.index), replace = False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        if self.mode == True:
            return self.source[self.index[item]], self.target[self.target_index[item]]
        else:
            return self.source[self.index[item]], self.target[self.index[item]]


def calculate_GMM_coe():
    ### 数据归一化
    image = loadmat(r"C:\Users\HSX\Desktop\TCSVT\abu-urban-1_norm.mat")['data']
    label = loadmat(r"C:\Users\HSX\Desktop\TCSVT\abu-urban-1_norm.mat")['map']
    sample = image[0, 0, :]
    ### Find the source domain image background
    for i in range(100):
        for j in range(100):
            if label[i, j] == 0:
                sample = np.vstack([sample, image[i, j, :]])
    sample = np.reshape(sample[1:, :], [-1])
    max = np.max(sample)
    min = np.min(sample)
    data = (sample - min) / (max - min)
    ### calculate the GMM
    gmm = GaussianMixture(n_components=9, tol=1e-5, max_iter=500)
    gmm.fit(data)
    # print(gmm.means_)
    # print(gmm.covariances_)
    # print(gmm.weights_)

    weight = np.zeros((9, 1))
    means = np.zeros((9, 1))
    var = np.zeros((9, 1))

    for i in range(1):
        weight[:, i] = gmm.weights_
        means[:, i] = np.reshape(gmm.means_, [-1])
        var[:, i] = np.reshape(gmm.covariances_, [-1])

    weight = np.mean(weight, axis=1)
    means = np.mean(means, axis=1)
    var = np.mean(var, axis=1)
    return weight, means, var


# if __name__ == "__main__":
#
#     # dataload = Dataload("data", "data31", "data32", False)
#     # sample = DataLoader(dataload, batch_size = 32, shuffle = True)
#     # for step, (source, target) in enumerate(sample):
#     #     print(step, source.shape, target.shape)

