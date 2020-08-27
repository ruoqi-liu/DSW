import numpy as np
import torch
from torch.utils import data

gamma=0.1

data_dir = 'simulation/data_mymodel_new2_{}/'.format(gamma)


# dataset meta data
n_X_features = 100
n_X_static_features = 5
n_X_t_types = 1
n_classes = 1


def get_dim():
    return n_X_features, n_X_static_features, n_X_t_types, n_classes



class SyntheticDataset(data.Dataset):
    def __init__(self, list_IDs, obs_w, treatment):
        '''Initialization'''
        self.list_IDs = list_IDs
        self.obs_w = obs_w
        self.treatment = treatment


    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        ID = self.list_IDs[index]

        # Load labels
        label = np.load(data_dir + '{}.y.npy'.format(ID))

        # Load data
        X_demographic = np.load(data_dir + '{}.static.npy'.format(ID))
        X_all = np.load(data_dir + '{}.x.npy'.format(ID))
        X_treatment_res = np.load(data_dir + '{}.a.npy'.format(ID))

        X = torch.from_numpy(X_all.astype(np.float32))
        X_demo = torch.from_numpy(X_demographic.astype(np.float32))
        X_treatment = torch.from_numpy(X_treatment_res.astype(np.float32))
        y = torch.from_numpy(label.astype(np.float32))

        return X, X_demo, X_treatment, y

