# 1. simulate treatment A
# a. 33% treated, 66% control.
# b. for 33% treated, simulate the initial point among T time stamps
# 2. simulate covariates X and hidden confounders Z using A
# 3. simulate outcome Y

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# timestamp
T = 30
# num of covariates
k = 100
# num of static features
k_s = 5
# num of hidden
h = 1
# num of samples
N = 4000
N_treated = 1000
# seed
np.random.seed(66)
# num of P
p = 5
# weight of hidden confounders
gamma_h = 0.1
# bias
kappa = 10

S = 20

data_synthetic = "../data_synthetic"

eta, epsilon = np.random.normal(0,0.001, size=(N,T,k)),np.random.normal(0,0.001, size=(N,T,h))
w = np.random.uniform(-1, 1, size=(h+1, 2))
b = np.random.normal(0, 0.1, size=(N, 2))


# 1. simulate treatment A
A = np.zeros(shape=(N, T))
for n in range(N_treated):
    initial_point = np.random.choice(range(T))
    a = np.zeros(T)
    a[initial_point:] = 1
    A[n] = a

np.random.shuffle(A)


# 2. simulate covariates X and hidden confounders Z using A
# 3. simulate outcome Y
X = np.random.normal(0, 0.5, size=(N,k))
X[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, k))
X_static = np.random.normal(0, 0.5, size=(N,k_s))
X_static[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, k_s))
Z = np.random.normal(0, 0.5, size=(N,h))
Z[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, h))
delta = np.random.uniform(-1, 1, size=(k + k_s, h))

A_final = np.where(np.sum(A, axis=1)>0, 1, 0)

X_all, Z_all = [X], [Z]
for t in range(1, T+1):
    i = 1
    tmp_x = 0
    tmp_z = 0
    while (t-i) >= 0 and i <= p:
        alpha = np.random.normal(1 - (i / p), (1 / p), size=(N, k))
        beta = np.random.normal(0, 0.02, size=(N, k))
        beta[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.02, size=(N_treated, k))
        tmp_x += np.multiply(alpha, X_all[t - i]) + np.multiply(beta, np.tile(A[:, t - i], (k, 1)).T)

        mu = np.random.normal(1 - (i / p), (1 / p), size=(N, h))
        v = np.random.normal(0, 0.02, size=(N, h))
        v[np.where(np.sum(A, axis=1) > 0), :] = np.random.normal(1, 0.02, size=(N_treated, h))
        tmp_z += np.multiply(mu, Z_all[t - i]) + np.multiply(v, np.tile(A[:, t - i], (h, 1)).T)
        i += 1

    X = tmp_x/(i-1) + eta[:,t-1,:]
    Z = tmp_z/(i-1) + epsilon[:,t-1,:]

    X_all.append(X)
    Z_all.append(Z)

    Q = gamma_h * Z + (1-gamma_h) * np.expand_dims(np.mean(np.concatenate((X, X_static), axis=1), axis=1), axis=1)

w = np.random.uniform(-1, 1, size=(1, 2))
b = np.random.normal(0, 0.1, size=(N, 2))
Y = np.matmul(Q, w) + b
Y_f = A_final * Y[:,0] + (1-A_final) * Y[:,1]
Y_cf = A_final * Y[:,1] + (1-A_final) * Y[:,0]

# w = np.random.uniform(-1, 1, size=(2, 1))
# b = np.random.normal(0, 0.1, size=(N, 1))
# Y_f = np.matmul(np.concatenate((Q, np.expand_dims(A_final, axis=1)),axis=1), w) + b
# # Y_f = (Y_f-np.mean(Y_f))/np.std(Y_f)
#
# w = np.random.uniform(-1, 1, size=(2, 1))
# b = np.random.normal(0, 0.1, size=(N, 1))
# A_final_cf = np.where(A_final==1, 0, 1)
# Y_cf = np.matmul(np.concatenate((Q, np.expand_dims(A_final_cf, axis=1)),axis=1), w) + b
# Y_cf = (Y_cf-np.mean(Y_cf))/np.std(Y_cf)


dir = '{}/data_syn_{}'.format(data_synthetic, gamma_h)
dir_base = '{}/data_baseline_syn_{}'.format(data_synthetic,gamma_h)

os.makedirs(dir, exist_ok=True)
os.makedirs(dir_base, exist_ok=True)

for n in tqdm(range(N)):
    x = np.zeros(shape=(T, k))
    out_x_file = '{}/{}.x.npy'.format(dir, n)
    out_static_file = '{}/{}.static.npy'.format(dir,n)
    out_a_file = '{}/{}.a.npy'.format(dir,n)
    out_y_file = '{}/{}.y.npy'.format(dir,n)
    for t in range(1, T+1):
        x[t-1, :] = X_all[t][n,:]
    x_static = X_static[n,:]
    a = A[n,:]

    y = [Y_f[n], Y_cf[n]]

    np.save(out_x_file, x)
    np.save(out_static_file, x_static)
    np.save(out_a_file, a)
    np.save(out_y_file, y)


all_idx = np.arange(N)
np.random.shuffle(all_idx)

train_ratio = 0.7
val_ratio = 0.1

train_idx = all_idx[:int(len(all_idx)*train_ratio)]
val_idx = all_idx[int(len(all_idx) * train_ratio):int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio)]
test_idx = all_idx[int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio):]

split = np.ones(N)
split[test_idx] = 0
split[val_idx] = 2

df = pd.DataFrame(split, dtype=int)
df.to_csv('{}/train_test_split.csv'.format(dir), index=False, header=False)


for t in tqdm(range(1, T+1)):
    # a + y_f + y_cf + n_covariates + split

    out_matrix = np.zeros((N, k+k_s+1+2+1))

    out_matrix[:,0] = A_final
    out_matrix[:,3:3+k] = X_all[t]
    out_matrix[:,3+k:3+k+k_s] = X_static

    out_matrix[:,1] = Y_f
    out_matrix[:,2] = Y_cf

    out_matrix[:,-1] = split

    df = pd.DataFrame(out_matrix)
    df.to_csv('{}/{}.csv'.format(dir_base,t), index=False)
