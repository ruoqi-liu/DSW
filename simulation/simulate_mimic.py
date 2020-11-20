import pandas as pd
import numpy as np
from tqdm import tqdm
import os

treatment_option = 'vaso'
observation_window = 30
step = 3

# icustay ids for sepsis patient in MIMIC-III
treatment_ids = pd.read_csv('../data/icustay_ids.txt')

# hadm ids for sepsis patient in MIMIC-III
icu2hadm = pd.read_json('../data/icu_hadm_dict.json', typ='series').to_dict()
hadm2icu = {icu2hadm[icu]:icu for icu in icu2hadm.keys()}


# Extract A, X, X_static from the imputed  data
# Missing values of X are imputed via filling the mean values
# X: Time-varying covariates
# X format：T * N (T: total time in ICU with 3hr as interval; N: number of features ["var" below])
# We use first 30hr (10 timestamps * 3 hr interval) data since ICU admission

# A: Treatment assignments
# A format: 10 * 1

# X_static: patients' demographics
# X_static format: n_patient * n_static_covariate

data_dir = "../data"
data_synthetic = "../data_synthetic"

var = ['hemoglobin','C-reactive protein','heartrate','creatinine',
 'hematocrit','sysbp','tempc','pt','sodium','diasbp', 'gcs_min','platelet','ptt',
 'chloride','resprate','glucose','bicarbonate','bands', 'bun',
 'Magnesium','urineoutput','inr','lactate','aniongap','spo2','wbc','meanbp']

n_classes = 1
n_X_features = len(var)
n_X_static_features = 12

X = []
used_ID = []
for ii, ID in tqdm(enumerate(hadm2icu.keys())):
    tmp = pd.read_csv(data_dir+'x/{}.csv'.format(ID))
    tmp = tmp[(tmp['time'] > 0) & (tmp['time'] < 33)]
    if len(tmp) == 11:
        icu = hadm2icu[ID]
        used_ID.append(icu)
        X.append(tmp[var][:-1].to_numpy())

X_new= np.zeros(shape=(len(used_ID), observation_window//3, n_X_features))

for i in range(len(used_ID)):
    X_new[i] = X[i]

X_static = np.random.normal(0, 0.5, size=(len(used_ID),n_X_static_features))
for j, ID in enumerate(used_ID):
    if os.path.exists(data_dir + 'static/%d.static.npy' % ID):
        X_static[j] = np.load(data_dir + 'static/%d.static.npy' % ID)
    else:
        print(ID)
X_static=np.nan_to_num(X_static)


A = np.zeros(shape=(len(used_ID), observation_window//step))
for ii, ID in tqdm(enumerate(used_ID)):
    if os.path.exists('{}/treatment/{}.npy'.format(data_dir, ID)):
        A[ii, :] = np.load('{}/treatment/{}.npy'.format(data_dir, ID))
    else:
        A[ii, :] = np.zeros(observation_window//step)

X_mean = np.mean(X_new, axis=(0,1))
X_std = np.std(X_new, axis=(0,1))

X_static_mean = np.mean(X_static, axis=0)
X_static_std = np.std(X_static, axis=0)

X_norm = np.zeros(shape=(len(used_ID),observation_window//step,n_X_features))
X_static_norm = np.zeros(shape=(len(used_ID),n_X_static_features))


for i in range(observation_window//step):
    for j in range(n_X_features):
        X_norm[:,i,j] = (X_new[:,i,j]-X_mean[j])/X_std[j]

for c in range(n_X_static_features):
    if c in (0, 8, 9, 10):
        X_static_norm[:, c] = (X_static[:,c]-X_static_mean[c])/X_static_std[c]
    else:
        X_static_norm[:, c] = X_static[:,c]

N_treated = len(np.where(np.sum(A, axis=1)>0)[0])

A_final = np.where(np.sum(A, axis=1)>0, 1, 0)

all_idx = np.arange(len(used_ID))
np.random.shuffle(all_idx)

train_ratio = 0.7
val_ratio = 0.1

train_idx = all_idx[:int(len(all_idx)*train_ratio)]
val_idx = all_idx[int(len(all_idx) * train_ratio):int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio)]
test_idx = all_idx[int(len(all_idx) * train_ratio)+int(len(all_idx) * val_ratio):]

train_ids = [used_ID[x] for x in train_idx]
val_ids = [used_ID[x] for x in val_idx]


split = []
for x in used_ID:
    if x in train_ids:
        split.append([x, 1])
    elif x in val_ids:
        split.append([x, 2])
    else:
        split.append([x, 0])


# num of P
p = 5
# weight of hidden confounders
gamma_h = 0.1
# num of hidden
h = 1
N_treated = len(np.where(np.sum(A, axis=1)>0)[0])
N = len(used_ID)

eta, epsilon = np.random.normal(0,0.001, size=(N,observation_window//step,n_X_features)),np.random.normal(0,0.001, size=(N,observation_window//step,h))
delta = np.random.uniform(-1, 1, size=(n_X_features+ n_X_static_features, h))

A_final = np.where(np.sum(A, axis=1)>0, 1, 0)
Z = np.random.normal(0, 0.5, size=(N,h))
Z[np.where(np.sum(A, axis=1)>0), :] = np.random.normal(1, 0.5, size=(N_treated, h))
Z_all = [Z]
Q_all = []
for t in range(1, observation_window//step+1):
    i = 1
    tmp_x = 0
    tmp_z = 0
    while (t-i) >= 0 and i <= p:

        mu = np.random.normal(1 - (i / p), (1 / p), size=(N, h))
        v = np.random.normal(0, 0.02, size=(N, h))
        v[np.where(np.sum(A, axis=1) > 0), :] = np.random.normal(1, 0.02, size=(N_treated, h))
        tmp_z += np.multiply(mu, Z_all[t - i]) + np.multiply(v, np.tile(A[:, t - i], (h, 1)).T)

        i += 1

    X_sample = X_norm[:,t-1,:]
    Z = tmp_z/(i-1) + epsilon[:,t-1,:]

    Z_all.append(Z)

    Q = gamma_h * Z + (1 - gamma_h) * np.expand_dims(np.mean(np.concatenate((X_sample, X_static_norm), axis=1), axis=1), axis=1)

    Q_all.append(Q)

w = np.random.uniform(-1, 1, size=(2, 1))
b = np.random.normal(0, 0.1, size=(N, 1))
Y_f = np.matmul(np.concatenate((Q, np.expand_dims(A_final, axis=1)),axis=1), w) + b

w = np.random.uniform(-1, 1, size=(2, 1))
b = np.random.normal(0, 0.1, size=(N, 1))
A_final_cf = np.where(A_final==1, 0, 1)
Y_cf = np.matmul(np.concatenate((Q, np.expand_dims(A_final_cf, axis=1)),axis=1), w) + b

Y_f_norm = Y_f
Y_cf_norm = Y_cf

dir = '{}/data_mimic_mean_syn_{}'.format(data_synthetic, gamma_h)
dir_base = '{}/data_baseline_mimic_mean_syn_{}'.format(data_synthetic,gamma_h)

os.makedirs(dir, exist_ok=True)
os.makedirs(dir_base, exist_ok=True)

for n in tqdm(range(len(used_ID))):
    x = np.zeros(shape=(observation_window//step, n_X_features))
    ID = used_ID[n]
    out_x_file = '{}/{}.x.npy'.format(dir, ID)
    out_static_file = '{}/{}.static.npy'.format(dir,ID)
    out_a_file = '{}/{}.a.npy'.format(dir,ID)
    out_y_file = '{}/{}.y.npy'.format(dir,ID)
    for t in range(observation_window//step):
        x[t, :] = X_norm[n,t,:]
    x_static = X_static_norm[n,:]
    a = A[n,:]

    y = [Y_f_norm[n], Y_cf_norm[n]]

    np.save(out_x_file, x)
    np.save(out_static_file, x_static)
    np.save(out_a_file, a)
    np.save(out_y_file, y)


for t in tqdm(range(observation_window//step)):
    # a + y_f + y_cf + n_covariates + split

    out_matrix = np.zeros((len(used_ID), n_X_features+n_X_static_features+1+n_classes*2+1))

    out_matrix[:,0] = A_final
    out_matrix[:,(1+n_classes*2):(1+n_classes*2+n_X_features)] = X_norm[:,t,:]
    out_matrix[:,(1+n_classes*2+n_X_features):(1+n_classes*2+n_X_features+n_X_static_features)] = X_static_norm

    out_matrix[:, 1:2] = Y_f_norm
    out_matrix[:, 2:3] = Y_cf_norm

    out_matrix[:,-1] = np.array(split)[:,-1]

    df = pd.DataFrame(out_matrix)
    df.to_csv('{}/{}.csv'.format(dir_base,t+1), index=False)


df = pd.DataFrame(np.array(split))

df.to_csv('{}/train_test_split.csv'.format(dir), index=False, header=False)

