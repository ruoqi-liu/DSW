import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from tqdm import tqdm

import os
from sklearn.metrics import mean_squared_error

import data_loader_syn
from model_synthetic import LSTMModel

HIDDEN_SIZE = 32
CUDA = False


def trainInitIPTW(train_loader, val_loader,test_loader, model, epochs, optimizer, criterion,
                  l1_reg_coef=None, use_cuda=False, save_model=None):

    if use_cuda:
        print("====> Using CUDA device: ", torch.cuda.current_device(), flush=True)
        model.cuda()
        model = model.to('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Train network

    best_pehe_val = np.float('inf')
    best_loss_val = np.float('inf')
    best_pehe_test = np.float('inf')
    best_ate_test = np.float('inf')
    best_mse_test = np.float('inf')
    for epoch in range(epochs):
        ipw_epoch_losses = []
        outcome_epoch_losses = []
        f_train_outcomes = []
        f_train_treatments = []

        for x_inputs, x_static_inputs, x_fr_inputs, targets in tqdm(train_loader):
            model.train()

            # train IPW
            optimizer.zero_grad()

            fr_targets = x_fr_inputs
            if use_cuda:
                x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                targets, fr_targets = targets.cuda(), fr_targets.cuda()

            ipw_outputs, f_outcome_out, cf_outcome_out, _ = model(x_inputs, x_static_inputs, fr_targets)
            f_treatment = torch.where(fr_targets.sum(1) > 0, torch.Tensor([1]), torch.Tensor([0]))

            f_train_outcomes.append(targets[:,0])
            f_train_treatments.append(f_treatment)

            ipw_loss = 0

            for i in range(len(ipw_outputs)):
                ipw_pred_norm = ipw_outputs[i].squeeze(1)
                ipw_loss += criterion(ipw_pred_norm, fr_targets[:, i].float())

            ipw_loss = ipw_loss/len(ipw_outputs)

            weights = torch.zeros(len(ipw_outputs[-1]))
            treat_sum = torch.sum(fr_targets, axis=1)
            p_treated = torch.where(treat_sum == 0)[0].size(0) / treat_sum.size(0)

            ipw_outputs = torch.cat(ipw_outputs, dim=1)
            ps = torch.sigmoid(ipw_outputs)

            for i in range(len(ps)):
                for t in range(ipw_outputs.size(1)):
                    if treat_sum[i] != 0:
                        weights[i] += p_treated / ps[i, t]
                    else:
                        weights[i] += (1 - p_treated) / (1 - ps[i, t])

            weights = weights / ipw_outputs.size(1)

            weights = torch.where(weights >= 100, torch.Tensor([100]), weights)
            weights = torch.where(weights <= 0.01, torch.Tensor([0.01]), weights)

            outcome_loss = torch.mean(weights*(f_outcome_out - targets[:,0]) ** 2)

            loss = ipw_loss * 0.05 + outcome_loss

            if l1_reg_coef:
                l1_regularization = torch.zeros(1)
                for pname, param in model.hidden2hidden_ipw.named_parameters():
                    if 'weight' in pname:
                        l1_regularization += torch.norm(param, 1)
                for pname, param in model.hidden2out_outcome_f.named_parameters():
                    if 'weight' in pname:
                        l1_regularization += torch.norm(param, 1)
                loss += (l1_reg_coef * l1_regularization).squeeze()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 20)

            optimizer.step()
            ipw_epoch_losses.append(ipw_loss.item())
            outcome_epoch_losses.append(loss.item())


        epoch_losses_ipw = np.mean(ipw_epoch_losses)
        outcome_epoch_losses = np.mean(outcome_epoch_losses)


        print('Epoch: {}, IPW train loss: {}'.format(epoch, epoch_losses_ipw), flush=True)
        print('Epoch: {}, Outcome train loss: {}'.format(epoch, outcome_epoch_losses), flush=True)


        # validation
        print('Validation:')

        pehe_val, _, mse_val, loss_val = model_eval(model, val_loader, criterion, eval_use_cuda=use_cuda)

        # if pehe_val < best_pehe_val:
        #     best_pehe_val = pehe_val

        if loss_val < best_loss_val:
            best_loss_val = loss_val

            if save_model:
                print('Best model. Saving...\n')
                torch.save(model, save_model)
                
                print('Test:')
                pehe_test,ate_test,mse_test,_ = model_eval(model, test_loader,criterion, eval_use_cuda=use_cuda)
                best_pehe_test = pehe_test
                best_ate_test = ate_test
                best_mse_test = mse_test

    print(np.sqrt(best_pehe_test))
    print(best_ate_test)
    print(np.sqrt(best_mse_test))
    return best_pehe_test


def transfer_data(model, dataloader, criterion, eval_use_cuda=False):
    with torch.no_grad():
        model.eval()
        f_outcome_outputs = []
        cf_outcome_outputs = []
        f_outcome_true = []
        cf_outcome_true = []
        ipw_true = []
        loss_all = []

        for x_inputs, x_static_inputs, x_fr_inputs, targets in dataloader:
            fr_targets = x_fr_inputs
            if eval_use_cuda:
                x_inputs, x_static_inputs, x_fr_inputs = x_inputs.cuda(), x_static_inputs.cuda(), x_fr_inputs.cuda()
                targets, fr_targets = targets.cuda(), fr_targets.cuda()


            ipw_outputs, f_outcome_out, cf_outcome_out, _ = model(x_inputs, x_static_inputs, fr_targets)

            ipw_loss = 0
            for i in range(len(ipw_outputs)):
                ipw_pred_norm = ipw_outputs[i].squeeze(1)
                ipw_loss += criterion(ipw_pred_norm, fr_targets[:, i].float())


            outcome_loss = torch.mean((f_outcome_out - targets[:,0]) ** 2)

            loss = ipw_loss * 0.05 + outcome_loss


            if eval_use_cuda:
                for i in range(len(ipw_outputs)):
                    ipw_outputs[i]=ipw_outputs[i].to('cpu').detach().data.numpy()
                fr_targets = fr_targets.to('cpu').detach().data.numpy()
                targets = targets.to('cpu').detach().data.numpy()
                f_outcome_out = f_outcome_out.to('cpu').detach().data.numpy()
                cf_outcome_out = cf_outcome_out.to('cpu').detach().data.numpy()
                loss = loss.to('cpu').detach().data.numpy()
            else:
                for i in range(len(ipw_outputs)):
                    ipw_outputs[i]=ipw_outputs[i].detach().data.numpy()
                ipw_outputs = np.array(ipw_outputs)
                x_fr_inputs = x_fr_inputs.detach().data.numpy()
                targets = targets.detach().data.numpy()
                # outcome_outputs = outcome_outputs.detach().data.numpy()

            ipw_true.append(np.where(fr_targets.sum(1) > 0, 1, 0))
            f_outcome_true.append(targets[:,0])
            cf_outcome_true.append(targets[:, 1])
            f_outcome_outputs.append(f_outcome_out)
            cf_outcome_outputs.append(cf_outcome_out)
            loss_all.append(loss)


        ipw_true = np.concatenate(ipw_true).transpose()
        f_outcome_true = np.concatenate(f_outcome_true)
        cf_outcome_true = np.concatenate(cf_outcome_true)
        f_outcome_outputs = np.concatenate(f_outcome_outputs)
        cf_outcome_outputs = np.concatenate(cf_outcome_outputs)
        # loss_all = np.concatenate(loss_all)
        loss_all = np.mean(loss_all)

        return ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs, loss_all


def compute_pehe_ate(t, y_f, y_cf, y_pred_f, y_pred_cf):

    y_treated_true = t * y_f + (1-t) * y_cf
    y_control_true = t * y_cf + (1 - t) * y_f

    y_treated_pred = t * y_pred_f + (1 - t) * y_pred_cf
    y_control_pred = t * y_pred_cf + (1 - t) * y_pred_f

    pehe = np.mean(np.square((y_treated_pred-y_control_pred)-(y_treated_true-y_control_true)))
    ate = np.mean(np.abs((y_treated_pred - y_control_pred) - (y_treated_true - y_control_true)))

    return pehe,ate



def model_eval(model, dataloader, criterion ,eval_use_cuda=False):

    ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs,loss_all = transfer_data(model, dataloader, criterion, eval_use_cuda)

    pehe,ate = compute_pehe_ate(ipw_true, f_outcome_true, cf_outcome_true, f_outcome_outputs, cf_outcome_outputs)

    mse = mean_squared_error(f_outcome_true,f_outcome_outputs)

    print('PEHE: {:.4f}\tATE: {:.4f}\nRMSE: {:.4f}\n'.format(np.sqrt(pehe),ate, np.sqrt(mse)))

    return pehe, ate, mse, loss_all


# MAIN
if __name__ == '__main__':
    
    # ---------------------------------------- # 
    # Parse input arguments

    treatment_option = 'vaso'
    gamma = '0.1'

    parser = argparse.ArgumentParser(description='Synthetic Dataset')

    parser.add_argument('--observation_window', type=int, default=12, required=True,
                        metavar='OW', help='observation window')

    parser.add_argument('--epochs', type=int, default=30, required=True,
                        metavar='EPOC', help='train epochs')
    
    parser.add_argument('--batch-size', type=int, default=128, 
                        metavar='BS',help='batch size')

    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-5)

    parser.add_argument('--l1', '--l1-reg-coef', default=1e-5, type=float,
                        metavar='L1', help='L1 reg coef')

    parser.add_argument('--resume', default=''.format(treatment_option), type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--save_model', default='checkpoints/mimic-6-7-{}.pt'.format(gamma), type=str, metavar='PATH',
                        help='path to save new checkpoint (default: none)')

    parser.add_argument('--cuda-device', default=1, type=int, metavar='N',
                        help='which GPU to use')

    parser.add_argument('--split_file', default='data_synthetic/data_syn_{}/train_test_split.csv'.format(gamma), type=str, metavar='PATH',
                        )

    args = parser.parse_args()

    # Settings
    if CUDA:
        torch.cuda.set_device(args.cuda_device)

    print("batch size ==> ", args.batch_size )
    print("lr ==> ", args.lr)
    print("observation window == >", args.observation_window)
    torch.manual_seed(666)

    train_test_split = np.loadtxt(args.split_file, delimiter=',', dtype=int)

    train_iids = np.where(train_test_split==1)[0]
    val_iids = np.where(train_test_split == 2)[0]
    test_iids = np.where(train_test_split == 0)[0]

    # Datasets
    train_dataset = data_loader_syn.SyntheticDataset(train_iids, args.observation_window, treatment_option)
    val_dataset = data_loader_syn.SyntheticDataset(val_iids, args.observation_window, treatment_option)
    test_dataset = data_loader_syn.SyntheticDataset(test_iids, args.observation_window, treatment_option)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, args.batch_size, shuffle=True)
    
    n_X_features, n_X_static_features, n_X_fr_types, n_classes = data_loader_syn.get_dim()
    # ---------------------------------------- # 
    # Model 
    
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))

            model = torch.load(args.resume)
            model=model.cuda()

            print("=> loaded checkpoint '{}'"
                  .format(args.resume))

        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:

        attn_model = 'concat2'
        n_Z_confounders = HIDDEN_SIZE

        model = LSTMModel(n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                 attn_model, n_classes, args.observation_window,
                          args.batch_size, hidden_size = HIDDEN_SIZE)

    adam_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    trainInitIPTW(train_loader, val_loader,test_loader,
                  model, epochs= args.epochs,
                  criterion=F.binary_cross_entropy_with_logits, optimizer=adam_optimizer,
                  l1_reg_coef=args.l1,
                  use_cuda=CUDA,
                  save_model=args.save_model)
