import torch.nn as nn
import torch
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','concat2']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'concat2':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def concat_score2(self, hidden, encoder_output):
        h = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        h = torch.cat((h, hidden*encoder_output),2)
        energy = self.attn(h).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'concat2':
            attn_energies = self.concat_score2(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LSTMModel(nn.Module):
    def __init__(self, n_X_features, n_X_static_features, n_X_fr_types, n_Z_confounders,
                 attn_model, n_classes, obs_w,
                 batch_size, hidden_size,
                 num_layers=2, bidirectional=True, dropout = 0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_X_features = n_X_features
        self.n_X_static_features = n_X_static_features
        self.n_classes = n_classes
        self.obs_w = obs_w
        self.num_layers = num_layers
        self.x_emb_size = 32
        self.x_static_emb_size = 16
        self.z_dim = n_Z_confounders

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.n_t_classes = 1

        self.rnn_f = nn.GRUCell(input_size=self.x_emb_size + 1 + n_Z_confounders, hidden_size=hidden_size)
        self.rnn_cf = nn.GRUCell(input_size=self.x_emb_size + 1 + n_Z_confounders, hidden_size=hidden_size)

        self.attn_f = Attn(attn_model, hidden_size)
        self.concat_f = nn.Linear(hidden_size * 2, hidden_size)

        self.attn_cf = Attn(attn_model, hidden_size)
        self.concat_cf = nn.Linear(hidden_size * 2, hidden_size)



        self.x2emb = nn.Linear(n_X_features, self.x_emb_size)
        self.x_static2emb = nn.Linear(n_X_static_features, self.x_static_emb_size)

        # IPW
        self.hidden2hidden_ipw = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_emb_size + hidden_size + self.x_static_emb_size, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_ipw = nn.Linear(hidden_size, self.n_t_classes, bias=False)

        # Outcome
        self.hidden2hidden_outcome_f = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear((self.x_emb_size + hidden_size) + self.x_static_emb_size + 1, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_outcome_f = nn.Linear(hidden_size, self.n_classes, bias=False)

        self.hidden2hidden_outcome_cf = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.x_emb_size + hidden_size + self.x_static_emb_size + 1, hidden_size),
            nn.Dropout(0.3),
            nn.ReLU(),
        )
        self.hidden2out_outcome_cf = nn.Linear(hidden_size, self.n_classes, bias=False)


    def feature_encode(self, x, x_fr):

        f_hx = torch.randn(x.size(0), self.hidden_size)
        cf_hx = torch.randn(x.size(0), self.hidden_size)
        f_old = f_hx
        cf_old = cf_hx
        f_outputs = []
        f_zxs = []
        cf_outputs = []
        cf_zxs = []
        for i in range(x.size(1)):
            x_emb = self.x2emb(x[:, i, :])
            f_zx = torch.cat((x_emb, f_old), -1)
            f_zxs.append(f_zx)

            cf_zx = torch.cat((x_emb, cf_old), -1)
            cf_zxs.append(cf_zx)

            f_inputs = torch.cat((f_zx, x_fr[:,i].unsqueeze(1)), -1)

            cf_treatment = torch.where(x_fr.sum(1)==0, torch.Tensor([1]), torch.Tensor([0])).unsqueeze(1)
            cf_inputs = torch.cat((cf_zx, cf_treatment), -1)

            f_hx = self.rnn_f(f_inputs, f_hx)
            cf_hx = self.rnn_cf(cf_inputs, cf_hx)

            if i == 0:
                f_concat_input = torch.cat((f_hx, f_hx), 1)
                cf_concat_input = torch.cat((cf_hx, cf_hx), 1)
            else:
                f_attn_weights = self.attn_f(f_hx, torch.stack(f_outputs))
                f_context = f_attn_weights.bmm(torch.stack(f_outputs).transpose(0, 1))
                f_context = f_context.squeeze(1)
                f_concat_input = torch.cat((f_hx, f_context), 1)

                cf_attn_weights = self.attn_cf(cf_hx, torch.stack(cf_outputs))
                cf_context = cf_attn_weights.bmm(torch.stack(cf_outputs).transpose(0, 1))
                cf_context = cf_context.squeeze(1)
                cf_concat_input = torch.cat((cf_hx, cf_context), 1)

            f_concat_output = torch.tanh(self.concat_f(f_concat_input))
            f_old = f_concat_output

            cf_concat_output = torch.tanh(self.concat_cf(cf_concat_input))
            cf_old = cf_concat_output

            f_outputs.append(f_hx)
            cf_outputs.append(cf_hx)

        return f_zxs, cf_zxs


    def forward(self, x, x_demo, x_fr):

        f_zxs, cf_zxs = self.feature_encode(x, x_fr)

        # IPW
        ipw_outputs = []
        x_demo_emd = self.x_static2emb(x_demo)
        for i in range(len(f_zxs)):
            h = torch.cat((f_zxs[i], x_demo_emd), -1)
            h = self.hidden2hidden_ipw(h)
            ipw_out = self.hidden2out_ipw(h)
            ipw_outputs.append(ipw_out)


        # Outcome
        f_treatment = torch.where(x_fr.sum(1) > 0, torch.Tensor([1]), torch.Tensor([0])).unsqueeze(1)
        cf_treatment = torch.where(x_fr.sum(1) > 0, torch.Tensor([0]), torch.Tensor([1])).unsqueeze(1)

        # factual prediction

        f_zx_maxpool = torch.max(torch.stack(f_zxs), 0)

        f_hidden = torch.cat((f_zx_maxpool[0], x_demo_emd, f_treatment), -1)
        f_h = self.hidden2hidden_outcome_f(f_hidden)

        f_outcome_out = self.hidden2out_outcome_f(f_h)

        # counterfactual prediction

        cf_zx_maxpool = torch.max(torch.stack(cf_zxs), 0)

        cf_hidden = torch.cat((cf_zx_maxpool[0], x_demo_emd, cf_treatment), -1)
        cf_h = self.hidden2hidden_outcome_cf(cf_hidden)

        cf_outcome_out = self.hidden2out_outcome_cf(cf_h)


        return ipw_outputs, f_outcome_out, cf_outcome_out, f_h