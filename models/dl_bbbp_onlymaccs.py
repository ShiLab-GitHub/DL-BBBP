import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class DL_BBBP_onlymaccs(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=128, dropout=0.2, 
                 fp_size=26, input_size=1, hidden_size=100, num_layers=1, n_output=1):
        super(DL_BBBP_onlymaccs, self).__init__()
        self.fp_size = fp_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # MACCS branch
        neuro_num = 512
        self.fc_m1 = nn.Linear(fp_size, neuro_num)
        self.fc_m2 = nn.Linear(neuro_num, output_dim)

        # fc layers
        self.fc1 = nn.Linear(1*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):

        # get MACCS input
        xm = data.MACCS.view(-1, self.fp_size) # torch.Size([256, 206])
        xm = self.fc_m1(xm)
        xm = self.relu(xm)
        xm = self.dropout(xm)
        xm = self.fc_m2(xm)
        xm = self.relu(xm)
        xm = self.dropout(xm)

        # concat
        xc = xm
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out