import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class DL_BBBP_onlygraph(torch.nn.Module):
    def __init__(self, num_features_xd=78, output_dim=128, dropout=0.2, 
                 fp_size=26, input_size=1, hidden_size=100, num_layers=1, n_output=1):
        super(DL_BBBP_onlygraph, self).__init__()
        self.fp_size = fp_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_output = n_output

        # graph branch
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # fc layers
        self.fc1 = nn.Linear(1*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        # get graph input
        xg, edge_index, batch = data.x, data.edge_index, data.batch
        xg = self.conv1(xg, edge_index)
        xg = self.relu(xg)
        xg = self.conv2(xg, edge_index)
        xg = self.relu(xg)
        xg = self.conv3(xg, edge_index)
        xg = self.relu(xg)  # 运行完后torch.Size([6411, 312])
        xg = gmp(xg, batch) # global max pooling, 运行完后torch.Size([256, 312])
        # flatten
        xg = self.fc_g1(xg)
        xg = self.relu(xg)
        xg = self.dropout(xg)
        xg = self.fc_g2(xg)
        xg = self.dropout(xg)

        # concat
        xc = xg
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = self.sigmoid(out)
        return out