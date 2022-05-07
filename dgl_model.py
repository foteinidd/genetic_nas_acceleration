import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv, MaxPooling


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.sage_lstm_h_1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='lstm')
        self.sage_lstm_h_2 = SAGEConv(in_feats=h_feats, out_feats=h_feats, aggregator_type='lstm')
        self.pool_1_m = MaxPooling()
        self.fc1 = nn.Linear(in_features=h_feats, out_features=h_feats//2)
        # self.fc2 = nn.Linear(in_features=h_feats//2, out_features=h_feats//4)
        self.fc3 = nn.Linear(in_features=h_feats//2, out_features=1)

    def forward(self, g):
        with g.local_scope():
            x = g.ndata['layer'].float()
            # print(x.shape)
            x = self.sage_lstm_h_1(g, x)
            # print(x.shape)
            # x = F.relu(x)
            # print(x.shape)
            x = self.sage_lstm_h_2(g, x)
            # print(x.shape)
            # x = F.relu(x)
            # print(x.shape)
            x = self.pool_1_m(g, x)
            # print(x.shape)
            x = x.view(x.size()[0], -1)
            # print(x.shape)
            x = self.fc1(x)
            # print(x.shape)
            x = F.relu(x)
            # print(x.shape)
            # x = self.fc2(x)
            # print(x.shape)
            # x = F.relu(x)
            # print(x.shape)
            x = self.fc3(x)
            # print(x.shape)
            # x = F.softmax(x, dim=1)
            # print(x.shape)
            return x
