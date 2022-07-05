import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import SAGEConv, MaxPooling, SortPooling, GATv2Conv, GroupRevRes


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, dout_prob):
        super(GCN, self).__init__()
        self.sage_lstm_h_1 = SAGEConv(in_feats=in_feats, out_feats=h_feats, aggregator_type='lstm')
        self.sage_lstm_h_2 = SAGEConv(in_feats=h_feats, out_feats=h_feats//2, aggregator_type='lstm')
        self.sage_lstm_h_3 = SAGEConv(in_feats=h_feats//2, out_feats=h_feats//4, aggregator_type='lstm')
        self.sage_lstm_h_4 = SAGEConv(in_feats=h_feats//4, out_feats=h_feats//8, aggregator_type='lstm')
        self.sage_lstm_h_5 = SAGEConv(in_feats=h_feats//8, out_feats=h_feats//16, aggregator_type='lstm')
        self.pool_1_m = MaxPooling()
        self.fc1 = nn.Linear(in_features=h_feats//16, out_features=h_feats//32)
        self.fc2 = nn.Linear(in_features=h_feats//32, out_features=1)
        # self.dropout = nn.Dropout(p=dout_prob)

    def forward(self, g):
        with g.local_scope():
            x = g.ndata['layer'].float()
            x = self.sage_lstm_h_1(g, x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.sage_lstm_h_2(g, x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.sage_lstm_h_3(g, x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.sage_lstm_h_4(g, x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.sage_lstm_h_5(g, x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.pool_1_m(g, x)
            x = x.view(x.size()[0], -1)
            x = self.fc1(x)
            x = F.relu(x)
            # x = self.dropout(x)
            x = self.fc2(x)
            x = F.relu(x)
            return x
