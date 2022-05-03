import dgl
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes=1):
        super(GCN, self).__init__()
        # self.conv1 = GraphConv(in_feats, h_feats)
        # self.conv2 = GraphConv(h_feats, num_classes)
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g):
        in_feat = g.ndata['layer'].float()
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')
