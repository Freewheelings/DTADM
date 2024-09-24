import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GATModel(nn.Module):
    def __init__(self, d_node,  config):
        super(GATModel, self).__init__()
        self.config = config
        self.head = config['gat_head']
        self.emb = nn.Linear(1, d_node)
        self.gat1 = GATConv(d_node, d_node//self.head, heads=self.head)
        self.gat2 = GATConv(d_node, d_node, heads=1)
        self.gat3 = GATConv(d_node, d_node, heads=1)
        self.gat4 = GATConv(d_node, d_node, heads=1)
        nn.init.xavier_normal_(self.emb.weight)
        nn.init.constant_(self.emb.bias, 0)

    def forward(self, x, edge_index):
        x = torch.FloatTensor(x).to(self.config['device'])
        edge_index = torch.LongTensor(edge_index).to(self.config['device'])
        # x = F.dropout(x, training=self.training)
        x = self.emb(x)
        edge_index = edge_index.t().contiguous()
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        #
        x = self.gat2(x, edge_index)
        x = F.relu(x)
        #
        x = self.gat3(x, edge_index)
        x = F.relu(x)
        #
        x = self.gat4(x, edge_index)
        return x
