import torch
import torch.nn as nn
import torch.nn.functional as F
from GATModel import GATModel


# from MultiHeadAtten import MultiHeadAttention
# from torch_geometric.utils import to_scipy_sparse_matrix


class MultiAttenFusion(nn.Module):
    def __init__(self, input_dim, config):
        super(MultiAttenFusion, self).__init__()
        self.cur_emb = GATModel(input_dim, config)
        self.oth_emb = GATModel(input_dim, config)
        # self.fc = nn.Linear(input_dim * config['gat_head'], input_dim)
        
        self.fc1 = nn.Linear(128, input_dim)  # New layer: fc1
        self.fc2 = nn.Linear(input_dim, input_dim)  # New layer: fc2
        self.fc3 = nn.Linear(input_dim, input_dim)  # New layer: fc3
        self.fc4 = nn.Linear(input_dim, input_dim)  # New layer: fc4
        
        self.two_layer_attn = nn.ModuleList()
        for i in range(config['layers']):
            self.two_layer_attn.append(nn.MultiheadAttention(input_dim, config['n_head'], dropout=0.2))
        self.all_layer_attn = nn.MultiheadAttention(input_dim, config['n_head'], dropout=0.2, batch_first=True)
        
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.xavier_normal_(self.fc4.weight)
        
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.constant_(self.fc4.bias, 0)
        

#     def forward(self, x_s, edge_index_s, src_x, tgt_x, layer):
#         cur_x, cur_edge_index = x_s[layer], edge_index_s[layer]

#         cur_emb = self.cur_emb(cur_x, cur_edge_index)
#         cur_src = torch.cat([cur_emb[index].unsqueeze(0) for index in src_x], dim=0)
#         cur_tgt = torch.cat([cur_emb[index].unsqueeze(0) for index in tgt_x], dim=0)
#         cur = torch.cat((cur_src, cur_tgt), dim=-1)
#         cur = self.fc(cur)
#         oth = []
#         attns = []
#         for i in range(len(x_s)):
#             if i != layer:
#                 oth_emb = self.oth_emb(x_s[i], edge_index_s[i])
#                 x, att = self.two_layer_attn[i](cur, oth_emb, oth_emb)
#                 oth.append(x.unsqueeze(1))
#                 attns.append(att)
#         oth = torch.cat(oth, dim=1)
#         # print(oth.size())
#         cur = cur.unsqueeze(1)
#         # print(cur.size())
#         oth, attn = self.all_layer_attn(cur, oth, oth)
#         x = torch.cat((cur, oth), dim=-1)
#         # print(len(attn))
#         x = x.squeeze(1)
#         return x, attn, attns
    def forward(self, x_s, edge_index_s, src_x, layer):
            cur_x, cur_edge_index = x_s[layer], edge_index_s[layer]

            cur_emb = self.cur_emb(cur_x, cur_edge_index)
            cur_src = torch.cat([cur_emb[index].unsqueeze(0) for index in src_x], dim=0)
            cur = cur_src
            # print(cur.shape)
            cur = self.fc1(cur)
            cur = F.relu(cur)
            cur = self.fc2(cur)
            cur = F.relu(cur)
            cur = self.fc3(cur)
            cur = F.relu(cur)
            cur = self.fc4(cur)
            
            oth = []
            attns = []
            for i in range(len(x_s)):
                if i != layer:
                    oth_emb = self.oth_emb(x_s[i], edge_index_s[i])
                    x, att = self.two_layer_attn[i](cur, oth_emb, oth_emb)
                    oth.append(x.unsqueeze(1))
                    attns.append(att)
            oth = torch.cat(oth, dim=1)
            # print(oth.size())
            cur = cur.unsqueeze(1)
            # print(cur.size())
            oth, attn = self.all_layer_attn(cur, oth, oth)
            x = torch.cat((cur, oth), dim=-1)
            # print(len(attn))
            x = x.squeeze(1)
            return x, attn, attns


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)

    def forward(self, x):
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.squeeze(1).sigmoid()
        return x


class FullModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(FullModel, self).__init__()
        self.multi_attn = MultiAttenFusion(input_dim, config)
        self.classifier = Classifier(input_dim*2, hidden_dim, output_dim)

    # def forward(self, x_s, edge_index_s, src_x, tgt_x, layer):
    #     x, attn, attns = self.multi_attn(x_s, edge_index_s, src_x, tgt_x, layer)
    #     x = self.classifier(x)
    #     return x, attn, attns
    def forward(self, x_s, edge_index_s, src_x, layer):
        x, attn, attns = self.multi_attn(x_s, edge_index_s, src_x, layer)#
        x = self.classifier(x)
        return x, attn, attns

