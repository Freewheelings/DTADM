import torch
import torch.nn as nn
from CrossLayerAttn import FullModel
from copy import deepcopy
import torch.optim as opt
from Evaluator import Evaluator
import numpy as np


class MetaLearner(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(MetaLearner, self).__init__()
        self.model = FullModel(input_dim, hidden_dim, output_dim, config)
        self.config = config
        self.loss = nn.BCELoss()
        self.evaluator = Evaluator()

    def forward(self, x_s, edge_index_s, batch_tasks, lr1, lr2):
        model_dict = self.model.state_dict()
        task_num = len(batch_tasks)
        que_loss, que_acc, que_auc, que_f1, que_pre = 0.0, 0.0, 0.0, 0.0, 0.0
        for k in range(task_num):
            sup, que = batch_tasks[k]
            # src, tgt, y, layer = sup
            src, y, layer = sup
            jump = set(y)
            # print(jump)
            if len(jump) == 1:
                print('jump')
                continue
            m = deepcopy(self.model)
            optim = opt.SGD(m.parameters(), lr1)
            for i in range(self.config['reptile']):
                optim.zero_grad()
                # pred, _, _ = m(x_s, edge_index_s, src, tgt, layer)
                pred, _, _ = m(x_s, edge_index_s, src, layer)#
                y_true = torch.FloatTensor(y).cuda()#
                loss = self.loss(pred, y_true)
                loss.backward()
                optim.step()
            # src, tgt, y, layer = que
            # pred, _, _ = m(x_s, edge_index_s, src, tgt, layer)
            src, y, layer = que
            pred, _, _ = m(x_s, edge_index_s, src, layer)
            y = torch.FloatTensor(y).cuda()
            loss = self.loss(pred, y)
            loss.backward()
            acc, f1, auc, ap = self.evaluator.evaluation(y, pred)
            que_loss += loss.item()
            que_acc += acc
            que_f1 += f1
            que_auc += auc
            que_pre += ap
            for name, param in m.named_parameters():
                if param.grad is not None:
                    model_dict[name] -= lr2 * param.grad
        self.model.load_state_dict(model_dict)
        return que_loss / task_num, que_acc / task_num, que_f1 / task_num, que_auc / task_num, que_pre / task_num

    def evaluate(self, x_s, edge_index_s, batch_tasks, lr1, lr2):
        task_num = len(batch_tasks)
        # print(batch_tasks)
        que_loss, que_acc, que_auc, que_f1, que_pre = 0.0, 0.0, 0.0, 0.0, 0.0
        ats, atts = [], []
        # p, s, t, tr = [], [], [], []
        p, s, tr = [], [], []
        for k in range(task_num):
            sup, que = batch_tasks[k]
            # src, tgt, y, layer = sup
            src, y, layer = sup

            m = deepcopy(self.model)
            optim = opt.SGD(m.parameters(), lr1)
            for i in range(self.config['reptile']):
                optim.zero_grad()
                # pred, _, _ = m(x_s, edge_index_s, src, tgt, layer)
                pred, _, _ = m(x_s, edge_index_s, src, layer)

                
                y_true = torch.FloatTensor(y).cuda()
                loss = self.loss(pred, y_true)
                loss.backward()
                optim.step()
            # src, tgt, y, layer = que
            # pred, attn, attns = m(x_s, edge_index_s, src, tgt, layer)
            src, y, layer = que
            pred, attn, attns = m(x_s, edge_index_s, src, layer)
            # print('####################')
            # print('####################')
            # print('####################')
            # print(pred)
            # temp = set(y)
            # if len(temp) == 1:
            #     pred = torch.FloatTensor(np.concatenate((pred.cpu().detach().numpy(), np.array([1, 0])))).cuda()
            #     y = y + [0,1]
            y = torch.FloatTensor(y).cuda()
            loss = self.loss(pred, y)
            loss.backward()
            acc, f1, auc, ap = self.evaluator.evaluation(y, pred)#
            que_loss += loss.item()
            que_acc += acc
            que_f1 += f1
            que_auc += auc
            que_pre += ap
            ats.append(attn)
            atts.append(attns)
            p.append(pred)
            s.append(src)
            # t.append(tgt)
            tr.append(y)

        # return p, s, t, tr, que_loss / task_num, que_acc / task_num, que_f1 / task_num, que_auc / task_num, que_pre / task_num, ats, atts
        return p, s, tr, que_loss / task_num, que_acc / task_num, que_f1 / task_num, que_auc / task_num, que_pre / task_num, ats, atts


#