from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch
import numpy as np


class Evaluator:
    def evaluation(self, real_score, pred_score):
        real_score = real_score.cpu().detach().numpy()
        pred_score = pred_score.cpu().detach().numpy()
        pred_label = []
        for p in pred_score:
            if p > 0.5:
                pred_label.append(1)
            else:
                pred_label.append(0)

        pred_label = np.array(pred_label).astype(int)
        # print('real_score', real_score)
        # print('pred_score', pred_score)
        # print('pred_label', pred_label)

        # pred_label = pred_score > pred_score.mean()
        
        t = set(real_score)
        if len(t) == 1:
            print('mark')
            real_score += np.array([0, 1])
            pred_label += [1,0]

        acc = accuracy_score(real_score, pred_label)
        ap = average_precision_score(real_score, pred_label)
        f1 = f1_score(real_score, pred_label, average='macro')
        auc = roc_auc_score(real_score, pred_score)

        # print('micro', f1_score(real_score, pred_label, average='micro'))
        # print('macro', f1_score(real_score, pred_label, average='macro'))

        return acc, f1, auc, ap
