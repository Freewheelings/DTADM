import random
from datetime import datetime

import torch
import numpy as np
from Config import config_wikipedia as config #select your dataset here

from TaskHelper import TaskHelper, read_edges, read_all_edges

from MetaLearner import MetaLearner


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# config['k_shot'] = 2

# 设置随机数种子
setup_seed(12345)

target_layer = 5

train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg = read_edges(config)
all_edges = read_all_edges(config)

task_helper = TaskHelper(all_edges, train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg, target_layer, config)

train_tasks = task_helper.train_tasks
ft_tasks = task_helper.fine_tune_tasks
# print(ft_tasks)
test_tasks = task_helper.test_tasks
x, edge_index = task_helper.nodes, task_helper.edges

config['input_dim'] = 128
config['hidden_dim'] = 128
print(config)
input_dim = config['input_dim']
hidden_dim = config['hidden_dim']
output_dim = config['output_dim']
batch_size = config['batch_size']
meta_train_epoch = config['meta_train_epoch']
ft_epoch = config['fine_tune_epoch']
device = config['device']
model = MetaLearner(input_dim, hidden_dim, output_dim, config)
model = model.to(device)

print(model)

best_test_auc = -1
best_test_f1 = -1
best_test_acc = -1
best_test_pre = -1


def meta_train():
    global best_test_auc, best_test_acc, best_test_f1, best_test_pre
    for e in range(meta_train_epoch):
        random.shuffle(train_tasks)
        i = 0
        while i < len(train_tasks):
            batch_task = train_tasks[i:i + batch_size if i + batch_size < len(train_tasks) else len(train_tasks)]
            ls, acc, f1, auc, ap = model(x, edge_index, batch_task, config['meta_lr'], config['base_lr'])
            i += batch_size
            print("Time: ", datetime.now(), " Epoch {}/{}:".format(str(e+1), str(i + 1)),
                  " Meta Train loss:%.6f" % ls, " ACC:%.6f" % acc,
                  " F1:%.6f" % f1, " AUC:%.6f" % auc, "Pre:%.6f"%ap)
        if e != 0:
        # if True:
            best_test_auc, best_test_acc, best_test_f1, best_test_pre, = fine_tune(model, best_test_auc, best_test_acc, best_test_f1, best_test_pre, e)#


            print("best: t_auc:%.6f   t_acc:%.6f   t_f1:%.6f   t_pre:%.6f" % (best_test_auc, best_test_acc, best_test_f1, best_test_pre))


def fine_tune(model, best_test_auc, best_test_acc, best_test_f1, best_test_pre, epoch):
    print("##################_begine_fine_tune_##################")
    fine_tune_model = MetaLearner(config['input_dim'], config['hidden_dim'], 1, config)
    fine_tune_model = fine_tune_model.to(device)
    fine_tune_model.load_state_dict(model.state_dict())
    for e in range(ft_epoch):
        random.shuffle(ft_tasks)
        i = 0
        while i < len(ft_tasks):

            batch_task = ft_tasks[i:i + batch_size if i + batch_size < len(ft_tasks) else len(ft_tasks)]
            ls, acc, f1, auc, ap = fine_tune_model(x, edge_index, batch_task, config['fine_tune_meta_lr'], config['fine_tune_base_lr'])
            i += batch_size
            print("Time: ", datetime.now(), " Epoch {}/{}:".format(str(e + 1), str(i + 1)),
                  " Fine tune loss:%.6f" % ls, " ACC:%.6f" % acc,
                  " F1:%.6f" % f1, " AUC:%.6f" % auc, "Pre:%.6f"%ap)
        acc, f1, auc, pre = eval(fine_tune_model)#
        fine_tune_model.train()
        if auc > best_test_auc:
            best_test_auc = auc
            best_test_acc = acc
            best_test_f1 = f1
            best_test_pre = pre
            dic = {"state_dict": fine_tune_model.state_dict(), "meta_train_epoch": epoch, "fine_tune_epoch": e}
            torch.save(dic, 'Saved_models/pac_main_1.pkl')
    return best_test_auc, best_test_acc, best_test_f1, best_test_pre


def eval(fine_tune_model):
    fine_tune_model.eval()
    _,_,_,ls, acc, f1, auc, ap, _, _ = fine_tune_model.evaluate(x, edge_index, test_tasks, config['fine_tune_meta_lr'], config['fine_tune_base_lr'])#
    print("Time: ", datetime.now(),
          " Test loss:%.6f" % ls, " ACC:%.6f" % acc,
          " F1:%.6f" % f1, " AUC:%.6f" % auc, "Pre:%.6f"%ap)
    return acc, f1, auc, ap


meta_train()#
