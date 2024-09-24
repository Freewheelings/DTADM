import random
import networkx as nx

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2
dataset = '/home/ty/PythonProject7/Dataset/pac/'
layer_num = 16
max_node_num = 514


def write_file(filename, edges):
    f = open(filename, 'w')
    for edge in edges:
        f.write(str(edge[0]) + ' ' + str(edge[1]) + '\n')
    f.close()


def generate_neg_sample(edges):
    nodes = [i for i in range(max_node_num)]
    g = nx.Graph()
    g.add_edges_from(edges)
    # g.add_nodes_from(nodes)
    count = 0
    nodes = list(g.nodes)
    negs = []
    while count < len(edges):
        src = random.choice(nodes)
        tgt = random.choice(nodes)
        if src == tgt:
            continue
        if [src, tgt] in edges or [tgt, src] in edges:
            continue
        if [src, tgt] in negs or [tgt, src] in negs:
            continue
        negs.append([src, tgt])
        count += 1
    return negs


def split_data():
    for i in range(1, layer_num + 1):
        with open(dataset + str(i) + '.txt', 'r') as f:
            lines = f.readlines()
        edges = []
        for line in lines:
            edge = line.strip('\n').split()
            edges.append([edge[0], edge[1]])
        neg = generate_neg_sample(edges)
        pos = edges
        train_edge_num = int(train_ratio * len(pos))
        train_pos = random.sample(pos, train_edge_num)
        train_neg = random.sample(neg, train_edge_num)
        test_edge_num = int(test_ratio * len(pos))
        rest_pos = [e for e in pos if e not in train_pos]
        rest_neg = [e for e in neg if e not in train_neg]
        test_pos = random.sample(rest_pos, test_edge_num)
        test_neg = random.sample(rest_neg, test_edge_num)
        valid_pos = [e for e in rest_pos if e not in test_pos]
        valid_neg = [e for e in rest_neg if e not in test_neg]
        train_pos_file = dataset + str(i) + '_train_pos.txt'
        train_neg_file = dataset + str(i) + '_train_neg.txt'
        valid_pos_file = dataset + str(i) + '_valid_pos.txt'
        valid_neg_file = dataset + str(i) + '_valid_neg.txt'
        test_pos_file = dataset + str(i) + '_test_pos.txt'
        test_neg_file = dataset + str(i) + '_test_neg.txt'
        write_file(train_pos_file, train_pos)
        write_file(train_neg_file, train_neg)
        write_file(valid_pos_file, valid_pos)
        write_file(valid_neg_file, valid_neg)
        write_file(test_pos_file, test_pos)
        write_file(test_neg_file, test_neg)

split_data()