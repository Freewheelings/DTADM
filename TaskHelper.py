import random
import networkx as nx


class TaskHelper:
    def __init__(self, all_edges, train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg, target_layer, config):
        self.target_layer = target_layer
        self.all_edges = all_edges
        self.layers = [i for i in range(len(all_edges)) if i != target_layer]
        self.config = config
        self.train_num_tasks = config['train_task_num']
        self.ft_num_tasks = config['ft_task_num']
        self.k_shot = config['k_shot']
        self.graphs, self.dics, self.degrees = self.generate_graphs()
        self.nodes, self.edges = self.x_each_layer()
        # self.train_pos = self.process_edges(train_pos)
        # self.train_neg = self.process_edges(train_neg)
        # self.valid_pos = self.process_edges(valid_pos)
        # self.valid_neg = self.process_edges(valid_neg)
        # self.test_pos = self.process_edges(test_pos)
        # self.test_neg = self.process_edges(test_neg)
        self.train_pos = self.process_nodes(train_pos)
        self.train_neg = self.process_nodes(train_neg)
        self.valid_pos = self.process_nodes(valid_pos)
        self.valid_neg = self.process_nodes(valid_neg)
        self.test_pos = self.process_nodes(test_pos)
        self.test_neg = self.process_nodes(test_neg)
        self.train_tasks = []
        self.fine_tune_tasks = []
        # random.seed(config['seed'])

        i = 0
        while i < self.train_num_tasks:
            l = random.choice(self.layers)
            task = self.construct_task(self.train_pos, self.train_neg, l)#
            self.train_tasks.append(task)
            i += 1
        i = 0
        while i < self.ft_num_tasks:
            task = self.construct_task(self.train_pos, self.train_neg, self.target_layer)
            self.fine_tune_tasks.append(task)
            i += 1
        self.test_tasks = self.construct_test_task(self.train_pos, self.train_neg, self.test_pos[self.target_layer],
                                                   self.test_neg[self.target_layer], self.target_layer)

    def generate_graphs(self):
        graphs = []
        dics = []
        degrees = []
        for i in range(len(self.all_edges)):
            g = nx.Graph()
            g.add_edges_from(self.all_edges[i])
            graphs.append(g)
            nodes = list(g.nodes)
            dic = {}
            degree = {}
            for j in range(len(nodes)):
                dic[nodes[j]] = j
                degree[nodes[j]] = nx.degree(g, nodes[j])
            dics.append(dic) #点索引
            degrees.append(degree)
        return graphs, dics, degrees

    # def process_edges(self, edges):
    #     res = []
    #     for i in range(len(edges)):
    #         edge = edges[i]
    #         # graph = self.graphs[i]
    #         dic = self.dics[i]
    #         # degree = self.degrees[i]
    #         temp = []
    #         for e in edge:
    #             temp.append([dic[e[0]], dic[e[1]]])
    #         res.append(temp)
    #     return res
    
    def process_nodes(self, edges):
        res = []
        for i in range(len(edges)):
            edge = edges[i]
            # graph = self.graphs[i]
            dic = self.dics[i]
            # degree = self.degrees[i]
            temp = []
            for e in edge:
                temp.append(dic[e[0]])
            res.append(temp)
        return res

    def x_each_layer(self):
        total_nodes = self.config['total_nodes']
        nodes, edges = [], []
        for g in self.graphs:
            node = list(g.nodes)
            degrees = {}
            edge = list(g.edges)
            dic = {}
            for n in range(len(node)):
                dic[node[n]] = n
                degrees[node[n]] = nx.degree(g, node[n])
            d_node = [[degrees[n]] for n in node]
            d_edge = [[dic[e[0]], dic[e[1]]] for e in edge]
            reverse = [[e[1], e[0]] for e in d_edge]
            self_loop = [[dic[n], dic[n]] for n in node]
            edge = d_edge + reverse + self_loop
            nodes.append(d_node)
            edges.append(edge)
        return nodes, edges

    def construct_task(self, pos, neg, layer):
        # pos = pos[layer]
        # neg = neg[layer]
        # sup_pos = random.sample(pos, self.k_shot)
        # sup_neg = random.sample(neg, self.k_shot)
        # sup_y = [1] * self.k_shot + [0] * self.k_shot
        # src = [e[0] for e in sup_pos] + [e[0] for e in sup_neg]#source
        # tgt = [e[1] for e in sup_pos] + [e[1] for e in sup_neg]
        # sup = (src, tgt, sup_y, layer)
        # que_pos = random.sample([e for e in pos if e not in sup_pos], self.k_shot)
        # que_neg = random.sample([e for e in neg if e not in sup_neg], self.k_shot)
        # que_y = [1] * self.k_shot + [0] * self.k_shot
        # src = [e[0] for e in que_pos] + [e[0] for e in que_neg]
        # tgt = [e[1] for e in que_pos] + [e[1] for e in que_neg]
        # que = (src, tgt, que_y, layer)
        # return (sup, que)
        
        pos = pos[layer]
        neg = neg[layer]
        sup_pos = random.sample(pos, self.k_shot)
        sup_neg = random.sample(neg, self.k_shot)
        sup_y = [1] * self.k_shot + [0] * self.k_shot
        src = [e for e in sup_pos] + [e for e in sup_neg]#source
        # del src[-1]
        # del sup_y[-1]
        sup = (src, sup_y, layer)
        # print(sup)
        #有效点
        
        # print('len_sup_pos:', len(sup_pos))
        # print(sup_pos)
        # print('len_sup_neg:', len(sup_neg))
        # print(sup_neg)
        
        que_pos = random.sample([e for e in pos if e not in sup_pos], self.k_shot)
        que_neg = random.sample([e for e in neg if e not in sup_neg], self.k_shot)
        # que_pos = random.sample([e for e in pos], self.k_shot)
        # que_neg = random.sample([e for e in neg], self.k_shot)
        
        que_y = [1] * self.k_shot + [0] * self.k_shot
        src = [e for e in que_pos] + [e for e in que_neg]
        # del src[-1]
        # del que_y[-1]
        que = (src, que_y, layer)
        # print(que)
        # if(len(que_y) > 0 and len(sup_y) > 0):
        return (sup, que)


    # def construct_test_task(self, pos, neg, test_pos, test_neg, layer):
        # i = 0
        # tasks = []
        # while i < len(test_pos):
        #     que_pos = test_pos[i:i + self.k_shot if i + self.k_shot < len(test_pos) else len(test_pos)]
        #     que_neg = test_neg[i:i + self.k_shot if i + self.k_shot < len(test_pos) else len(test_pos)]
        #     que_y = [1] * len(que_pos) + [0] * len(que_neg)
        #     src = [e[0] for e in que_pos] + [e[0] for e in que_neg]
        #     tgt = [e[1] for e in que_pos] + [e[1] for e in que_neg]
        #     que = (src, tgt, que_y, layer)
        #     p = pos[layer]
        #     n = neg[layer]
        #     sup_pos = random.sample(p, len(que_pos))
        #     sup_neg = random.sample(n, len(que_pos))
        #     sup_y = [1] * len(que_pos) + [0] * len(que_pos)
        #     src = [e[0] for e in sup_pos] + [e[0] for e in sup_neg]
        #     tgt = [e[1] for e in sup_pos] + [e[1] for e in sup_neg]
        #     sup = (src, tgt, sup_y, layer)
        #     tasks.append((sup, que))
        #     i += self.k_shot
        # return tasks
        
    def construct_test_task(self, pos, neg, test_pos, test_neg, layer):
        i = 0
        tasks = []
        while i < len(test_pos):
            que_pos = test_pos[i:i + self.k_shot if i + self.k_shot < len(test_pos) else len(test_pos)]
            que_neg = test_neg[i:i + self.k_shot if i + self.k_shot < len(test_pos) else len(test_pos)]
            que_y = [1] * len(que_pos) + [0] * len(que_neg)
            src = [e for e in que_pos] + [e for e in que_neg]
            que = (src, que_y, layer)
            
            p = pos[layer]
            n = neg[layer]
            sup_pos = random.sample(p, len(que_pos))
            sup_neg = random.sample(n, len(que_pos))
            sup_y = [1] * len(que_pos) + [0] * len(que_pos)
            src = [e for e in sup_pos] + [e for e in sup_neg]
            sup = (src, sup_y, layer)
            
            if len(set(que_y))==1 or len(set(sup_y))==1:
                break
            
            tasks.append((sup, que))
            i += self.k_shot
        return tasks
        


def read_edges(config):
    dataset = 'Dataset/' + config['dataset']
    layer_num = config['layers']
    train_pos, train_neg = [], []
    valid_pos, valid_neg = [], []
    test_pos, test_neg = [], []

    def read(filename):
        edges = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            edge = line.strip('\n').split()
            edges.append((int(edge[0]), int(edge[1])))
        return edges
    
    # def read(filename):
    #     edges = []
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()
    #     for line in lines:
    #         edge = line.strip('\n').split()
    #         nodes.append(int(edge[0]))
    #     return nodes

    for i in range(1, 1 + layer_num):
        train_p = read(dataset + '/' + str(i) + '_train_pos.txt')
        train_n = read(dataset + '/' + str(i) + '_train_neg.txt')
        val_p = read(dataset + '/' + str(i) + '_valid_pos.txt')
        val_n = read(dataset + '/' + str(i) + '_valid_neg.txt')
        test_p = read(dataset + '/' + str(i) + '_test_pos.txt')
        test_n = read(dataset + '/' + str(i) + '_test_neg.txt')
        train_pos.append(train_p)
        train_neg.append(train_n)
        valid_pos.append(val_p)
        valid_neg.append(val_n)
        test_pos.append(test_p)
        test_neg.append(test_n)
    return train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg

def read_all_edges(config):
    dataset = 'Dataset/' + config['dataset']
    layer_num = config['layers']
    def read(filename):
        ed = []
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            edge = line.strip('\n').split()
            ed.append((int(edge[0]), int(edge[1])))
        return ed
    edges = []
    for i in range(1, 1 + layer_num):
        edge = read(dataset + '/' + str(i) + '.txt')
        edges.append(edge)
    return edges