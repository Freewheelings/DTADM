for i in range(1, 37):
    f = open('eu/' + str(i) + '_train_pos.txt', 'r')
    lines = f.readlines()
    nodes = []
    for line in lines:
        l = line.strip('\n').split(' ')
        nodes.append(int(l[0]))
        nodes.append(int(l[1]))
    print(i, len(list(set(nodes))))

