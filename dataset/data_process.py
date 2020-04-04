
path_ori = './FIW_List/grandmother-granddaughter/gmgd.csv'
path_train = './FIW_List/grandmother-granddaughter/gmgd_train.csv'
path_test = './FIW_List/grandmother-granddaughter/gmgd_test.csv'


lines = 0
list = []
a = [[0 for i in range(50)] for j in range(1000)]

with open(path_ori, 'r') as f, open(path_train, 'w') as f1:
    for line in f.readlines()[1:]:
        _, label, p1, p2 = line.strip().split(',')
        F, MID, P = p1.split('/')
        if a[int(F[1:4])][int(MID[3:])] % 15 == 0:
            a[int(F[1:4])][int(MID[3:])] += 1
            # f2.write(label + ',' + p1 + ',' + p2 + '\n')
            lines += 1
            list.append((label, p1, p2))
        else:
            a[int(F[1:4])][int(MID[3:])] += 1
            f1. write(label + ',' + p1 + ',' + p2 + '\n')

with open(path_train, 'a') as f1, open(path_test, 'w') as f2:
    cnt = 1
    for i in range(1):
        for j in range(300):
            label, p1, p2 = list[i * 300 + j]
            f2.write(label + ',' + p1 + ',' + p2 + '\n')
        for j in range(300):
            label, p1, p2 = list[lines - i * 300 - j - 1]
            f2.write(label + ',' + p1 + ',' + p2 + '\n')

    for i in range(301, lines - 300 + 1):
        label, p1, p2 = list[i]
        f1.write(label + ',' + p1 + ',' + p2 + '\n')
