
f1 = open('ss_train2.csv', 'w')
f2 = open('ss_test2.csv', 'w')
f = open('ss.csv', 'r')

lines = 0
list = []
for line in f.readlines():
    label, p1, p2 = line.strip().split(',')
    list.append((label, p1, p2))
    lines += 1

cnt = 1
for i in range(10):
    for j in range(300):
        label, p1, p2 = list[i * 300 + j]
        f2.write(label + ',' + p1 + ',' + p2 + '\n')
    for j in range(300):
        label, p1, p2 = list[lines - i * 300 - j - 1]
        f2.write(label + ',' + p1 + ',' + p2 + '\n')

for i in range(3001, lines - 3000 + 1):
    label, p1, p2 = list[i]
    f1.write(label + ',' + p1 + ',' + p2 + '\n')