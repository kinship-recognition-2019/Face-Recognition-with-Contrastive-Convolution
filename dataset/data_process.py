
f1 = open('fs_train.csv', 'w')
f2 = open('fs_test.csv', 'w')
f = open('fs.csv', 'r')

# lines = 0
# list = []
# for line in f.readlines():
#     label, p1, p2 = line.strip().split(',')
#     list.append((label, p1, p2))
#     lines += 1
#
# cnt = 1
# for i in range(10):
#     for j in range(300):
#         label, p1, p2 = list[i * 300 + j]
#         f2.write(label + ',' + p1 + ',' + p2 + '\n')
#     for j in range(300):
#         label, p1, p2 = list[lines - i * 300 - j - 1]
#         f2.write(label + ',' + p1 + ',' + p2 + '\n')
#
# for i in range(3001, lines - 3000 + 1):
#     label, p1, p2 = list[i]
#     f1.write(label + ',' + p1 + ',' + p2 + '\n')

a = [[0 for i in range(50)] for j in range(1000)]

for line in f.readlines()[1:]:
    _, label, p1, p2 = line.strip().split(',')
    F, MID, P = p1.split('/')
    if a[int(F[1:4])][int(MID[3:])] % 20 == 0:
        a[int(F[1:4])][int(MID[3:])] += 1
        f2.write(label + ',' + p1 + ',' + p2 + '\n')
    else:
        a[int(F[1:4])][int(MID[3:])] += 1
        f1. write(label + ',' + p1 + ',' + p2 + '\n')


