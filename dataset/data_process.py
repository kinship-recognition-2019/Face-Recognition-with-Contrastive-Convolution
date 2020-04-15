# ori = ['./FIW_List/brother-brother/bb.csv', './FIW_List/father-daughter/fd.csv', './FIW_List/father-son/fs.csv',
#        './FIW_List/grandfather-granddaughter/gfgd.csv', './FIW_List/grandfather-grandson/gfgs.csv',
#        './FIW_List/grandmother-granddaughter/gmgd.csv', './FIW_List/grandmother-grandson/gmgs.csv',
#        './FIW_List/mother-daughter/md.csv', './FIW_List/mother-son/ms.csv',
#        './FIW_List/sibs/sibs.csv', './FIW_List/sister-sister/ss.csv']
# train = ['./FIW_List/brother-brother/bb_train.csv', './FIW_List/father-daughter/fd_train.csv', './FIW_List/father-son/fs_train.csv',
#        './FIW_List/grandfather-granddaughter/gfgd_train.csv', './FIW_List/grandfather-grandson/gfgs_train.csv',
#        './FIW_List/grandmother-granddaughter/gmgd_train.csv', './FIW_List/grandmother-grandson/gmgs_train.csv',
#        './FIW_List/mother-daughter/md_train.csv', './FIW_List/mother-son/ms_train.csv',
#        './FIW_List/sibs/sibs_train.csv', './FIW_List/sister-sister/ss_train.csv']
# test = ['./FIW_List/brother-brother/bb_test.csv', './FIW_List/father-daughter/fd_test.csv', './FIW_List/father-son/fs_test.csv',
#        './FIW_List/grandfather-granddaughter/gfgd_test.csv', './FIW_List/grandfather-grandson/gfgs_test.csv',
#        './FIW_List/grandmother-granddaughter/gmgd_test.csv', './FIW_List/grandmother-grandson/gmgs_test.csv',
#        './FIW_List/mother-daughter/md_test.csv', './FIW_List/mother-son/ms_test.csv',
#        './FIW_List/sibs/sibs_test.csv', './FIW_List/sister-sister/ss_test.csv']
# nums = [20000, 4500, 10000, 1200, 360, 1000, 320, 3600, 6000, 9000, 1500]

ori = ['./FIW_List/father-son/fs.csv']
train = ['./FIW_List/father-son/fs_train']
test = ['./FIW_List/father-son/fs_test']

nums = [10000]

path_ori = ''
path_train = ''
path_test = ''

for i in range(1):
    path_ori = ori[i]
    path_train = train[i]
    path_test = test[i]
    cnt1 = 0

    with open(path_ori, 'r') as f, open(path_train+'9.csv', 'w') as f1, open(path_test+'9.csv', 'w') as f2:
        for line in f.readlines()[1:]:
            _, label, p1, p2 = line.strip().split(',')
            F1, MID1, P1 = p1.split('/')
            F2, MID2, P2 = p2.split('/')
            if 900 < int(F1[1:5]) <= 1000 and 900 < int(F2[1:5]) <= 1000:
                if label == '1' and cnt1 >= nums[i]:
                    continue
                f2.write(label + ',' + p1 + ',' + p2 + '\n')
                cnt1 += 1
            elif (int(F1[1:5]) <= 900 or int(F1[1:5]) > 1000) and (int(F2[1:5]) <= 900 or int(F2[1:5]) > 1000):
                f1.write(label + ',' + p1 + ',' + p2 + '\n')
#         lines = 0
#         list = []
#         for line in f.readlines()[1:]:
#             _, label, p1, p2 = line.strip().split(',')
#             list.append((label, p1, p2))
#             lines += 1
#         cnt = 1
#         base = 300
#         if i == 3 or i == 4 or i == 5 or i == 6:
#             base = 30
#         for i in range(10):
#             for j in range(base):
#                 label, p1, p2 = list[i * base + j]
#                 f2.write(label + ',' + p1 + ',' + p2 + '\n')
#             for j in range(base):
#                 label, p1, p2 = list[lines - i * base - j - 1]
#                 f2.write(label + ',' + p1 + ',' + p2 + '\n')
#
#         for i in range(base + 1, lines - base + 1):
#             label, p1, p2 = list[i]
#             f1.write(label + ',' + p1 + ',' + p2 + '\n')
#
#         # a = [[0 for i in range(50)] for j in range(1000)]
# #
# # with open(path_ori, 'r') as f, open(path_train, 'w') as f1:
# #     for line in f.readlines()[1:]:
# #         _, label, p1, p2 = line.strip().split(',')
# #         F, MID, P = p1.split('/')
# #         if a[int(F[1:4])][int(MID[3:])] % 15 == 0:
# #             a[int(F[1:4])][int(MID[3:])] += 1
# #             # f2.write(label + ',' + p1 + ',' + p2 + '\n')
# #             lines += 1
# #             list.append((label, p1, p2))
# #         else:
# #             a[int(F[1:4])][int(MID[3:])] += 1
# #             f1. write(label + ',' + p1 + ',' + p2 + '\n')
#
#     # with open(path_train, 'a') as f1, open(path_test, 'w') as f2:
#     #     cnt = 1
#     #     for i in range(1):
#     #         for j in range(300):
#     #             label, p1, p2 = list[i * 300 + j]
#     #             f2.write(label + ',' + p1 + ',' + p2 + '\n')
#     #         for j in range(300):
#     #             label, p1, p2 = list[lines - i * 300 - j - 1]
#     #             f2.write(label + ',' + p1 + ',' + p2 + '\n')
#     #
#     #     for i in range(301, lines - 300 + 1):
#     #         label, p1, p2 = list[i]
#     #         f1.write(label + ',' + p1 + ',' + p2 + '\n')
