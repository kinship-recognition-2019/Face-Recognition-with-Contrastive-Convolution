import random

# paths = ['./FIW_List/brother-brother/bb_test.csv', './FIW_List/father-daughter/fd_test.csv', './FIW_List/father-son/fs_test.csv',
#        './FIW_List/grandfather-granddaughter/gfgd_test.csv', './FIW_List/grandfather-grandson/gfgs_test.csv',
#        './FIW_List/grandmother-granddaughter/gmgd_test.csv', './FIW_List/grandmother-grandson/gmgs_test.csv',
#        './FIW_List/mother-daughter/md_test.csv', './FIW_List/mother-son/ms_test.csv',
#        './FIW_List/sibs/sibs_test.csv', './FIW_List/sister-sister/ss_test.csv']
# nums = [20000, 4500, 10000, 1200, 360, 1000, 320, 3600, 6000, 9000, 1500]

paths = ['./FIW_List/father-daughter/fd_test',
        './FIW_List/grandmother-grandson/gmgs_test',
        './FIW_List/sister-sister/ss_test']

nums = [4200, 140, 1000]
for k in range(0, 3):
    path = paths[k] + '0.csv'

# TASK 2
    lines = []
    cnt = 0
    with open(path, 'r') as f:
        for line in f.readlines():
            label, p1, p2 = line.strip().split(',')
            lines.append((label, p1, p2))
            cnt += 1

    with open(path, 'w') as f:
        base = int(nums[k] / 10)
        for i in range(10):
            # print(1, i*base, i*base+base-1)
            # print(0, nums[k] * 2 - i * base - base - 1, nums[k] * 2 - i * base - 1)
            for j in range(base):
                label, p1, p2 = lines[i * base + j]
                f.write(label + ',' + p1 + ',' + p2 + '\n')
            for j in range(base):
                label, p1, p2 = lines[nums[k] * 2 - i * base - j - 1]
                f.write(label + ',' + p1 + ',' + p2 + '\n')

# TASK 1
#     cnt1 = 0
#     cnt0 = 0
#     lines = []
#     with open(path, 'r') as f:
#         for line in f.readlines():
#             label, p1, p2 = line.strip().split(',')
#             if label == '1':
#                 cnt1 += 1
#             else:
#                 cnt0 += 1
#             lines += (p1, p2)
#         print(path.split('/')[-1:], '1=',cnt1, '0=',cnt0, 'tot=',cnt0+cnt1)
#
#         while cnt0 < cnt1:
#             line1 = random.choice(lines)
#             line2 = random.choice(lines)
#             while line1[0:5] == line2[0:5]:
#                 line2 = random.choice(lines)
#             # print(line1[0:5], line2[0:5])
#             with open(path, 'a') as f_:
#                 f_.write('0' + ',' + line1 + ',' + line2 + '\n')
#                 cnt0 += 1