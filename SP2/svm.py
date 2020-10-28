from __future__ import print_function
import argparse
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os
import numpy as np
from gen_model import GenModel
from base_model import Contrastive_4Layers
from FIW_traindataset import FIWTrainDataset
import matplotlib.pyplot as plt
from FIW_testdataset import FIWTestDataset
from tqdm import tqdm
from get_triples_list import get_csv
from densenet import densenet121
# 运行main，用于原论文 - 两张人脸是否属于同一个人问题

# 测试函数
def ttest(test_loader, basemodel, genmodel, reg_model, epoch, device, args):
    basemodel.eval()
    genmodel.eval()
    reg_model.eval()

    labels, distance, distances = [], [], []

    pbar = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_idx, (data_a, data_b, label) in pbar:
            data_a, data_b = data_a.to(device), data_b.to(device)
            # concatenated = torch.cat((data_a, data_b), 0)
            # imshow(torchvision.utils.make_grid(data_a))
            # imshow(torchvision.utils.make_grid(data_b))

            if args.compute_contrastive:
                out1_a, out1_b, k1, k2 = compute_contrastive_features(data_a, data_b, basemodel, genmodel, device)

                SA = reg_model(out1_a)
                SB = reg_model(out1_b)
                SAB = (SA + SB) / 2.0
            SAB = torch.squeeze(SAB, 1)

            distances.append(SAB.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

            if batch_idx % args.log_interval == 0:
                pbar.set_description('Epoch: {} [{}/{} ({:.0f}%)]'.format(
                    epoch, batch_idx * len(data_a), len(test_loader.dataset),
                           100. * batch_idx / len(test_loader)))

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        accuracy = evaluate(1 - distances, labels)
        print("acc", accuracy)

        return np.mean(accuracy)

def compute_contrastive_features(data_1, data_2, basemodel, genmodel, device):
    data_1, data_2 = (data_1).to(device), (data_2).to(device)

    data_1 = basemodel(data_1)
    data_2 = basemodel(data_2)

    kernel_1 = genmodel(data_1).to(device)
    kernel_2 = genmodel(data_2).to(device)

    norm_kernel1 = torch.norm(kernel_1, 2, 2)
    norm_kernel2 = torch.norm(kernel_2, 2, 2)
    norm_kernel1_1 = torch.unsqueeze(norm_kernel1, 2)
    norm_kernel2_2 = torch.unsqueeze(norm_kernel2, 2)
    kernel_1 = kernel_1 / norm_kernel1_1
    kernel_2 = kernel_2 / norm_kernel2_2

    F1, F2 = data_1, data_2

    Kab = torch.abs(kernel_1 - kernel_2)

    bs, featuresdim, h, w = F1.size()
    F1 = F1.view(1, bs * featuresdim, h, w)
    F2 = F2.view(1, bs * featuresdim, h, w)
    noofkernels = 14
    kernelsize = 3
    T = Kab.view(noofkernels * bs, -1, kernelsize, kernelsize)

    F1_T_out = F.conv2d(F1, T, stride=1, padding=2, groups=bs)
    F2_T_out = F.conv2d(F2, T, stride=1, padding=2, groups=bs)

    A_list = F1_T_out.view(bs, -1)
    B_list = F2_T_out.view(bs, -1)

    return A_list, B_list, kernel_1, kernel_2


def main():
    # 参数
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Convolution for FR')
    parser.add_argument('--basemodel',type=str,default="4-layers",help="basemodel")
    parser.add_argument("--feature extraction_model",type=str,default="",help="feature extraction model file")
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--fiw-train-list-path', type=str, default='/media/zhang/新加卷/pycode/Contrastive-Convolution-new/dataset/FIW_List/father-daughter/fd_train.csv',
                        help='path to fiw train list')
    parser.add_argument('--fiw-test-list-path', type=str, default='/media/zhang/新加卷/pycode/Contrastive-Convolution-new/dataset/FIW_List/father-daughter/fd_test.csv',
                        help='path to fiw test list')
    parser.add_argument('--fiw-train-triples-list-path', type=str, default='/media/zhang/新加卷/pycode/Contrastive-Convolution-new/dataset/ss_train_triples.csv',
                        help='path to fiw train list')
    parser.add_argument('--fiw-img-path', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/FIDs_NEW', help='path to fiw')
    args = parser.parse_args()

    # cuda设置
    device = torch.device("cuda")
    # 训练集的transform函数
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), ])

    genmodel = GenModel(512).to(device)

    if args.basemodel=="4-layers":
        basemodel=Contrastive_4Layers()
        print("4-layers")

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['iterno']
            genmodel.load_state_dict(checkpoint['state_dict1'])
            basemodel.load_state_dict(checkpoint['state_dict2'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    train_dataset = FIWTrainDataset(img_path=args.fiw_img_path, list_path=args.fiw_train_list_path,noofpairs=args.batch_size, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataset = FIWTestDataset(img_path=args.fiw_img_path,list_path=args.fiw_test_list_path,noofpairs=args.batch_size,transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = args.batch_size,shuffle = True)

if __name__ == '__main__':
    main()
