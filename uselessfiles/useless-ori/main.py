import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from CASIA_dataset import CasiaFaceDataset
from LFW_dataset import LFWDataset
from FIW_traindataset import FIWTrainDataset
from FIW_testdataset import FIWTestDataset

from base_model import Network4Layers
from gen_model import GenModel
from reg_model import Regressor
from idreg_model import Identity_Regressor
from eval_metrics import evaluate

import numpy as np
import argparse


def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        if 100000 < epoch < 140000:
            print('Learning rate is 0.01')
            param_group['lr'] = 0.01
        elif epoch > 140000:
            print('Learning rate is 0.001')
            param_group['lr'] = 0.001


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
    bs, featuredim, h, w = F1.size()

    F1 = F1.view(1, bs * featuredim, h, w)
    F2 = F2.view(1, bs * featuredim, h, w)

    noofkernerls = 14
    kernelsize = 3

    T = Kab.view(noofkernerls * bs, -1, kernelsize, kernelsize)

    F1_T_out = F.conv2d(input=F1, weight=T, stride=1, padding=2, groups=bs)
    F2_T_out = F.conv2d(input=F2, weight=T, stride=1, padding=2, groups=bs)

    A_list = F1_T_out.view(bs, -1)
    B_list = F2_T_out.view(bs, -1)

    return A_list, B_list, kernel_1, kernel_2


def train(base_model, gen_model, reg_model, idreg_model, device, train_loader, optimizer, criterion1, criterion2, iter):
    gen_model.train()
    reg_model.train()
    idreg_model.train()

    for batch_idx, (data_1, data_2, c1, c2, target) in enumerate(train_loader):
        data_1, data_2 = (data_1).to(device), (data_2).to(device)
        c1, c2 = torch.from_numpy(np.asarray(c1)).to(device), torch.from_numpy(np.asarray(c2)).to(device)
        target = torch.from_numpy(np.asarray(target)).to(device)
        print(c1.shape)
        print(target.shape)

        target = target.float().unsqueeze(1)
        optimizer.zero_grad()

        A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(data_1, data_2, base_model, gen_model, device)
        reg_1 = reg_model(A_list)
        reg_2 = reg_model(B_list)
        print(reg_1.shape)
        SAB = (reg_1 + reg_2) / 2.0

        loss1 = criterion1(SAB, target)

        hk1 = idreg_model(org_kernel_1)
        hk2 = idreg_model(org_kernel_2)
        print(hk1.shape)
        loss2 = 0.5 * (criterion2(hk1, c1) + criterion2(hk2, c2))

        loss = loss1 + loss2

        loss.backward()
        optimizer.step()
        f = open('result.txt', 'a')
        print('Iteration' + str(iter), 'Batch' + str(batch_idx), "loss=", loss.item(), "loss1=", loss1.item(), "loss2=", loss2.item())
        f.write('Iteration' + str(iter) + ' Batch' + str(batch_idx) + " loss=" + str(loss.item()) + " loss1=" + str(loss1.item()) + " loss2=" + str(loss2.item()) + '\n')
        f.close()


def test(test_loader, base_model, gen_model, reg_model, device):
    base_model.eval()

    labels, distance, distances = [], [], []

    with torch.no_grad():
        for batch_idx, (data_a, data_b, label) in enumerate(test_loader):
            data_a, data_b = data_a.to(device), data_b.to(device)

            out1_a, out1_b, k1, k2 = compute_contrastive_features(data_a, data_b, base_model, gen_model, device)
            SA = reg_model(out1_a)
            SB = reg_model(out1_b)
            SAB = (SA + SB) / 2.0
            SAB = torch.squeeze(SAB, 1)
            print(SAB.shape)
            print(label.shape)

            distances.append(SAB.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])

        accuracy = evaluate(1 - distances, labels)
        return np.mean(accuracy)


def main():
    parser = argparse.ArgumentParser(description='Kinship-Recognition-with-Contrastive-Convolution')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--fiw-img-path', type=str, default='./dataset/FIDs_NEW', help='path to fiw')
    parser.add_argument('--fiw-train-list-path', type=str, default='./dataset/train_list.csv', help='path to fiw train list')
    parser.add_argument('--fiw-test-list-path', type=str, default='./dataset/test_list.csv', help='path to fiw test list')
    parser.add_argument('--casia-img-path', type=str, default='./dataset/CASIA-WebFace', help='path to casia')
    parser.add_argument('--casia-list-path', type=str, default='./dataset/casialist.txt', help='path to casia list')
    parser.add_argument('--lfw-img-path', type=str, default='./dataset/lfw', help='path to lfw')
    parser.add_argument('--lfw-list-path', type=str, default='./dataset/pairs.txt', help='path to lfw list')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='BST', help='input batch size for testing')
    parser.add_argument('--num-classes', default=10575, type=int, metavar='N', help='number of classes')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number')
    parser.add_argument('--iters', type=int, default=200000, metavar='N', help='number of iterations to train')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(128),
            transforms.ToTensor()])
    test_dataset = FIWTestDataset(img_path=args.fiw_img_path, pairs_path=args.fiw_test_list_path, transform=test_transform)
    # test_dataset = LFWDataset(img_path=args.lfw_img_path, pairs_path=args.lfw_list_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    base_model = Network4Layers().to(device)
    gen_model = GenModel(512).to(device)
    reg_model = Regressor(686).to(device)
    idreg_model = Identity_Regressor(14*512*3*3, args.num_classes).to(device)

    params = list(base_model.parameters()) + list(gen_model.parameters()) + list(reg_model.parameters()) + list(idreg_model.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

    criterion1 = nn.BCELoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)

    for iter in range(args.start_epoch + 1, args.iters + 1):
        adjust_learning_rate(optimizer, iter)
        train_dataset = FIWTrainDataset(img_path=args.fiw_img_path, list_path=args.fiw_train_list_path,
                                         noofpairs=args.batch_size, transform=train_transform)
        # train_dataset = CasiaFaceDataset(img_path=args.casia_img_path, list_path=args.casia_list_path,
        #                                 noofpairs=args.batch_size, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        train(base_model, gen_model, reg_model, idreg_model, device, train_loader, optimizer, criterion1, criterion2, iter)
        if iter > 0 and iter % 1 == 0:
            testacc = test(test_loader, base_model, gen_model, reg_model, device)
            f = open('result.txt', 'a')
            print("testacc:" + str(testacc))
            f.write("testacc:" + str(testacc) + '\n')
            f.close()


if __name__ == '__main__':
    main()





