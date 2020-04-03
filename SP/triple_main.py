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
from eval_metrics import evaluate
from regressor import Regressor
from identity_regressor import Identity_Regressor
from FIW_traintripledataset import FIWTrainTripleDataset
#from reg_kinship import RegressorKinship
import matplotlib.pyplot as plt
from FIW_testdataset import FIWTestDataset
from tqdm import tqdm
from triple_loss import Triple_loss

# 运行main，用于原论文 - 两张人脸是否属于同一个人问题


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 训练函数
def train(args, basemodel, reg_model,idreg_model, genmodel, device, train_loader, optimizer, criterion, criterion1, iteration):
    # reg_model_kinship.train()
    # basemodel.eval()
    # genmodel.eval()
    # reg_model.eval()
    # idreg_model.eval()
    basemodel.train()
    genmodel.train()
    idreg_model.train()

    for batch_idx, (data_1, data_2,data_3, c1, c2, c3) in enumerate(train_loader):
        data_1, data_2,data_3, c1, c2, c3, = (data_1).to(device), (data_2).to(device), (data_3).to(device),torch.from_numpy(np.asarray(c1)).to(
            device), torch.from_numpy(np.asarray(c2)).to(device), torch.from_numpy(np.asarray(c2)).to(device)
        concatenated = torch.cat((data_1, data_2), 0)

        optimizer.zero_grad()

        Ab_list, B_list, Ac_list,C_list, org_kernel_1,org_kernel_2,org_kernel_3= compute_triple_contrastive_features(data_1, data_2,data_3,basemodel, genmodel,device)

        reg_Ab = reg_model(Ab_list)
        reg_B = reg_model(B_list)
        SAB = (reg_Ab + reg_B) / 2.0
        #print("SAB",SAB.sum())
        reg_Ac = reg_model(Ac_list)
        reg_C = reg_model(C_list)
        SAC = (reg_Ac + reg_C) / 2.0
        #print("SAC",SAC.sum())
        loss1 = criterion1(SAC, SAB)

        #hk1 = idreg_model(org_kernel_1)
        #hk2 = idreg_model(org_kernel_2)
        #hk3 = idreg_model(org_kernel_3)

        #loss2 =  (criterion(hk1, c1) + criterion(hk2, c2)+criterion(hk3, c3))/3
        loss2 = criterion2(org_kernel_1,org_kernel_2,org_kernel_3)
        # loss = loss2 + loss1 + loss3
        # print(reg_model_kinship.state_dict())
        #loss = loss1
        loss = loss1 + loss2
        loss.backward()

        optimizer.step()

        # print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
        #     iteration, batch_idx * len(data_1), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
        #     loss.item()))
        print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} {:.4f} {:.4f}'.format(
            iteration, batch_idx * len(data_1), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            loss.item(), loss1.item(), loss2.item()))

# 测试函数
def ttest(test_loader, basemodel, genmodel, epoch, device, args):
    basemodel.eval()
    genmodel.eval()

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
            SAB = F.pairwise_distance(out1_a,out1_b,p=2)
            #SAB = torch.squeeze(SAB, 1)

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

# 保存断点
def save_checkpoint(state, filename):
    torch.save(state, filename)

# 利用第一层和第二层网络进行人脸特征对比
# 输出通过特定生成的kernel而卷积产生的人脸数据和处理过的kernel
# 输出作为regressor和identity regressor的输入
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

def compute_triple_contrastive_features(data_1, data_2,data_3, basemodel, genmodel, device):
    data_1, data_2 = (data_1).to(device), (data_2).to(device)

    data_1 = basemodel(data_1)
    data_2 = basemodel(data_2)
    data_3 = basemodel(data_3)

    kernel_1 = genmodel(data_1).to(device)
    kernel_2 = genmodel(data_2).to(device)
    kernel_3 = genmodel(data_3).to(device)

    norm_kernel1 = torch.norm(kernel_1, 2, 2)
    norm_kernel2 = torch.norm(kernel_2, 2, 2)
    norm_kernel3 =torch.norm(kernel_3,2,2)

    norm_kernel1_1 = torch.unsqueeze(norm_kernel1, 2)
    norm_kernel2_2 = torch.unsqueeze(norm_kernel2, 2)
    norm_kernel3_1 = torch.unsqueeze(norm_kernel3, 2)

    kernel_1 = kernel_1 / norm_kernel1_1
    kernel_2 = kernel_2 / norm_kernel2_2
    kernel_3 = kernel_3 / norm_kernel3_1

    F1, F2, F3= data_1, data_2, data_3

    Kab = torch.abs(kernel_1 - kernel_2)
    Kac = torch.abs(kernel_1 - kernel_3)

    bs, featuresdim, h, w = F1.size()
    F1 = F1.view(1, bs * featuresdim, h, w)
    F2 = F2.view(1, bs * featuresdim, h, w)
    F3 = F3.view(1, bs * featuresdim, h, w)

    noofkernels = 14
    kernelsize = 3

    Tab = Kab.view(noofkernels * bs, -1, kernelsize, kernelsize)
    Tac = Kab.view(noofkernels * bs, -1, kernelsize, kernelsize)

    F1_Tab_out = F.conv2d(F1, Tab, stride=1, padding=2, groups=bs)
    F2_Tab_out = F.conv2d(F2, Tab, stride=1, padding=2, groups=bs)
    F1_Tac_out = F.conv2d(F1, Tac, stride=1, padding=2, groups=bs)
    F3_Tac_out = F.conv2d(F3, Tac, stride=1, padding=2, groups=bs)

    Ab_list = F1_Tab_out.view(bs, -1)
    B_list = F2_Tab_out.view(bs, -1)
    Ac_list = F1_Tac_out.view(bs, -1)
    C_list = F3_Tac_out.view(bs, -1)

    return Ab_list, B_list,Ac_list,C_list,kernel_1,kernel_2,kernel_3

# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        if epoch > 100000 and epoch < 140000:
            print('Learning rate is 0.01')
            param_group['lr'] = 0.01
        elif epoch > 140000:
            print('Learning rate is 0.001')
            param_group['lr'] = 0.001

def main():
    # 参数
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Convolution for FR')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='BST', help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=200000, metavar='N', help='number of iterations to train (default: 10)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--pretrained', default=False, type=bool, metavar='N', help='use pretrained ligthcnn model:True / False no pretrainedmodel )')
    parser.add_argument('--save_path', default='', type=str, metavar='PATH', help='path to save checkpoint (default: none)')
    # parser.add_argument('--resume', default='model200000_checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--compute_contrastive', default=True, type=bool,
                        metavar='N', help='use contrastive featurs or base mode features: True / False )')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('--lfw-dir', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/lfw',
                        help='path to dataset')
    parser.add_argument('--lfw_pairs_path', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/pairs.txt',
                        help='path to pairs file')
    parser.add_argument('--root_path', default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/CASIA-WebFace', type=str, metavar='PATH',
                        help='path to root path of images (default: none)')
    # parser.add_argument('--num_classes', default=10574, type=int,
    #                    metavar='N', help='number of classes (default: 10574)')
    parser.add_argument('--num_classes', default=1000, type=int,
                        metavar='N', help='number of classes (default: 10574)')

    parser.add_argument('--fiw-train-list-path', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/fs_train_triples.csv',
                        help='path to fiw train list')
    parser.add_argument('--fiw-test-list-path', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/fs_test.csv',
                        help='path to fiw test list')
    parser.add_argument('--fiw-img-path', type=str, default='/media/zhang/新加卷/pycode/Face-Recognition-with-Contrastive-Convolution-pytorch/dataset/FIDs_NEW', help='path to fiw')
    args = parser.parse_args()

    # cuda设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 测试集处理，采用LFW人脸测试集
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(128),
        transforms.ToTensor()])

    test_dataset = FIWTestDataset(img_path=args.fiw_img_path, pairs_path=args.fiw_test_list_path,
                                  transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    # 训练集的transform函数
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), ])

    basemodel = Contrastive_4Layers(num_classes=args.num_classes).to(device)
    genmodel = GenModel(512).to(device)
    reg_model = Regressor(686).to(device)
    idreg_model = Identity_Regressor(14 * 512 * 3 * 3, args.num_classes).to(device)

    params = list(basemodel.parameters()) + list(genmodel.parameters())  + list(idreg_model.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['iterno']
            genmodel.load_state_dict(checkpoint['state_dict1'])
            basemodel.load_state_dict(checkpoint['state_dict2'])
            reg_model.load_state_dict(checkpoint['state_dict3'])
            idreg_model.load_state_dict(checkpoint['state_dict4'])
            # reg_model_kinship.load_state_dict(checkpoint['state_dict3'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # print("=> loaded checkpoint '{}' (epoch {})"
            #       .format(args.resume, checkpoint['iterno']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    # 损失函数
    criterion2 = nn.CrossEntropyLoss().to(device)
    criterion1 = nn.TripletMarginLoss(margin=0.5, p=2).to(device)
    #criterion1 = Triple_loss(0.5).to(device)

    print('Device being used is :' + str(device))

    for iterno in range(args.start_epoch + 1, args.iters + 1):
        adjust_learning_rate(optimizer, iterno)
        train_dataset = FIWTrainTripleDataset(img_path=args.fiw_img_path, list_path=args.fiw_train_list_path,
                                        noofpairs=args.batch_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # 训练
        train(args, basemodel, reg_model, idreg_model, genmodel, device, train_loader,optimizer, criterion2, criterion1, iterno)

        if iterno % 1 == 0:
            # 每100轮训练进行一次测试
            testacc = ttest(test_loader, basemodel, genmodel,iterno, device, args)
            f = open('LFW_performance.txt', 'a')
            f.write('\n' + str(iterno) + ': ' + str(testacc * 100))
            f.close()
            print('Test accuracy: {:.4f}'.format(testacc * 100))

        if iterno % 1 == 0:
             save_name = args.save_path + 'model_kinship' + str(iterno) + '_checkpoint.pth.tar'
             save_checkpoint(
                 {'iterno': iterno,
                  'state_dict1': genmodel.state_dict(), 'state_dict2': basemodel.state_dict(),
                  'state_dict3': idreg_model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  }, save_name)


if __name__ == '__main__':
    main()
