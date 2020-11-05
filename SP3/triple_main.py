#coding:utf-8
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
from regressor import Regressor
from eval_metrics import evaluate
from identity_regressor import Identity_Regressor
from FIW_traindataset import FIWTrainDataset
#from reg_kinship import RegressorKinship
import matplotlib.pyplot as plt
from FIW_traintripledataset import FIWTrainTripleDataset
from FIW_testdataset import FIWTestDataset
from tqdm import tqdm
from triple_loss import Triple_loss
from get_triples_list import get_csv
from base_model import Contrastive_4Layers
from contrastive_cnn import Contrastive_14Layers,Contrastive_34Layers
# 运行main，用于原论文 - 两张人脸是否属于同一个人问题


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def get_alpha(ratios):
	#f=open("result.txt","w")
	#for ratio in ratios:
	#	f.write(str(ratio.item())+"\n")
	alpha=torch.mean(torch.tensor(ratios).to("cuda"))
	#f.write("ave"+str(alpha.item())+"\n")
	print("adjust alpha to :",alpha.item())
	return alpha

def train(args, basemodel, idreg_model, genmodel, reg_model,  device, train_loader, optimizer, criterion, criterion1, iteration,alpha,ratios):

    basemodel.train()
    genmodel.train()
    reg_model.train()
    idreg_model.train()

    for batch_idx, (data_1, data_2, data_3,target1,target2) in enumerate(train_loader):
        data_1, data_2, data_3 = (data_1).to(device), (data_2).to(device),  (data_3).to(device)

        optimizer.zero_grad()
        target1 = target1.float().unsqueeze(1).to(device)
        target2 = target2.float().unsqueeze(1).to(device)
        
        A_list, B_list, org_kernel_1, org_kernel_2 = compute_contrastive_features(data_1, data_2, basemodel, genmodel,
                                                                                  device)
        reg_1 = reg_model(A_list)
        reg_2 = reg_model(B_list)
        SAB = (reg_1 + reg_2) / 2.0
        #print("SAB",SAB.size())
        bs,length =SAB.size()
        loss1 = criterion1(SAB, torch.ones(bs,length).to(device))

        bs,m,n=org_kernel_1.size()
        feature_vector1 = org_kernel_1.view(-1,m*n)
        feature_vector2 = org_kernel_2.view(-1,m*n)
        pos_dist = F.pairwise_distance(feature_vector1,feature_vector2,1)

        C_list, D_list, org_kernel_3, org_kernel_4 = compute_contrastive_features(data_1, data_3, basemodel, genmodel,
                                                                                  device)
        reg_3 = reg_model(C_list)
        reg_4 = reg_model(D_list)
        SCD = (reg_3 + reg_4) / 2.0
        #loss1 = criterion(SCD,SAB)
        loss1 = (criterion1(SAB,target1)+criterion1(SCD, target2))/2

        feature_vector3 = org_kernel_3.view(-1,m*n)
        feature_vector4 = org_kernel_4.view(-1,m*n)
        neg_dist = F.pairwise_distance(feature_vector3,feature_vector4,1)

        loss2 = criterion(pos_dist,neg_dist)
	
        ratios.append(torch.clamp(loss1/(loss2+1e-6),0,0.01))
        #print(ratios[0].item())
        loss = loss1 + alpha*loss2
        loss.backward()

        optimizer.step()

        print('Train iter: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f} {:.4f} {:.4f}'.format(
            iteration, batch_idx * len(data_1), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
            loss.item(), loss1.item(), loss2.item()))
    return ratios
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

# 调整学习率
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        if epoch >= 50000  and epoch <100000:
            print('Learning rate is 0.001')
            param_group['lr'] = 0.001
        elif epoch >= 100000:
            print('Learning rate is 3e-4')
            param_group['lr'] = 3e-4

def main():
    # 参数
    parser = argparse.ArgumentParser(description='PyTorch Contrastive Convolution for FR')
    parser.add_argument('--basemodel',type=str,default="Contrastive_34Layers",help="basemodel (Contrastive_4Layers/seresnet50/resnet50/inception_resnet_v2")
    parser.add_argument('--batch_size', type=int, default=44, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=128, metavar='BST', help='input batch size for testing (default: 1000)')
    parser.add_argument('--iters', type=int, default=20000, metavar='N', help='number of iterations to train (default: 10)')
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

    parser.add_argument('--lfw-dir', type=str, default='../dataset/lfw',
                        help='path to dataset')
    parser.add_argument('--lfw_pairs_path', type=str, default='../dataset/pairs.txt',
                        help='path to pairs file')
    parser.add_argument('--root_path', default='../dataset/CASIA-WebFace', type=str, metavar='PATH',
                        help='path to root path of images (default: none)')
    # parser.add_argument('--num_classes', default=10574, type=int,
    #                    metavar='N', help='number of classes (default: 10574)')
    parser.add_argument('--num_classes', default=1000, type=int,
                        metavar='N', help='number of classes (default: 10574)')
    parser.add_argument('--fiw-train-list-path', type=str, default='../dataset/FIW_List/sister-sister/ss_train.csv',
                        help='path to fiw train list')
    parser.add_argument('--fiw-train-triples-list-path', type=str, default='../dataset/Triple-list/ss_train_triples.csv',
                        help='path to fiw train list')
    parser.add_argument('--fiw-test-list-path', type=str, default='../dataset/FIW_List/sister-sister/ss_test.csv',
                        help='path to fiw test list')
    parser.add_argument('--fiw-img-path', type=str, default='../dataset/FIDs_NEW', help='path to fiw')
    parser.add_argument('--alpha',default=0.0071,type=float,help="the weight of loss2")
    args = parser.parse_args()

    # cuda设置
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # 测试集处理，采用LFW人脸测试集
    test_transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=1),
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
    if(args.basemodel == "Contrastive_34Layers"):
        print("Contrastive_34Layers")
        basemodel = Contrastive_34Layers(num_classes=args.num_classes)
    basemodel = basemodel.to(device)

    genmodel = GenModel(512).to(device)
    reg_model = Regressor(686).to(device)
    idreg_model = Identity_Regressor(14 * 512 * 3 * 3, args.num_classes).to(device)

    params = list(basemodel.parameters()) + list(genmodel.parameters()) + list(reg_model.parameters()) + list(idreg_model.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            genmodel.load_state_dict(checkpoint['state_dict1'])
            basemodel.load_state_dict(checkpoint['state_dict2'])
            reg_model.load_state_dict(checkpoint['state_dict3'])
            idreg_model.load_state_dict(checkpoint['state_dict4'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    criterion2 = Triple_loss(256.0)
    criterion1 = nn.BCELoss().to(device)
    ratios = []

    print('Device being used is :' + str(device))

    for iterno in range(args.start_epoch + 1, args.iters + 1):
        #adjust_learning_rate(optimizer, iterno)
        train_dataset = FIWTrainTripleDataset(img_path=args.fiw_img_path, list_path=args.fiw_train_triples_list_path,noofpairs=1000*args.batch_size, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

        # 训练
        ratios=train(args, basemodel, idreg_model, genmodel, reg_model,  device, train_loader,optimizer, criterion2, criterion1, iterno,args.alpha,ratios)

        if iterno % 1 == 0:
            # 每100轮训练进行一次测试
            testacc = ttest(test_loader, basemodel, genmodel, reg_model, iterno, device, args)
            f = open('ss1/LFW_performance.txt', 'a')
            f.write('\n' + str(iterno) + ': ' + str(testacc * 100))
            f.close()
            print('Test accuracy: {:.4f}'.format(testacc * 100))

            save_name = args.save_path + 'model_kinship' + str(iterno) + '_checkpoint.pth.tar'
            save_checkpoint(
                 {'iterno': iterno,
                  'state_dict1': genmodel.state_dict(), 'state_dict2': basemodel.state_dict(),
                  'state_dict3': reg_model.state_dict(), 'state_dict4': idreg_model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  }, save_name)
            print("Regenerate sample pair triples......")
            get_csv(args.fiw_train_list_path,args.fiw_train_triples_list_path,args.fiw_img_path)
            
            args.alpha=get_alpha(ratios)
            ratios = []

if __name__ == '__main__':
    main()
