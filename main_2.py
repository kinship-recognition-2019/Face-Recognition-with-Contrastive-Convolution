import argparse
import torchvision

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from siamese_network import SiameseNetwork
from FIW_traindataset import FIWTrainDataset
from FIW_testdataset import FIWTestDataset
from torchvision import transforms
from eval_metrics import evaluate


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        # print(euclidean_distance)
        # print(label)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kinship-Recognition-with-Contrastive-Convolution')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--fiw-img-path', type=str, default='./dataset/FIDs_NEW', help='path to fiw')
    parser.add_argument('--fiw-train-list-path', type=str, default='./dataset/fs.csv', help='path to fiw train list')
    parser.add_argument('--fiw-test-list-path', type=str, default='./dataset/fs_test.csv', help='path to fiw test list')
    parser.add_argument('--iters', type=int, default=200000, metavar='N', help='number of iterations to train')
    args = parser.parse_args()

    net = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), 0.001, betas=(0.9, 0.99))
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()])

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(128),
        transforms.ToTensor()])
    test_dataset = FIWTestDataset(img_path=args.fiw_img_path, pairs_path=args.fiw_test_list_path, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    # criterion = nn.BCELoss()

    for epoch in range(0, args.iters):
        train_dataset = FIWTrainDataset(img_path=args.fiw_img_path, list_path=args.fiw_train_list_path, noofpairs=args.batch_size, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        for batch_idx, (img_1, img_2, label) in enumerate(train_loader):
            # concatenated = torch.cat((img_1, img_2), 0)
            # imshow(torchvision.utils.make_grid(concatenated))

            optimizer.zero_grad()
            output1, output2 = net(img_1, img_2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            print('Iteration' + str(epoch), 'Batch' + str(batch_idx), "loss=", loss.item())
        if epoch % 10 == 0:
            with torch.no_grad():
                labels, distances = [], []
                for batch_idx, (data_a, data_b, label) in enumerate(test_loader):
                    output1, output2 = net(data_a, data_b)
                    euclidean_distance = F.pairwise_distance(output1, output2)

                    distances.append(euclidean_distance.numpy())
                    labels.append(label.numpy())
                labels = np.array([sublabel for label in labels for sublabel in label])
                distances = np.array([subdist for dist in distances for subdist in dist])
                print(distances)
                print(labels)
                acc = evaluate(distances, labels)
                print(acc)
                print("acc", str(np.mean(acc)))

