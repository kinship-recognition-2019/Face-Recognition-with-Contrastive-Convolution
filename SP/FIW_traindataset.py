from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np

# FIW亲属数据集的训练集处理
# 继承pytorch的Dataset类，迭代器返回两张人脸、两张人脸各自所属的家庭编号以及是否同一家族标签


class FIWTrainDataset(Dataset):
    def __init__(self, img_path, list_path, noofpairs=4, transform=None):
        super().__init__()
        self.transform = transform
        self.img_path = img_path
        self.list_path = list_path
        self.no_of_pairs = noofpairs
        self.image_list = self.get_image_list()
        self.train_list = self.create_pairs()

    def get_image_list(self):
        pair_list = []
        with open(self.list_path, 'r') as f:
            for line in f.readlines():
                label, p1, p2 = line.strip().split(',')
                path1 = os.path.join(self.img_path, p1)
                path2 = os.path.join(self.img_path, p2)
                id1 = int(p1[1:5]) - 1
                id2 = int(p2[1:5]) - 1
                label = int(label)
                pair_list.append((path1, path2, id1, id2, label))
        return pair_list

    def create_pairs(self):
        pairsList = []
        for n in range(self.no_of_pairs):
            CatList = np.arange(0, len(self.image_list))
            i = np.random.choice(CatList)
            while self.image_list[i][4] != 1:
                i = np.random.choice(CatList)
            j = np.random.choice(CatList)
            while self.image_list[j][4] != 0:
                j = np.random.choice(CatList)

            imageA, imageB, c1, c2, target= self.image_list[i]
            pairsList.append([imageA, imageB, c1, c2, target])
            # print(target)

            imageA, imageB, c1, c2, target = self.image_list[j]
            pairsList.append([imageA, imageB, c1, c2, target])
            # print(target)

        return pairsList

    def __getitem__(self, i):
        path_img1 = self.train_list[i][0]
        path_img2 = self.train_list[i][1]
        id1 = self.train_list[i][2]
        id2 = self.train_list[i][3]
        label = self.train_list[i][4]

        if os.path.exists(path_img1) and os.path.exists(path_img2):
            img1 = Image.open(path_img1).convert('L')
            img2 = Image.open(path_img2).convert('L')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, id1, id2, label

    def __len__(self):
        return len(self.train_list)
