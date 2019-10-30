from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np


class FIWTrainDataset(Dataset):
    def __init__(self, img_path, pairs_path, transform, noofpairs=4):
        super().__init__()
        self.img_path = img_path
        self.pairs_path = pairs_path
        self.transform = transform
        self.noofpairs = noofpairs
        self.image_list = self.get_pairs()
        self.noofcategories = len(self.image_list) - 1
        self.trainset = self.create_pairs()

    def get_pairs(self):
        pair_list = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines():
                label, p1, p2 = line.strip().split(',')
                path1 = os.path.join(self.img_path, p1)
                path2 = os.path.join(self.img_path, p2)
                id1 = int(p1[1:5])
                id2 = int(p2[1:5])
                label = int(label)
                pair_list.append((path1, path2, id1, id2, label))
        return pair_list

    def create_pairs(self):
        pairsList = []
        for n in range(self.noofpairs):
            posCatList = np.arange(0, self.noofcategories)
            i = np.random.choice(posCatList)
            negativeCategoriesList = np.delete(np.arange(0, self.noofcategories), i)
            j = np.random.choice(negativeCategoriesList)

            imageA, c1, imageB, c2, target= self.image_list[i]
            pairsList.append([imageA, imageB, c1, c2, target])

            imageA, c1, imageB, c2, target = self.image_list[j]
            pairsList.append([imageA, imageB, c1, c2, target])

        return pairsList

    def __getitem__(self, i):
        path_img1 = self.trainset[i][0]
        path_img2 = self.trainset[i][2]
        id1 = self.trainset[i][1]
        id2 = self.trainset[i][3]
        label = self.trainset[i][4]

        if os.path.exists(path_img1) and os.path.exists(path_img2):
            img1 = Image.open(path_img1).convert('L')
            img2 = Image.open(path_img2).convert('L')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, int(id1), int(id2), int(label)

    def __len__(self):
        return len(self.trainset)