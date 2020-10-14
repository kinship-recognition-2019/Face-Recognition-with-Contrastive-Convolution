from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import random

class FIWTrainTripleDataset(Dataset):
    def __init__(self, img_path, list_path, noofpairs, transform=None):
        super().__init__()
        self.transform = transform
        self.img_path = img_path
        self.list_path = list_path
        self.noofpairs = noofpairs
        self.train_list = self.get_triples()
    def get_triples(self):
        f=open(self.list_path)
        lines=f.readlines()
        idxs=np.arange(len(lines))
        triples=[]
        cnt=0
        while cnt < self.noofpairs:
            cnt += 1
            i = random.choice(idxs)
            img1,img2,img3=lines[i][:-1].split(",")
            path1 = os.path.join(self.img_path, img1)
            path2 = os.path.join(self.img_path, img2)
            path3 = os.path.join(self.img_path, img3)
            id1 = int(img1[1:5]) - 1
            id2 = int(img2[1:5]) - 1
            id3 = int(img3[1:5]) - 1
            triples.append((path1,path2,path3,id1,id2,id3))
        return triples

    def __getitem__(self, i):
        path_img1 = self.train_list[i][0]
        path_img2 = self.train_list[i][1]
        path_img3 = self.train_list[i][2]
        id1 = self.train_list[i][3]
        id2 = self.train_list[i][4]
        id3 = self.train_list[i][5]

        if os.path.exists(path_img1) and os.path.exists(path_img2):
            img1 = Image.open(path_img1).convert('L')
            img2 = Image.open(path_img2).convert('L')
            img3 = Image.open(path_img3).convert('L')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
                img3 = self.transform(img3)

            return img1, img2, img3

    def __len__(self):
        return len(self.train_list)
