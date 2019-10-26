import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CasiaFaceDataset(Dataset):
    def __init__(self, img_path, list_path, noofpairs=4, transform=None):
        super().__init__()
        self.transform = transform
        self.img_path = img_path
        self.list_path = list_path
        self.noofpairs = noofpairs
        self.imagelist = self.get_id_imagelist()
        self.noofcategories = len(self.imagelist) - 1
        self.train_list = self.create_pairs()

    def get_id_imagelist(self):
        f = open(self.list_path)
        lines = f.readlines()

        subjectdict = dict()
        for name in lines[:]:
            subject = name.split('/')[0]
            subjectdict.setdefault(subject, [])
            subjectdict[subject].append(name)
        f.close()

        imagelist = []
        for i, (key, value) in enumerate(subjectdict.items()):
            imagelist.append((i, key, value))
        return imagelist

    @staticmethod
    def get_random_two_images(tupleA, tupleB):
        classA = tupleA[0]
        classB = tupleB[0]
        listA = tupleA[2]
        listB = tupleB[2]
        imageA = np.random.choice(listA)
        imageB = np.random.choice(listB)

        while imageA == imageB:
            imageB = np.random.choice(listB)
        return "/".join(imageA.split("/")[-2:]), classA, "/".join(imageB.split("/")[-2:]), classB

    def create_pairs(self):
        pairsList = []
        for n in range(self.noofpairs):
            posCatList = np.arange(0, self.noofcategories)
            i = np.random.choice(posCatList)
            negativeCategoriesList = np.delete(np.arange(0, self.noofcategories), i)
            j = np.random.choice(negativeCategoriesList)

            imageA, c1, imageB, c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[i])
            pairsList.append([imageA, c1, imageB, c2, "1"])

            imageA, c1, imageB, c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[j])
            pairsList.append([imageA, c1, imageB, c2, "0"])

        return pairsList

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, i):
        image_name1 = self.train_list[i][0][:-1]
        image_name2 = self.train_list[i][2][:-1]
        id1 = self.train_list[i][1]
        id2 = self.train_list[i][3]
        label = self.train_list[i][4]

        path_img1 = os.path.join(self.img_path, image_name1)
        path_img2 = os.path.join(self.img_path, image_name2)

        if os.path.exists(path_img1) and os.path.exists(path_img2):
            img1 = Image.open(path_img1).convert('L')
            img2 = Image.open(path_img2).convert('L')

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, int(id1), int(id2), int(label)
