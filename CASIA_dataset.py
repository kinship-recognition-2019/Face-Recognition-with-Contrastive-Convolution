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
        self.no_of_pairs = noofpairs
        self.image_list = self.get_image_list()  # (id, 'subject', ['subject/xxx.jpg', 'subject/yyy.jpg', ...])
        self.no_of_categories = len(self.image_list)
        self.train_list = self.create_pairs() # [noofpairs * ['pos/x.jpg', id1, 'pos/y.jpg', id1, '1'], ['pos/x.jpg', id1, 'neg/y.jpg', id2, '0']]

    def get_image_list(self):
        f = open(self.list_path)
        lines = f.readlines()

        subjectdict = dict()
        for name in lines[:]:
            subject = name.split('/')[0]
            subjectdict.setdefault(subject, [])  # if not exist
            subjectdict[subject].append(name)
        f.close()

        imagelist = []
        for i, (key, value) in enumerate(subjectdict.items()):
            imagelist.append((i, key, value))

        return imagelist

    def get_random_two_images(self, tupleA, tupleB):
        classA = tupleA[0]
        classB = tupleB[0]
        listA = tupleA[2]
        listB = tupleB[2]
        imageA = np.random.choice(listA)
        imageB = np.random.choice(listB)
        while imageA == imageB:
            imageB = np.random.choice(listB)

        # return imageA, classA, imageB, classB
        return '/'.join(imageA.split("/")[-2:]), classA, '/'.join(imageB.split("/")[-2:]), classB

    def create_pairs(self):
        pairsList = []
        for n in range(self.no_of_pairs):
            posCatList = np.arange(0, self.no_of_categories)
            i = np.random.choice(posCatList)
            negCatList = np.delete(np.arange(0, self.no_of_categories), i)

            imageA, c1, imageB, c2 = self.get_random_two_images(self.image_list[i], self.image_list[i])
            pairsList.append([imageA, c1, imageB, c2, "1"])

            j = np.random.choice(negCatList)
            imageA, c1, imageB, c2 = self.get_random_two_images(self.image_list[i], self.image_list[j])
            pairsList.append([imageA, c1, imageB, c2, "0"])

        return pairsList

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, i):
        image_name1 = self.train_list[i][0][:-1]  # [:-1] -> without '\n'
        image_name2 = self.train_list[i][2][:-1]
        id1 = self.train_list[i][1]
        id2 = self.train_list[i][3]
        label = self.train_list[i][4]

        path_img1 = os.path.join(self.img_path, image_name1)
        path_img2 = os.path.join(self.img_path, image_name2)

        if os.path.exists(path_img1) and os.path.exists(path_img2):
            img1 = Image.open(path_img1).convert('L')  # change into gray
            img2 = Image.open(path_img2).convert('L')
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, int(id1), int(id2), int(label)



if __name__ == '__main__':
    img_path = 'C:/Users/VineAsh/Desktop/Face-Recognition-with-Contrastive-Convolution/dataset/CASIA-WebFace'
    list_path = 'C:/Users/VineAsh/Desktop/Face-Recognition-with-Contrastive-Convolution/dataset/casialist.txt'
    C = CasiaFaceDataset(img_path, list_path)
    list = C.create_pairs()
    f = open('out.txt', 'w')
    f.write(str(list)+'\n')
    f.write(str(list[0])+'\n')
    f.write(str(list[0][0][:-1]))
    f.write(str(C.no_of_categories))