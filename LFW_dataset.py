import numpy as np
import os
import tensorflow as tf
from PIL import Image


class LFWDataset():
    def __init__(self):
        self.pairs_path = "dataset/pairs.txt"
        self.img_path = "dataset/lfw"
        self.cur = 0
        self.testlist = self.get_lfw_paths(self.img_path)

    def read_lfw_pairs(self):
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, file_ext="jpg"):
        pairs = self.read_lfw_pairs()

        path_list = []
        issame_list = []

        for i in range(len(pairs)):
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = 1
            elif len(pair) == 4:
                path0 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(self.img_path, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = 0
            if os.path.exists(path0) and os.path.exists(path1):
                    path_list.append((path0, path1, issame))
                    issame_list.append(issame)
        return path_list

    def parse_function(self, filename):
        img = Image.open(filename)
        image_resized = img.resize((128, 128))
        image_gray = image_resized.convert('L')
        image_arr = np.array(image_gray)
        image_ext = np.expand_dims(image_arr, axis=3)
        return image_ext

    def get_pairs(self, index):
        path_1, path_2, issame = self.testlist[index]
        img1, img2 = self.parse_function(path_1), self.parse_function(path_2)
        return img1, img2, issame


    # def get_batch(self, batch_size=64):
    #     self.trainlist = self.create_pairs()
    #     imageA, imageB = [], []
    #     id1, id2 = [], []
    #     label = []
    #
    #     for i in range(batch_size):
    #         imageA_cur, imageB_cur, id1_cur, id2_cur, label_cur = self.get_item(i)
    #
    #         imageA.append(imageA_cur)
    #         imageB.append(imageB_cur)
    #         id1.append([id1_cur])
    #         id2.append([id2_cur])
    #         label.append(label_cur)
    #
    #     id1_enc = np.array(self.enc.transform(id1).toarray())
    #     id2_enc = np.array(self.enc.transform(id2).toarray())
    #     label = np.array(label)
    #
    #     return np.array(imageA), np.array(imageB), id1_enc, id2_enc, label

    def get_batch(self, batch_size=64):
        imageA, imageB = [], []
        label = []
        self.testlist = self.get_lfw_paths()
        for i in range(self.cur, self.cur + batch_size):
            i = i % len(self.testlist)
            imageA_cur, imageB_cur, label_cur = self.get_pairs(i)
            imageA.append(imageA_cur)
            imageB.append(imageB_cur)
            label.append([label_cur])
        self.cur = self.cur + batch_size

        label = np.array(label)

        return np.array(imageA), np.array(imageB), label


if __name__ == '__main__':
    dataset = IFWDataset(img_path="dataset\lfw", pairs_path="dataset\pairs.txt")



