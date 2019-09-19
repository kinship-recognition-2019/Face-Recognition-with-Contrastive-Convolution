import os
import random
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class CasiaFaceDataset():
    def __init__(self, noofpairs=32, is_train=True, n_onehot=10575):
        self.istrain = is_train
        self.imagelist = self.get_id_imagelist()
        self.noofcategories = len(self.imagelist) - 1
        self.noofpairs = noofpairs
        self.trainlist = []
        self.enc = OneHotEncoder()
        self.enc.fit([[x] for x in range(n_onehot)])
        # self.trainlist = self.create_pairs()

    def get_id_imagelist(self):
        f = open("dataset/casialist.txt")
        lines = f.readlines()
        subjectdict = dict()
        for name in lines[:]:
            # name = name[:-1]
            subject = name.split('/')[0]
            subjectdict.setdefault(subject, [])
            subjectdict[subject].append(name)
        f.close()
        imagelist = []
        for i, (key, value) in enumerate(subjectdict.items()):
            imagelist.append((i, key, value))
        # print(imagelist)
        return imagelist

    def get_random_two_images(self, tupleA, tupleB):
        classA = tupleA[0]
        classB = tupleB[0]
        listA = tupleA[2]
        listB = tupleB[2]
        # print(listA)
        imageA = np.random.choice(listA)
        imageB = np.random.choice(listB)
        
        while imageA == imageB:
            imageB = np.random.choice(listB)

        return "/".join(imageA.split("/")[-2:]), classA, "/".join(imageB.split("/")[-2:]), classB

    def parse_function(self, filename):
        # print(filename)
        # image_string = tf.read_file(filename)
        # image_decoded = tf.image.decode_jpeg(image_string)
        # image_resized = tf.image.resize_images(image_decoded, [128, 128])
        # image_gray = tf.image.rgb_to_grayscale(image_resized)

        img = Image.open(filename)
        image_resized = img.resize((128, 128))
        image_gray = image_resized.convert('L')
        image_arr = np.array(image_gray)
        image_ext = np.expand_dims(image_arr, axis=3)
        return image_ext

    def create_pairs(self):
        pairlist = []

        for n in range(self.noofpairs):
            # print(n)
            posCatList = np.arange(0, self.noofcategories)
            i = np.random.choice(posCatList)
            # print(self.noofcategories)
            # print(self.imagelist[i])

            imageA, c1, imageB, c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[i])
            # imageA = self.parse_function(imageA)
            # imageB = self.parse_function(imageB)
            pairlist.append([imageA, c1, imageB, c2, 1])
            # img1_list.append([imageA])
            # img2_list.append([imageB])
            # c1_list.append([c1])
            # c2_list.append([c2])
            # target_list.append(["1"])

            negativeCategoriesList = np.delete(np.arange(0, self.noofcategories), i)

            j = np.random.choice(negativeCategoriesList)
            imageA, c1, imageB, c2 = self.get_random_two_images(self.imagelist[i], self.imagelist[j])

            pairlist.append([imageA, c1, imageB, c2, 0])
            # img1_list.append([imageA])
            # img2_list.append([imageB])
            # c1_list.append([c1])
            # c2_list.append([c2])
            # target_list.append(["0"])

            random.shuffle(pairlist)

        return pairlist

    def get_len(self):
        if self.istrain is True:
            # print(len(self.train_list))
            return len(self.trainlist)

    def get_item(self, i):
        if self.istrain is True:
            image_name1 = self.trainlist[i][0][:-1]
            image_name2 = self.trainlist[i][2][:-1]
            id1 = self.trainlist[i][1]
            id2 = self.trainlist[i][3]
            label = self.trainlist[i][4]
            # print(image_name1)
            path_img1 = os.path.join('Dataset/CASIA-WebFace', image_name1)  # Location to the image
            path_img2 = os.path.join('Dataset/CASIA-WebFace', image_name2)
            # print(path_img1,path_img2,id1,id2,label)

            # print(path_img1)
            if os.path.exists(path_img1) and os.path.exists(path_img2):
                # print('Both images exist')
                # img1 = Image.open(path_img1).convert('L')
                # img2 = Image.open(path_img2).convert('L')

                imageA = self.parse_function(path_img1)
                imageB = self.parse_function(path_img2)

                # print('Type image:',type(img1))
                # print('image name1 ,image name2 ,id1,id2,label:',img1,img2,id1,id2,label)
                return imageA, imageB, int(id1), int(id2), int(label)

    def get_batch(self, batch_size=64):
        self.trainlist = self.create_pairs()
        imageA, imageB = [], []
        id1, id2 = [], []
        label = []

        for i in range(batch_size):
            imageA_cur, imageB_cur, id1_cur, id2_cur, label_cur = self.get_item(i)

            imageA.append(imageA_cur)
            imageB.append(imageB_cur)
            id1.append([id1_cur])
            id2.append([id2_cur])
            label.append([label_cur])

        id1_enc = np.array(self.enc.transform(id1).toarray())
        id2_enc = np.array(self.enc.transform(id2).toarray())
        label = np.array(label)

        return np.array(imageA), np.array(imageB), id1_enc, id2_enc, label


if __name__ == '__main__':
    dataset = CasiaFaceDataset()
    dataset.get_batch()
    # # dataset.create_pairs()
    # print(dataset.get_batch())
    # sess = tf.Session()
    # labels = [1, 3, 5, 7, 9]
    # batch_size = tf.size(labels)
    # print(sess.run(batch_size))
    # labels = tf.expand_dims(labels, 1)
    # print(sess.run(labels))
    # indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    # print(sess.run(indices))
    # concated = tf.concat([indices, labels], 1)
    # print(sess.run(concated))
    # onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)
    # print(onehot_labels)
