import os
import random
from PIL import Image

import tensorflow as tf
import numpy as np


class CasiaFaceDataset():
    def __init__(self, noofpairs=32, is_train=True):
        self.istrain = is_train
        self.imagelist = self.get_id_imagelist()
        self.noofcategories = len(self.imagelist) - 1
        self.noofpairs = noofpairs
        self.trainlist = []
        # self.trainlist = self.create_pairs()

    def get_id_imagelist(self):
        f = open("dataset/afterlist.txt")
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
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [128, 128])
        image_gray = tf.image.rgb_to_grayscale(image_resized)
        return image_gray

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
            path_img1 = os.path.join('dataset/CASIA-WebFace/', image_name1)  # Location to the image
            path_img2 = os.path.join('dataset/CASIA-WebFace/', image_name2)
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

    def get_batch(self):
        self.trainlist = self.create_pairs()
        imageA = []
        imageB = []

        id1 = []
        id2 = []
        label = []

        for i in range(64):
            imageA_cur, imageB_cur, id1_cur, id2_cur, label_cur = self.get_item(i)
            imageA.append(imageA_cur)
            imageB.append(imageB_cur)
            # batch_size = 10575
            # id1_cur = tf.expand_dims(tf.constant(id1_cur), 0)
            # id1_cur = tf.expand_dims(id1_cur, 1)
            # indices = tf.expand_dims(tf.range(0, batch_size, 1), 0)
            # print(id1_cur.shape)
            # print(indices.shape)
            # concated = tf.concat([indices, id1_cur], 0)
            # onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 10]), 1.0, 0.0)

            id1.append(tf.constant(id1_cur))
            id2.append(tf.constant(id2_cur))
            label.append(tf.constant(label_cur))

        # print(id1)
        # print(imageA)

        img1_list = tf.stack([imageA[0], imageA[1], imageA[2], imageA[3], imageA[4], imageA[5], imageA[6], imageA[7],
                            imageA[8], imageA[9], imageA[10], imageA[11], imageA[12], imageA[13], imageA[14], imageA[15],
                            imageA[16], imageA[17], imageA[18], imageA[19], imageA[20], imageA[21], imageA[22], imageA[23],
                            imageA[24], imageA[25], imageA[26], imageA[27], imageA[28], imageA[29], imageA[30], imageA[31],
                            imageA[32], imageA[33], imageA[34], imageA[35], imageA[36], imageA[37], imageA[38], imageA[39],
                            imageA[40], imageA[41], imageA[42], imageA[43], imageA[44], imageA[45], imageA[46], imageA[47],
                            imageA[48], imageA[49], imageA[50], imageA[51], imageA[52], imageA[53], imageA[54], imageA[55],
                            imageA[56], imageA[57], imageA[58], imageA[59], imageA[60], imageA[61], imageA[62], imageA[63]
                            ])
        img2_list = tf.stack([imageB[0], imageB[1], imageB[2], imageB[3], imageB[4], imageB[5], imageB[6], imageB[7],
                            imageB[8], imageB[9], imageB[10], imageB[11], imageB[12], imageB[13], imageB[14], imageB[15],
                            imageB[16], imageB[17], imageB[18], imageB[19], imageB[20], imageB[21], imageB[22], imageB[23],
                            imageB[24], imageB[25], imageB[26], imageB[27], imageB[28], imageB[29], imageB[30], imageB[31],
                            imageB[32], imageB[33], imageB[34], imageB[35], imageB[36], imageB[37], imageB[38], imageB[39],
                            imageB[40], imageB[41], imageB[42], imageB[43], imageB[44], imageB[45], imageB[46], imageB[47],
                            imageB[48], imageB[49], imageB[50], imageB[51], imageB[52], imageB[53], imageB[54], imageB[55],
                            imageB[56], imageB[57], imageB[58], imageB[59], imageB[60], imageB[61], imageB[62], imageB[63]])
        c1_list = tf.stack([id1[0], id1[1], id1[2], id1[3], id1[4], id1[5], id1[6], id1[7],
                              id1[8], id1[9], id1[10], id1[11], id1[12], id1[13], id1[14], id1[15],
                              id1[16], id1[17], id1[18], id1[19], id1[20], id1[21], id1[22], id1[23],
                              id1[24], id1[25], id1[26], id1[27], id1[28], id1[29], id1[30], id1[31],
                              id1[32], id1[33], id1[34], id1[35], id1[36], id1[37], id1[38], id1[39],
                              id1[40], id1[41], id1[42], id1[43], id1[44], id1[45], id1[46], id1[47],
                              id1[48], id1[49], id1[50], id1[51], id1[52], id1[53], id1[54], id1[55],
                              id1[56], id1[57], id1[58], id1[59], id1[60], id1[61], id1[62], id1[63]])
        c2_list = tf.stack([id2[0], id2[1], id2[2], id2[3], id2[4], id2[5], id2[6], id2[7],
                              id2[8], id2[9], id2[10], id2[11], id2[12], id2[13], id2[14], id2[15],
                              id2[16], id2[17], id2[18], id2[19], id2[20], id2[21], id2[22], id2[23],
                              id2[24], id2[25], id2[26], id2[27], id2[28], id2[29], id2[30], id2[31],
                              id2[32], id2[33], id2[34], id2[35], id2[36], id2[37], id2[38], id2[39],
                              id2[40], id2[41], id2[42], id2[43], id2[44], id2[45], id2[46], id2[47],
                              id2[48], id2[49], id2[50], id2[51], id2[52], id2[53], id2[54], id2[55],
                              id2[56], id2[57], id2[58], id2[59], id2[60], id2[61], id2[62], id2[63]])
        target_list = tf.stack([label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7],
                            label[8], label[9], label[10], label[11], label[12], label[13], label[14], label[15],
                            label[16], label[17], label[18], label[19], label[20], label[21], label[22], label[23],
                            label[24], label[25], label[26], label[27], label[28], label[29], label[30], label[31],
                            label[32], label[33], label[34], label[35], label[36], label[37], label[38], label[39],
                            label[40], label[41], label[42], label[43], label[44], label[45], label[46], label[47],
                            label[48], label[49], label[50], label[51], label[52], label[53], label[54], label[55],
                            label[56], label[57], label[58], label[59], label[60], label[61], label[62], label[63]])

        batch_size = 64

        c1_list = tf.expand_dims(c1_list, 1)
        indices_c1 = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated_c1 = tf.concat([indices_c1, c1_list], 1)
        onehot_c1_list = tf.sparse_to_dense(concated_c1, tf.stack([batch_size, 10575]), 1.0, 0.0)

        c2_list = tf.expand_dims(c2_list, 1)
        indices_c2 = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated_c2 = tf.concat([indices_c2, c2_list], 1)
        onehot_c2_list = tf.sparse_to_dense(concated_c2, tf.stack([batch_size, 10575]), 1.0, 0.0)

        # target_list = tf.expand_dims(target_list, 1)
        # indices_target = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        # concated_target = tf.concat([indices_target, c1_list], 1)
        # onehot_target_list = tf.sparse_to_dense(concated_target, tf.stack([batch_size, 10575]), 1.0, 0.0)
        # print(onehot_c1_list)

        return img1_list, img2_list, onehot_c1_list, onehot_c2_list, target_list


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
