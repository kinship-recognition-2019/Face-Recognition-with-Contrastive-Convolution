import numpy as np
import os
import tensorflow as tf


class IFWDataset():
    def __init__(self, img_path, pairs_path):
        self.pairs_path = pairs_path
        self.img_path = img_path
        self.testlist = self.get_lfw_paths()
        self.size = 6000
        self.cur = 0
        # self.testlist = self.get_lfw_paths(img_dir)

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
        # print(filename)
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_resized = tf.image.resize_images(image_decoded, [128, 128])
        image_gray = tf.image.rgb_to_grayscale(image_resized)
        return image_gray

    def get_pairs(self, index):
        (path_1, path_2, issame) = self.testlist[index]
        img1, img2 = self.parse_function(path_1), self.parse_function(path_2)
        return img1, img2, issame

    def get_batch(self):
        imageA = []
        imageB = []
        label = []

        for i in range(self.cur, self.cur + 64):
            i = i % self.size
            imageA_cur, imageB_cur, label_cur = self.get_pairs(i)
            imageA.append(imageA_cur)
            imageB.append(imageB_cur)
            label.append(tf.constant(label_cur))

        img1_list = tf.stack([imageA[0], imageA[1], imageA[2], imageA[3], imageA[4], imageA[5], imageA[6], imageA[7],
                              imageA[8], imageA[9], imageA[10], imageA[11], imageA[12], imageA[13], imageA[14],
                              imageA[15],
                              imageA[16], imageA[17], imageA[18], imageA[19], imageA[20], imageA[21], imageA[22],
                              imageA[23],
                              imageA[24], imageA[25], imageA[26], imageA[27], imageA[28], imageA[29], imageA[30],
                              imageA[31],
                              imageA[32], imageA[33], imageA[34], imageA[35], imageA[36], imageA[37], imageA[38],
                              imageA[39],
                              imageA[40], imageA[41], imageA[42], imageA[43], imageA[44], imageA[45], imageA[46],
                              imageA[47],
                              imageA[48], imageA[49], imageA[50], imageA[51], imageA[52], imageA[53], imageA[54],
                              imageA[55],
                              imageA[56], imageA[57], imageA[58], imageA[59], imageA[60], imageA[61], imageA[62],
                              imageA[63]
                              ])
        img2_list = tf.stack([imageB[0], imageB[1], imageB[2], imageB[3], imageB[4], imageB[5], imageB[6], imageB[7],
                              imageB[8], imageB[9], imageB[10], imageB[11], imageB[12], imageB[13], imageB[14],
                              imageB[15],
                              imageB[16], imageB[17], imageB[18], imageB[19], imageB[20], imageB[21], imageB[22],
                              imageB[23],
                              imageB[24], imageB[25], imageB[26], imageB[27], imageB[28], imageB[29], imageB[30],
                              imageB[31],
                              imageB[32], imageB[33], imageB[34], imageB[35], imageB[36], imageB[37], imageB[38],
                              imageB[39],
                              imageB[40], imageB[41], imageB[42], imageB[43], imageB[44], imageB[45], imageB[46],
                              imageB[47],
                              imageB[48], imageB[49], imageB[50], imageB[51], imageB[52], imageB[53], imageB[54],
                              imageB[55],
                              imageB[56], imageB[57], imageB[58], imageB[59], imageB[60], imageB[61], imageB[62],
                              imageB[63]])
        label_list = tf.stack([label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7],
                                label[8], label[9], label[10], label[11], label[12], label[13], label[14], label[15],
                                label[16], label[17], label[18], label[19], label[20], label[21], label[22], label[23],
                                label[24], label[25], label[26], label[27], label[28], label[29], label[30], label[31],
                                label[32], label[33], label[34], label[35], label[36], label[37], label[38], label[39],
                                label[40], label[41], label[42], label[43], label[44], label[45], label[46], label[47],
                                label[48], label[49], label[50], label[51], label[52], label[53], label[54], label[55],
                                label[56], label[57], label[58], label[59], label[60], label[61], label[62], label[63]])
        return img1_list, img2_list, label_list


if __name__ == '__main__':
    dataset = IFWDataset(img_path="dataset\lfw", pairs_path="dataset\pairs.txt")
    print(dataset.get_lfw_paths())


