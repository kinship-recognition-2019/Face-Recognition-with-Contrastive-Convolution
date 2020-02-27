import os

from torchvision import datasets


class LFWDataset(datasets.ImageFolder):
    def __init__(self, img_path, pairs_path, transform=None):
        super(LFWDataset, self).__init__(img_path, transform)
        self.img_path = img_path
        self.pairs_path = pairs_path
        self.test_list = self.get_lfw_pairs()

    def get_lfw_pairs(self, file_ext='jpg'):
        pairs = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        pair_list = []
        for i in range(len(pairs)):
            pair = pairs[i]
            if len(pair) == 3:
                path0 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(self.img_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(self.img_path, pair[2], pair[2] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):
                pair_list.append((path0, path1, issame))
        return pair_list

    def __getitem__(self, i):
        (path_1, path_2, issame) = self.test_list[i]
        img1, img2 = self.transform(self.loader(path_1)), self.transform(self.loader(path_2))
        return img1, img2, issame

    def __len__(self):
        return len(self.test_list)


if __name__ == '__main__':
    img_path = 'C:/Users/VineAsh/Desktop/Face-Recognition-with-Contrastive-Convolution/dataset/lfw'
    list_path = 'C:/Users/VineAsh/Desktop/Face-Recognition-with-Contrastive-Convolution/dataset/pairs.txt'
    L = LFWDataset(img_path, list_path)
    f = open('out.txt', 'w')
    f.write(str(L.get_lfw_pairs()))