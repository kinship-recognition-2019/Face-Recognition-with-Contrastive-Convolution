import torchvision.datasets as datasets
import os


class FIWTrainDataset(datasets.ImageFolder):
    def __init__(self, img_path, pairs_path, transform):
        super(FIWTrainDataset, self).__init__(img_path)
        self.img_path = img_path
        self.pairs_path = pairs_path
        self.transform = transform
        self.trainset = self.get_train_pairs()

    def get_train_pairs(self):
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

    def __getitem__(self, index):
        (path_1, path_2, id1, id2, issame) = self.trainset[index]
        img1, img2 = self.transform(self.loader(path_1)), self.transform(self.loader(path_2))
        return img1, img2, id1, id2, issame

    def __len__(self):
        return len(self.trainset)