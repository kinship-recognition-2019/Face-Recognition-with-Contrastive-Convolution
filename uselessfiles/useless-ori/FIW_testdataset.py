import torchvision.datasets as datasets
import os


class FIWTestDataset(datasets.ImageFolder):
    def __init__(self, img_path, pairs_path, transform):
        super(FIWTestDataset, self).__init__(img_path)
        self.img_path = img_path
        self.pairs_path = pairs_path
        self.transform = transform
        self.test_list = self.get_test_pairs()

    def get_test_pairs(self):
        pair_list = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines():
                label, p1, p2 = line.strip().split(',')
                path1 = os.path.join(self.img_path, p1)
                path2 = os.path.join(self.img_path, p2)
                if label == '1':
                    label = True
                else:
                    label = False
                pair_list.append((path1, path2, label))
        return pair_list

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.test_list[index]
        img1, img2 = self.transform(self.loader(path_1)), self.transform(self.loader(path_2))
        return img1, img2, issame

    def __len__(self):
        return len(self.test_list)