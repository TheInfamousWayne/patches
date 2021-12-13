import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import ipdb
import numpy as np

SPLIT = ''


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'slices', 'images', line.strip())
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels


class DTDDataloader(data.Dataset):
    def __init__(self, path='DTD', transform=None, skip_mouse_id='-1', train=True):
        classes, class_to_idx = find_classes(os.path.join(path, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform

        if train:
            filename = [os.path.join(path, 'labels', 'all_mouse_data', 'train' + SPLIT + '.txt')] \
                if skip_mouse_id == "-1" else \
                [os.path.join(path, 'labels', f'without_mouse_{skip_mouse_id}', 'train' + SPLIT + '.txt')]
        else:
            filename = [os.path.join(path, 'labels', 'all_mouse_data', 'test' + SPLIT + '.txt')] \
                if skip_mouse_id == "-1" else \
                [os.path.join(path, 'labels', f'without_mouse_{skip_mouse_id}', 'test' + SPLIT + '.txt')]

        self.images, self.labels = make_dataset(filename, path, class_to_idx)
        assert (len(self.images) == len(self.labels))
        self.data = self.load_data()

    def __getitem__(self, index):
        _img = Image.open(self.images[index])
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label

    def __len__(self):
        return len(self.images)

    def load_data(self):
        data = []
        for img in self.images:
            data.append(np.asarray(Image.open(img)).reshape(1,20,20,1))  # number_of_images, H, W, C

        return np.concatenate(data, axis=0)

class Dataloder():
    def __init__(self, path, spatial_size, batchsize, skip_mouse_id):
        normalize = transforms.Normalize(mean=[0.486],
                                         std=[0.229])

        transform_train_norandom = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            normalize,
        ])
        transform_train = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            # transforms.RandomCrop(spatial_size),
            transforms.RandomResizedCrop(spatial_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(9*spatial_size//8),
            transforms.CenterCrop(spatial_size),
            transforms.ToTensor(),
            normalize,
        ])

        trainset_norandom = DTDDataloader(path, transform_train_norandom, skip_mouse_id, train=True)
        trainset = DTDDataloader(path, transform_train, skip_mouse_id, train=True)
        testset = DTDDataloader(path, transform_test, skip_mouse_id, train=False)

        kwargs = {'num_workers': 8, 'pin_memory': True}
        trainloader_norandom = torch.utils.data.DataLoader(trainset_norandom, batch_size=batchsize, shuffle=False, **kwargs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
        batchsize, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
       batchsize, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainset = trainset
        self.testset = testset
        self.trainloader_norandom = trainloader_norandom
        self.trainloader = trainloader
        self.testloader = testloader

    def getloader(self):
        return self.classes, self.trainset, self.testset, self.trainloader, self.testloader, self.trainloader_norandom
