import torch
import torchvision
from torch.utils.data import sampler
from PIL import Image
from src.utils import get_class_path


class STL10(object):
    """
    image shape : 96 x 96
    ['airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck']

    unlabeled ++
    """
    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        stl10_dataset = torchvision.datasets.STL10(root='./datasets',
                                                   split=mode,
                                                   transform=transformer,
                                                   download=True)

        stl10_loader = torch.utils.data.DataLoader(stl10_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=shuffle)
        return stl10_loader


class CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        if mode == 'train':
            train = True
        else:
            train = False

        cifar10_dataset = torchvision.datasets.CIFAR10(root='./datasets',
                                                       train=train,
                                                       transform=transformer,
                                                       download=True)

        cifar10_loader = torch.utils.data.DataLoader(cifar10_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=shuffle)

        return cifar10_loader


class CIFAR100(object):
    """
    image shape : 32 x 32
    """

    def __init__(self,
                 batch_size):

        self.classes = 10
        self.batch_size = batch_size

    def get_loader(self, transformer, mode='train', shuffle=True):
        if mode == 'train':
            train = True
        else:
            train = False

        cifar100_dataset = torchvision.datasets.CIFAR100(root='./datasets',
                                                         train=train,
                                                         transform=transformer,
                                                         download=True)

        cifar100_loader = torch.utils.data.DataLoader(cifar100_dataset,
                                                     batch_size=self.batch_size,
                                                     shuffle=shuffle)

        return cifar100_loader


class Custom_CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 root_path,
                 class_names,
                 dtype='train',
                 transformer=None):

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, cls in enumerate(class_names):
            class_path = get_class_path(root_path, dtype, cls)
            self.img_path += class_path
            self.labels += [i] * len(class_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)


class Custom_Binary_CIFAR10(object):
    """
    image shape : 32 x 32
    ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    """

    def __init__(self,
                 root_path,
                 class_names,
                 cls,
                 dtype='train',
                 transformer=None):

        self.img_path = []
        self.labels = []
        self.transformer = transformer

        for i, name in enumerate(class_names):
            class_path = get_class_path(root_path, dtype, name)

            self.img_path += class_path

            if i == cls:
                self.labels += [1] * len(class_path)
            else:
                self.labels += [0] * len(class_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx])
        label = self.labels[idx]

        if self.transformer is not None:
            img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.img_path)
