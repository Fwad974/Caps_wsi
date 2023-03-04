import torch
from torchvision import datasets, transforms

import torch
from torchvision import datasets, transforms
import pandas as pd
import os
import numpy as np
from torchvision.io import read_image

class ADP_Dataset(torch.utils.data.Dataset):
    def __init__(self, path="./ADP_V1.0_Release", transform=None):
        'Initialization'
        df = pd.read_csv(path + "/ADP_EncodedLabels_Release1_Flat.csv")

        self.labels = df[df.columns[1:]].to_numpy()

        self.paths = np.array([os.path.join(path, "img_res_1um_bicubic", i) for i in df['Patch Names']])
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        selected_path = self.paths[index]
        image = read_image(selected_path)
        # Load data and get label
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)

        return image, label


class Dataset:
    def __init__(self, dataset, _batch_size):
        super(Dataset, self).__init__()
        if dataset=="ADP":
            from torch.utils.data import DataLoader
            data = ADP_Dataset()

            self.train_dataset, self.test_dataset = torch.utils.data.random_split(data, [15901,1767])
            self.train_loader= DataLoader(self.train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader= DataLoader(self.test_dataset , batch_size=_batch_size, shuffle=True)
            if dataset == 'mnist':
                dataset_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])

                train_dataset = datasets.MNIST('./data/mnist', train=True, download=True,
                                               transform=dataset_transform)
                test_dataset = datasets.MNIST('./data/mnist', train=False, download=True,
                                              transform=dataset_transform)

                self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
                self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)

        elif dataset == 'cifar10':
            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            train_dataset = datasets.CIFAR10(
                './data/cifar', train=True, download=True, transform=data_transform)
            test_dataset = datasets.CIFAR10(
                './data/cifar', train=False, download=True, transform=data_transform)

            self.train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=_batch_size, shuffle=True)

            self.test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=_batch_size, shuffle=False)
        elif dataset == 'office-caltech':
            pass
        elif dataset == 'office31':
            pass
        if dataset == 'smallnorb':
            dataset_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            
            train_dataset = smallNORB('./data/SmallNORB', train=True, download=True,
                                           transform=dataset_transform)
            test_dataset = smallNORB('./data/SmallNORB', train=False, download=True,
                                          transform=dataset_transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=_batch_size, shuffle=False)
