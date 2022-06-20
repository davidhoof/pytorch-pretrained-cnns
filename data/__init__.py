import json
import os
import random
from collections import defaultdict

import git
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, KMNIST, FashionMNIST, ImageFolder, SVHN, SUN397
from torchvision.datasets.utils import download_and_extract_archive

import utils
from utils import CloneProgress


class ImageNet1k(Dataset):
    def __init__(self, data_root, split, transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}

        with open(os.path.join(data_root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)

        samples_dir = os.path.join(data_root, split)
        for syn_id in os.listdir(samples_dir):
            target = self.syn_to_class[syn_id]
            syn_folder = os.path.join(samples_dir, syn_id)
            for sample in os.listdir(syn_folder):
                sample_path = os.path.join(syn_folder, sample)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx])
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


class ImageNet1kData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 1000
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="train", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageNet1k(root=self.root_dir, split="val", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class GroceryStore(Dataset):
    _GIT_URL = "https://github.com/marcusklasson/GroceryStoreDataset.git"

    def __init__(self, root, split="train", transform=None, download=False):
        assert split in ['train', 'val', 'test']
        self.root = root
        self.samples_frame = []
        self.transform = transform

        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        dataset_path = None

        if split == "train":
            dataset_path = "train.txt"
        if split == "val":
            dataset_path = "val.txt"
        if split == "test":
            dataset_path = "test.txt"

        with open(os.path.join(root, "dataset", dataset_path), "rb") as f:
            self.samples_frame = pd.read_csv(f)

    def __len__(self):
        return len(self.samples_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root, "dataset",
                                self.samples_frame.iloc[idx, 0])
        x = Image.open(img_name)
        if self.transform:
            x = self.transform(x)
        return x, self.samples_frame.iloc[idx, 2]

    def _check_exists(self) -> bool:
        return os.path.exists(self.root)

    def _download(self) -> None:
        if self._check_exists():
            return
        git.Repo.clone_from(self._GIT_URL, self.root, progress=CloneProgress())


class GroceryStoreData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = GroceryStore
        self.mean = (0.5525, 0.4104, 0.2445)
        self.std = (0.2205, 0.1999, 0.1837)
        self.num_classes = 43  # needs to be one more than the 42 classes because its start with a 0. Dont ask why
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, split="train", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, split="val", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class HistAerial(ImageFolder):
    _DATASET_URL = "http://eidolon.univ-lyon2.fr/~remi1/HistAerialDataset/dataset/HistAerialDataset.zip"
    _DATASET_MD5 = "cff202e58d8905cf108ae0cc9cee9feb"

    def __init__(self, root, dataset_type="25x25", transform=None, download=False):
        assert dataset_type in ["25x25", "50x50", "100x100"]
        self.root = root

        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        root = os.path.join(root, f"{dataset_type}_overlap_0percent")
        super().__init__(root, transform)

    def _check_exists(self) -> bool:
        return os.path.exists(self.root)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)


class HistAerial25x25Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.valid_size = 0.15
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = HistAerial
        self.mean = (0.5525, 0.4104, 0.2445)
        self.std = (0.2205, 0.1999, 0.1837)
        self.num_classes = 7
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="25x25", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="25x25", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        valid_idx = indices[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=valid_sampler,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class HistAerial50x50Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.valid_size = 0.15
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = HistAerial
        self.mean = (0.4625, 0.4625, 0.4625)
        self.std = (0.2764, 0.2764, 0.2764)
        self.num_classes = 7
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="50x50", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="50x50", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        valid_idx = indices[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=valid_sampler,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class HistAerial100x100Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.valid_size = 0.15
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = HistAerial
        self.mean = (0.4616, 0.4616, 0.4616)
        self.std = (0.2759, 0.2759, 0.2759)
        self.num_classes = 42
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="100x100", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, dataset_type="100x100", transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        valid_idx = indices[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=valid_sampler,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class FractalDB60(ImageFolder):
    _DATASET_URL = f"https://onedrive.live.com/download?cid=A5E3C415BF70F5D8&resid=A5E3C415BF70F5D8%217797&authkey=AM3nK9dCz1LC2jk"  # own onedrive link
    _DATASET_MD5 = "843ba9a92bdd9dbfa972ad363f8ea8b6"

    def __init__(self, root, transform=None, download=False):
        self.root = root

        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(os.path.join(root, "fractaldb_cat60_ins1000"), transform)

    def _check_exists(self) -> bool:
        return os.path.exists(self.root)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, filename="fractaldb.tar.gz", download_root=self.root,
                                     md5=self._DATASET_MD5)


class FractalDB60Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.valid_size = 0.15
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = FractalDB60
        self.mean = (0.0622, 0.0622, 0.0622)
        self.std = (0.1646, 0.1646, 0.1646)
        self.num_classes = 60
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(48),
                transforms.CenterCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        valid_idx = indices[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=valid_sampler,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = CIFAR10
        self.mean = (0.49139968, 0.48215841, 0.44653091)
        self.std = (0.24703223, 0.24348513, 0.26158784)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class SUN397Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = SUN397
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = 899
        self.in_channels = 3
        self.valid_size = 0.15

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        train_idx = indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=train_sampler,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(38),
                transforms.RandomCrop(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, transform=transform, download=True)

        num_train = len(dataset)
        indices = list(range(num_train))
        split = int(np.floor(self.valid_size * num_train))

        valid_idx = indices[:split]
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=valid_sampler,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class TinyImageNetPaths:
    def __init__(self, root_dir):

        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNet(Dataset):
    _DATASET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    _DATASET_MD5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(self, root, mode='train', transform=None, download=False):
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.root = root

        if download:
            self._download()
        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        tinp = TinyImageNetPaths(os.path.join(root, "tiny-imagenet-200"))
        self.samples = tinp.paths[mode]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s[0])
        img = img.convert("RGB")
        label = s[1]

        if self.transform:
            img = self.transform(img)
        return img, label

    def _check_exists(self) -> bool:
        return os.path.exists(self.root)

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5)


class TinyImageNetData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = TinyImageNet
        self.mean = (0.4802, 0.4481, 0.3975)
        self.std = (0.2764, 0.2689, 0.2816)
        self.num_classes = 200
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, mode="train", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.CenterCrop(56),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, mode="val", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, mode="test", transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader


class SVHNData(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = SVHN
        self.mean = (0.4377, 0.4438, 0.4728)
        self.std = (0.1980, 0.2010, 0.1970)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, split="train", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, split="test", transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CIFAR100Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = CIFAR100
        self.mean = (0.50707516, 0.48654887, 0.44091784)
        self.std = (0.26733429, 0.25643846, 0.27615047)
        self.num_classes = 100
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class CINIC10Data(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, num_workers, part="all"):
        super().__init__()
        assert part in ["all", "imagenet", "cifar10"]
        self.part = part
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = (0.47889522, 0.47227842, 0.43047404)  # from https://github.com/BayesWatch/cinic-10
        self.std = (0.24205776, 0.23828046, 0.25874835)
        self.num_classes = 10
        self.in_channels = 3

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root=os.path.join(self.root_dir, "train"), transform=transform, is_valid_file= \
            lambda path: (self.part == "all") or \
                         (self.part == "imagenet" and not os.path.basename(path).startswith("cifar10-")) or \
                         (self.part == "cifar10" and os.path.basename(path).startswith("cifar10-")))
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = ImageFolder(root=os.path.join(self.root_dir, "valid"), transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class TensorData(pl.LightningDataModule):
    def __init__(self, data_class, root_dir, batch_size, num_workers):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_class = data_class

    def train_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),

            ]
        )
        dataset = self.data_class(root=self.root_dir, train=True, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = self.data_class(root=self.root_dir, train=False, transform=transform, download=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


class MNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(MNIST, root_dir, batch_size, num_workers)
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.num_classes = 10
        self.in_channels = 1


class KMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(KMNIST, root_dir, batch_size, num_workers)
        self.mean = (0.1918,)
        self.std = (0.3483,)
        self.num_classes = 49
        self.in_channels = 1


class FashionMNISTData(TensorData):

    def __init__(self, root_dir, batch_size, num_workers):
        super().__init__(FashionMNIST, root_dir, batch_size, num_workers)
        self.mean = (0.2860,)
        self.std = (0.3530,)
        self.num_classes = 10
        self.in_channels = 1


class RandomMinimizedCIFAR10Data(CIFAR10Data):
    def __init__(self, root_dir, batch_size, num_workers, dataset_size):
        super().__init__(root_dir, batch_size, num_workers)
        self.data_class = utils.minimize_dataset(CIFAR10, dataset_size, random.randint(0, dataset_size))


class RandomMinimizedCIFAR100Data(CIFAR100Data):
    def __init__(self, root_dir, batch_size, num_workers, dataset_size):
        super().__init__(root_dir, batch_size, num_workers)
        self.data_class = utils.minimize_dataset(CIFAR100, dataset_size, random.randint(0, dataset_size))


class RandomMinimizedGroceryStoreData(GroceryStoreData):
    def __init__(self, root_dir, batch_size, num_workers, dataset_size):
        super().__init__(root_dir, batch_size, num_workers)
        self.data_class = utils.minimize_dataset(GroceryStore, dataset_size, random.randint(0, dataset_size))


class RandomMinimizedSVHNData(SVHNData):
    def __init__(self, root_dir, batch_size, num_workers, dataset_size):
        super().__init__(root_dir, batch_size, num_workers)
        self.data_class = utils.minimize_dataset(SVHN, dataset_size, random.randint(0, dataset_size))


class RandomMinimizedTinyImageNetData(TinyImageNetData):
    def __init__(self, root_dir, batch_size, num_workers, dataset_size):
        super().__init__(root_dir, batch_size, num_workers)
        self.data_class = utils.minimize_dataset(TinyImageNet, dataset_size, random.randint(0, dataset_size))


all_datasets = {
    "cifar10": CIFAR10Data,
    "cifar100": CIFAR100Data,
    "mnist": MNISTData,
    "kmnist": KMNISTData,
    "fashionmnist": FashionMNISTData,
    "cinic10": CINIC10Data,
    "imagenet1k": ImageNet1kData,
    "svhn": SVHNData,
    "tinyimagenet": TinyImageNetData,
    "grocerystore": GroceryStoreData,
    "sun397": SUN397Data,
    "histaerial25x25": HistAerial25x25Data,
    "histaerial50x50": HistAerial50x50Data,
    "histaerial100x100": HistAerial100x100Data,
    "fractaldb60": FractalDB60Data
}

all_datasets_minimized = {
    "cifar10": RandomMinimizedCIFAR10Data,
    "cifar100": RandomMinimizedCIFAR100Data,
    "grocerystore": RandomMinimizedGroceryStoreData,
    "svhn": RandomMinimizedSVHNData,
    "tinyimagenet": RandomMinimizedTinyImageNetData
}


def get_dataset(name):
    return all_datasets.get(name)


def get_dataset_minimized(name):
    return all_datasets_minimized.get(name)
