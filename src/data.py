import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_loader_device(dataset_name: str, batch_size=128):
    transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )

    base_path = os.path.abspath(os.path.dirname(__file__))
    base_path = os.path.join(base_path, "..", "data")

    if dataset_name == "mnist":
        train_set = datasets.MNIST(
            base_path, train=True, download=False, transform=transform
        )

        test_set = datasets.MNIST(
            base_path, train=False, download=False, transform=transform
        )

    elif dataset_name == "fashion_mnist":
        train_set = datasets.FashionMNIST(
            base_path, train=True, download=False, transform=transform
        )

        test_set = datasets.FashionMNIST(
            base_path, train=False, download=False, transform=transform
        )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )

    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, drop_last=True
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return train_set, test_set, train_loader, test_loader, device


def imgs_labels(database: str, dataset_name: str):
    """Get the tensors with its repectively labels from database.

    :param database: the database, i.e, 'mnist' or 'fashion_mnist'.
    :param dataset_name: the set name, i.e, 'train_set' or 'test_set'.

    """
    base_path = os.path.abspath(os.path.dirname(__file__))
    path_save = os.path.join(base_path, "..", "data", "imgs_labels")

    (train_set, test_set, _, _, _) = set_loader_device(database)

    if dataset_name == "train_set":
        dataset = train_set
    if dataset_name == "test_set":
        dataset = test_set

    dict = {}
    for label in range(10):
        imgs_label = list()
        for idx in range(len(dataset)):
            # (img, label) = dataset[sample_idx]
            if dataset[idx][1] == label:
                imgs_label.append(dataset[idx][0])

        dict[str(label)] = imgs_label

    torch.save(dict, f"{path_save}" + f"/{database}_{dataset_name}_imgs.pth")

    return None
