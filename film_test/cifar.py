import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10

from film_test.data import CIFAR_PATH

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def vanilla_cifar():

    trainset = torchvision.datasets.CIFAR10(
        root=CIFAR_PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=CIFAR_PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
           'ship', 'truck')


class Cifar10QA(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        return super().__getitem__(index)

if __name__ == '__main__':
    c = Cifar10QA(root=CIFAR_PATH, train=True, download=True)

    print (len(c), c[0])