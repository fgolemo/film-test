import torch
import torchvision
from PIL import Image
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

QUESTION_A = "is_transportation"
QUESTION_B = "is_not_transportation"


class Cifar10QA(CIFAR10):

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):
        super().__init__(root, train, transform, target_transform, download)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        cutoff_idx = 25000
        if not self.train:
            cutoff_idx = 5000

        if index < cutoff_idx:
            question = QUESTION_A
            question_idx = 0
            if self.targets[index] in [0, 1, 7, 8, 9]:
                answer = 1
            else:
                answer = 0
        else:
            question = QUESTION_B
            question_idx = 1
            if self.targets[index] in [0, 1, 7, 8, 9]:
                answer = 0
            else:
                answer = 1

        return img, target, question, question_idx, answer


def qa_cifar():

    trainset = Cifar10QA(
        root=CIFAR_PATH, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True, num_workers=2)

    testset = Cifar10QA(
        root=CIFAR_PATH, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader


if __name__ == '__main__':
    c = Cifar10QA(
        root=CIFAR_PATH, train=True, download=True, transform=transform_train)
    print(len(c))
    print(c[0])
    print(c[25000])
    img = c[0][0]
    print(img.size())

    # c = Cifar10QA(root=CIFAR_PATH, train=False, download=True)
    # print(len(c))
    # print(c[0])
    # print(c[5000])
