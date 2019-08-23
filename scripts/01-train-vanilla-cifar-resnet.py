from comet_ml import Experiment
import torch

experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="film-test",
    workspace="fgolemo")

from torch import nn, optim
from tqdm import trange

from film_test.cifar import trainloader, testloader
from film_test.resnet import resnet18
from film_test.traintest import train, test, device

EPOCHS = 24

net = resnet18(num_classes=10)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

for epoch in trange(EPOCHS):
    experiment.log_metric("epoch", epoch)
    train(net, trainloader, epoch, optimizer, criterion, comet=experiment)
    test(net, testloader, criterion, comet=experiment)
