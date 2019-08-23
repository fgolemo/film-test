from comet_ml import Experiment
from film_test.cifar import qa_cifar

experiment = Experiment(
    api_key="ZfKpzyaedH6ajYSiKmvaSwyCs",
    project_name="film-test",
    workspace="fgolemo")
experiment.add_tag("qa")
experiment.add_tag("film-resnet")

from torch import nn, optim
from tqdm import trange

from film_test.resnet import resnet18
from film_test.traintest import train, test, device

EPOCHS = 24

net = resnet18(num_classes=2, film_inputs=2)
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

trainloader, testloader = qa_cifar()

for epoch in trange(EPOCHS):
    experiment.log_metric("epoch", epoch)
    train(
        net,
        trainloader,
        epoch,
        optimizer,
        criterion,
        qa=True,
        film=True,
        comet=experiment)
    test(net, testloader, criterion, qa=True, film=True, comet=experiment)
