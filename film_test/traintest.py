import torch
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training
def train(net, trainloader, epoch, optimizer, criterion, qa=False, comet=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data) in enumerate(tqdm(trainloader)):
        if not qa:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
        else:
            inputs, _, _, question_idxs, answers = data
            inputs, targets, question_idxs = inputs.to(device), \
                                             answers.to(device), \
                                             question_idxs.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        _loss = (train_loss / (batch_idx + 1))
        acc = 100. * correct / total

        if comet is not None and batch_idx % 20 == 0:
            comet.log_metric("train loss", _loss)
            comet.log_metric("train acc", acc)

        tqdm.write(f'[{batch_idx}/{len(trainloader)}]\t'
                   f'Loss: {loss} | Acc: {acc} ({correct}/{total})')


def test(net, testloader, criterion, qa=False, comet=None):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(tqdm(testloader)):
            if not qa:
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)
            else:
                inputs, _, _, question_idxs, answers = data
                inputs, targets, question_idxs = inputs.to(device), \
                                                 answers.to(device), \
                                                 question_idxs.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            _loss = test_loss / (batch_idx + 1)
            acc = 100. * correct / total

            if comet is not None:
                comet.log_metric("test loss", _loss)
                comet.log_metric("test acc", acc)

            tqdm.write(f'[{batch_idx}/{len(testloader)}]\t'
                       f'Loss: {loss} | Acc: {acc} ({correct}/{total})')

    # Save checkpoint.

    print(f"test acc: {acc}%")
    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     torch.save(state, './checkpoint/ckpt.pth')
    # best_acc = acc
