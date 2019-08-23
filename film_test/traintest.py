import torch
from tqdm import tqdm
import torch.nn.functional as F

from film_test.util import make_debug_qa_diagram

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Training
def train(net,
          trainloader,
          epoch,
          optimizer,
          criterion,
          qa=False,
          film=False,
          comet=None):
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
            inputs, img_class, question_idxs, answers = data
            inputs, targets, question_idxs = inputs.to(device), \
                                             answers.to(device), \
                                             question_idxs.to(device)
            question_idxs = F.one_hot(question_idxs, num_classes=2).float()

        optimizer.zero_grad()
        if film:
            outputs = net(inputs, question_idxs)
        else:
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


def test(net,
         testloader,
         criterion,
         qa=False,
         film=False,
         comet=None,
         epoch_no=None):
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
                inputs, img_class, question_idxs, answers = data
                inputs, targets, question_idxs = inputs.to(device), \
                                                 answers.to(device), \
                                                 question_idxs.to(device)
                question_idxs = F.one_hot(question_idxs, num_classes=2).float()

            if not film:
                outputs = net(inputs)
            else:
                outputs = net(inputs, question_idxs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx in [0, 50] and comet is not None:  # twice per epoch
                # get indices of bad predictions this minibatch (100 items per MB)
                bp = (predicted.eq(targets) == 0).nonzero()
                if len(bp) != 0:
                    # get first bad prediction
                    bad_pred = bp[0]
                    make_debug_qa_diagram(inputs[bad_pred],
                                          question_idxs[bad_pred],
                                          targets[bad_pred],
                                          predicted[bad_pred],
                                          img_class[bad_pred], epoch_no, comet)

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
