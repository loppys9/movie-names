import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR as StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import MovieNameDataset
from .model import MovieNameNetwork

NO = 96


def _get_best_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    return device


def _train(model, dataloader, loss_fn, optimizer):

    model.train()
    for batch, (x, y) in enumerate(dataloader):
        y_hat = model(x)
        loss = loss_fn(y_hat, y[:, :NO])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"loss: {loss.item():>7f}  [{batch}]")


def _to_chars(vals):
    s = ""
    for val in vals:
        s += chr(int(val.item()))

    print(s)


def _test(model, dataloader, loss_fn):
    model.eval()
    size = len(dataloader.dataset) * NO
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    num = 15
    total = 0
    bad = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y[:, :NO]).item()
            char_pred = (255 * pred).round()
            char_y = (255 * y[:, :NO]).round()
            for i, (a, b) in enumerate(zip(char_pred, char_y)):
                if total > num:
                    break
                if a[0] != b[0]:
                    bad += 1
                    total += 1
                    # print(a)
                    # print(b)
                    _to_chars(a)
                    _to_chars(b)
                    _to_chars(255 * X[i])
                    print("---")
                    # pred[i], y[:,:NO][i])
                else:
                    print("+++++++++++++++ Good? +++++++++++++++++++++++")
                    _to_chars(a)
                    _to_chars(b)
                    _to_chars(255 * X[i])
            correct += (char_pred == char_y).sum().item()

    print("bad: ", bad, size)
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train():

    batch_size = 64

    device = _get_best_device()

    training_dataset = MovieNameDataset("data/train_movie_names", device)
    testing_dataset = MovieNameDataset("data/test_movie_names", device)

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)

    model = MovieNameNetwork().to(device)
    # loss_fn = nn.MSELoss()
    loss_fn = nn.L1Loss()
    learning_rate = 0.001
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-7)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    epochs = 100
    for i in tqdm(range(epochs)):
        _train(model, train_dataloader, loss_fn, optimizer)
        if i == 99:
            _test(model, test_dataloader, loss_fn)
        scheduler.step()
