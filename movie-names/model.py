import torch
from torch import nn


class MovieNameNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            #nn.ReLU(),
            #nn.Linear(96, 96),
            nn.Sigmoid(),
            # nn.Softmax(),
        )

    def forward(self, x):
        y_hat = self.stack(x)
        return y_hat
