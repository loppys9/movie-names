import logging

from torch.utils.data import Dataset

from .data import read_data

log = logging.getLogger(__name__)


class MovieNameDataset(Dataset):
    def __init__(self, filename, device):
        # read_data reads a pre-processed tensor from a file.
        self.have, self.label = read_data(filename, True, device)
        self.have /= 255
        self.label /= 255
        # print(self.have.shape, self.label.shape)
        # print(self.have[0])
        # assert(self.have.shape[0] == self.label.shape[0])

    def __len__(self):
        return len(self.have)

    def __getitem__(self, index):
        return self.have[index], self.label[index]
