import torch
from torch.utils.data import Dataset

class TweetDataset(Dataset):
    def __init__(self, data_path):
        self.inputs, self.labels = torch.load(data_path)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
