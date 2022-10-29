import pandas as pd
import torch
from torch.utils.data import Dataset


class textDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        self.install_data = pd.read_csv(csv_path, encoding='utf8', sep=',')

    def __len__(self):
        return len(self.install_data)

    def __getitem__(self, idx):
        input_data = pd.DataFrame([])
        label = pd.DataFrame([])

        input_data = self.install_data.iloc[idx, 1]
        label = self.install_data.iloc[idx, 0]
        return label, input_data







