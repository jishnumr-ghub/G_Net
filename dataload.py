
import torch
from torch.utils.data import Dataset
import pickle
import os
import glob

class GestureDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for path in file_paths:
            with open(path, 'rb') as f:
                self.data.append(pickle.load(f))

    def __len__(self):
        return sum(len(item['feature']) for item in self.data)

    def __getitem__(self, idx):
        video_idx = 0
        while idx >= len(self.data[video_idx]['feature']):
            idx -= len(self.data[video_idx]['feature'])
            video_idx += 1

        feature = torch.tensor(self.data[video_idx]['feature'][idx], dtype=torch.float32)
        gesture = torch.tensor(self.data[video_idx]['gesture_GT'][idx], dtype=torch.long)
        error = torch.tensor(self.data[video_idx]['error_GT'][idx], dtype=torch.float32)
        return feature, gesture, error

def load_dataset(root_dir, pattern="*.pkl"):
    return glob.glob(os.path.join(root_dir, pattern))

def get_dataloader(file_paths, batch_size=1, shuffle=True):
    dataset = GestureDataset(file_paths)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
