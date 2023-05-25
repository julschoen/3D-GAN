import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os


class DATA(Dataset):
  def __init__(self, path):
    self.data = np.load(path)['X']
    self.len = self.data.shape[0]

  def __getitem__(self, index):
    image = self.data[index]
    image = np.clip(image, -1,1)
    return torch.from_numpy(image).float()

  def __len__(self):
    return self.len

class DATA_DIR(Dataset):
  def __init__(self, path):
    self.dir = path
    nums = os.listdir(self.dir)
    nums = [int(x[:-4]) for x in nums if x.endswith('.npz')]
    self.len = max(nums)

  def __getitem__(self, index):
    image = np.load(os.path.join(self.dir, f'{index}.npz'))['X']
    image = np.clip(image, -1,1)
    return torch.from_numpy(image).float()

  def __len__(self):
    return self.len
