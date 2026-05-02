import torch
from torch.utils.data import DataLoader, TensorDataset

ds = TensorDataset(torch.randn(10, 1))
try:
    dl = DataLoader(ds, batch_size='20,43')
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
