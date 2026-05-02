import torch.nn as nn
try:
    emb = nn.Embedding(10, '20,43')
except Exception as e:
    print(f"Error: {e}")
    print(f"Type: {type(e)}")
