import torch

print("CUDA available:", torch.cuda.is_available())  # usually False on Mac
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())
