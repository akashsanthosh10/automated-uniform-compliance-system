import torch
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(torch.__version__)
print(torch.cuda.is_available())
print(dev)