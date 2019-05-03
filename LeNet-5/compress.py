import torch


path = "models/model.bin"
model = torch.load(path)
print(model)
