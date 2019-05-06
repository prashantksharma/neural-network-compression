import torch


path = "models/model1.bin"
model = torch.load(path)
print(model)
