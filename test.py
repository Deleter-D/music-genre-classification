from AlexNet import AlexNet
import torch

x = torch.ones((32, 1, 32, 1600))
model = AlexNet()
output = model(x)
print(output.shape)
