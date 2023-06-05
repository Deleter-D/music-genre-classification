from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from data_generator import generate_dataloader
from model import Net

data_loader = generate_dataloader()

model = Net()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for data in tqdm(data_loader, desc=f'Epoch {epoch + 1}'):
        audio, label = data
        output = model(audio)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'test.pth')
