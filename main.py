import pandas as pd
import numpy as np
from tqdm import tqdm
from data_preprocess import generate_dataset
from model import Net
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

data = generate_dataset()
annotations = pd.read_csv('./data/annotations_cleaned.csv',
                          usecols=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
annotations = np.array(annotations, dtype=np.float32)
x = torch.from_numpy(data)
y = torch.from_numpy(annotations)
dataset = TensorDataset(x, y)
data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

model = Net()
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    for data in tqdm(data_loader):
        audio, label = data
        output = model(audio)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'test.pth')
