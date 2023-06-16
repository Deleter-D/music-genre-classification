import time

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam
from torchmetrics.classification import Accuracy, Recall
from matplotlib import pyplot as plt

from config import Config
# from data_generator import generate_dataloader
from data_generator_AlexNet import generate_dataloader
from model import ViT
from AlexNet import AlexNet

cmd_args = Config().parse()
train_loader, val_loader = generate_dataloader()
is_cuda_available = torch.cuda.is_available()

# model = ViT(
#     image_size=(32, 3200),
#     patch_size=32,
#     num_classes=20,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1,
#     channels=1
# )

model = AlexNet()

if is_cuda_available:
    model = model.cuda()

criterion = CrossEntropyLoss()

if is_cuda_available:
    criterion = criterion.cuda()

# optimizer = SGD(model.parameters(), lr=0.1)
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# metrics record lists
train_loss_record = []
train_acc_record = []
# train_recall_record = []
valid_loss_record = []
valid_acc_record = []
# valid_recall_record = []

for epoch in range(cmd_args.epoch):

    # training
    model.train()
    train_epoch_loss = []
    train_epoch_acc_predictions = []
    train_epoch_acc_targets = []
    pbar_train = tqdm(train_loader, desc=f'Epoch {epoch + 1} training')
    for data in pbar_train:
        audio, label = data
        if is_cuda_available:
            audio = audio.cuda()
            label = label.cuda()
        output = model(audio)
        output = output.cpu()
        label = label.cpu()

        # get training metrics
        loss = criterion(output, label)
        train_epoch_loss.append(loss.item())
        for i in range(output.shape[0]):
            train_epoch_acc_predictions.append(torch.argmax(output[i]).item())
            train_epoch_acc_targets.append(torch.argmax(label[i]).item())

        pbar_train.set_postfix({'loss': loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # calculate training metrics
    train_epoch_mean_loss = np.mean(train_epoch_loss)
    accuracy = Accuracy(task='multiclass', num_classes=20)
    train_epoch_acc = accuracy(torch.tensor(train_epoch_acc_predictions), torch.tensor(train_epoch_acc_targets))
    tqdm.write(
        f'Epoch {epoch + 1} training mean loss: {train_epoch_mean_loss}, accuracy: {train_epoch_acc}')

    # record metrics
    train_loss_record.append(train_epoch_mean_loss)
    train_acc_record.append(train_epoch_acc)

    # save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'./checkpoint/epoch_{epoch + 1}.pth')
    time.sleep(1)

    # validation
    model.eval()
    val_epoch_loss = []
    val_epoch_acc_predictions = []
    val_epoch_acc_targets = []
    pbar_val = tqdm(val_loader, desc=f'Epoch {epoch + 1} validation')
    with torch.no_grad():
        for data in pbar_val:
            audio, label = data
            if is_cuda_available:
                audio = audio.cuda()
                label = label.cuda()
            output = model(audio)
            output = output.cpu()
            label = label.cpu()

            # get validation metrics
            loss = criterion(output, label)
            val_epoch_loss.append(loss.item())
            for i in range(output.shape[0]):
                val_epoch_acc_predictions.append(torch.argmax(output[i]).item())
                val_epoch_acc_targets.append(torch.argmax(label[i]).item())

            pbar_train.set_postfix({'loss': loss.item()})

        # calculate training metrics
        valid_epoch_mean_loss = np.mean(val_epoch_loss)
        accuracy = Accuracy(task='multiclass', num_classes=20)
        val_epoch_acc = accuracy(torch.tensor(val_epoch_acc_predictions), torch.tensor(val_epoch_acc_targets))
        tqdm.write(
            f'Epoch {epoch + 1} validation mean loss: {valid_epoch_mean_loss}, accuracy: {val_epoch_acc}')

        # record metrics
        valid_loss_record.append(valid_epoch_mean_loss)
        valid_acc_record.append(val_epoch_acc)
    time.sleep(1)

print(f'train losses: {train_loss_record}')
print(f'valid losses: {valid_loss_record}')

plt.figure()
plt.plot(range(cmd_args.epoch), train_loss_record, label='training losses')
plt.plot(range(cmd_args.epoch), valid_loss_record, label='validation losses')
plt.title('losses')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.savefig('./result/loss.svg')

plt.figure()
plt.plot(range(cmd_args.epoch), train_acc_record, label='training accuracy')
plt.plot(range(cmd_args.epoch), valid_acc_record, label='validation accuracy')
plt.title('accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('./result/accuracy.svg')

# plt.figure()
# plt.plot(range(cmd_args.epoch), train_recall_record, label='training recall')
# plt.plot(range(cmd_args.epoch), valid_recall_record, label='validation recall')
# plt.title('recall')
# plt.xlabel('epoch')
# plt.ylabel('recall')
# plt.legend()
# plt.savefig('./result/recall.svg')
