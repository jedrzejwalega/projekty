import numpy as np
import pandas as pd
import csv
import keras
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import cifar
from torchvision import transforms
import timeit
from math import ceil


def one_hot_encode(data: torch.utils.data.Dataset):
    return keras.utils.to_categorical(data, 10)

def flatten_vector(data: torch.utils.data.Dataset):
    return torch.flatten(data)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(entry_len, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

training_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))
test_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))

entry_len = training_data[0][0].shape[0]

net = SimpleNet().to(device)
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 128
epochs = 500
batch_size_test = 1000

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)

start = timeit.default_timer()

losses = []
for epoch in range(epochs):
    running_loss = 0.0
    train_iter = iter(train_loader)
    for x_batch, y_batch in train_iter:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # forward pass
            preds = net(x_batch)

            # backward pass
            loss = criterion(preds, y_batch)
            optimizer.zero_grad()
            loss.backward()

            # update parameters
            optimizer.step()

            # print progress
            running_loss += loss.item()

    if epoch % 10 == 0:
        print("Epoch: {epoch}, Loss: {loss}".format(epoch=epoch, loss=running_loss))
        losses.append(running_loss)

print ("\n ### Finished Training ### \n")
stop = timeit.default_timer()
print(stop - start)

# Testing
testing_loss = []
test_iter = iter(test_loader)
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        prediction = net(x)
        loss = criterion(prediction, y).item()
        testing_loss.append(loss)
        max_value, predicted_label = torch.max(prediction.data,0)

figure, axes = plt.subplots(figsize=(12.8, 14.4))
axes.plot(range(0,epochs,10), losses, color="red")
plt.title("Loss change in model training", fontsize=18)
axes.set_xlabel("Epochs", fontsize=15)
axes.set_ylabel("Loss", fontsize=15)
plt.show()

figure, axes = plt.subplots(figsize=(12.8, 14.4))
axes.scatter(range(ceil(len(test_data)/batch_size_test)), testing_loss, color="red")
plt.title("Loss change in model testing", fontsize=18)
axes.set_ylim([0, max(testing_loss) * 1.2])
axes.set_xlabel("Batch", fontsize=15)
axes.set_ylabel("Loss", fontsize=15)
plt.show()