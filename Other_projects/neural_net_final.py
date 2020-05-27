import numpy as np
import pandas as pd
import csv
import keras
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.datasets import mnist
from torchvision import transforms
import timeit


def one_hot_encode(data: torch.utils.data.Dataset):
    return keras.utils.to_categorical(data, 10)

def flatten_vector(data: torch.utils.data.Dataset):
    return torch.flatten(data)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


training_data = list(mnist.MNIST("/home/jedrzej/Desktop/Machine_learning/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))
test_data = list(mnist.MNIST("/home/jedrzej/Desktop/Machine_learning/", download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))

net = SimpleNet()
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 128
epochs = 500

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), shuffle=False)

start = timeit.default_timer()

losses = []
for epoch in range(epochs):
    running_loss = 0.0
    train_iter = iter(train_loader)
    for x_batch, y_batch in train_iter:
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
test_iter = iter(test_loader)
with torch.no_grad():
    for x, y in test_loader:
        prediction = net(x)
        max_value, predicted_label = torch.max(prediction.data,0)
        
# Cuda - 1051s
# No cuda - 1946s 