import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
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


def adjust_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = next(learning_rates)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

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

learning_sets = [[8, 1, 0.5], [5, 0.5, 0.1], [2, 1.2, 0.3]]
criterion = nn.MSELoss(reduction="mean")
batch_size = 128
epochs = 60

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=50000, shuffle=False)

color_map = plt.cm.get_cmap('rainbow', len(learning_sets))
n=0

figure, axes = plt.subplots(figsize=(12.8, 14.4))
plt.title("Loss change in model training", fontsize=18)
axes.set_xlabel("Epochs", fontsize=15)
axes.set_ylabel("Loss", fontsize=15)


start = timeit.default_timer()

for rates in learning_sets:
    learning_rates = iter(rates)
    net = SimpleNet().to(device)
    optimizer = optim.SGD(net.parameters(), lr=next(learning_rates), momentum=0.9)
    losses = []
    testing_losses = []
    for epoch in range(epochs):
        print(get_lr(optimizer))
        running_loss_count = 0.0
        running_loss_sum = 0.0
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
                    running_loss_count += batch_size
                    running_loss_sum += loss.item() * batch_size
        running_loss_mean = running_loss_sum/running_loss_count
        losses.append(running_loss_mean)
        print(f"In epoch {epoch}: Training loss mean: {running_loss_mean}")

        if epoch % 20 == 0 and epoch > 0:
            adjust_lr(optimizer)

        running_loss_test_count = 0.0
        running_loss_test_sum = 0.0
        test_iter = iter(test_loader)
        with torch.no_grad():
            for x_test, y_test in test_iter:
                x_test, y_test = x_test.to(device), y_test.to(device)
                predictions = net(x_test)
                testing_loss = criterion(predictions, y_test)
                running_loss_test_count += batch_size
                running_loss_test_sum += testing_loss.item() * batch_size
            running_loss_test_mean = running_loss_test_sum / running_loss_test_count
            Ntesting_losses.append(running_loss_test_mean)
        print(f"In epoch {epoch}: Testing loss mean: {running_loss_test_mean}")

    stop = timeit.default_timer()
    print (f"\n ### Finished Training in {stop - start} ### \n")

    axes.plot(range(epochs), losses, color=color_map(n), label=f"Training lr = {rates}")
    axes.plot(range(epochs), testing_losses, color=np.array(color_map(n)) * 0.6, label=f"Testing lr = {rates}")
    n += 1
axes.legend()
plt.show()
