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
from typing import List
from numpy import isnan


def adjust_lr(optimizer, learning_rates):
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

def train_and_test_model(learning_sets:List[List[float]], batch_size, epochs, file_name:str):
    lr = [x[0] if len(x) == 1 else x[1] for x in learning_sets]
    criterion = nn.MSELoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50000, shuffle=False)

    start = timeit.default_timer()
    training_losses_minimal = []
    testing_losses_minimal = []

    for rates in learning_sets:
        torch.manual_seed(1)
        learning_rates = iter(rates)
        net = SimpleNet().to(device)
        optimizer = optim.SGD(net.parameters(), lr=next(learning_rates), momentum=0.9)
        losses = []
        testing_losses = []
        for epoch in range(epochs):
            print(epoch)
            if epoch % 2 == 0 and epoch > 0:
                adjust_lr(optimizer, learning_rates)
                losses= []
                testing_losses = []
            lr = get_lr(optimizer)
            print("Learning rate: ", lr)
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
                # print(loss.item())
                losses.append(loss.item())

            test_iter = iter(test_loader)
            with torch.no_grad():
                for x_test, y_test in test_iter:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    predictions = net(x_test)
                    testing_loss = criterion(predictions, y_test)
                    testing_losses.append(testing_loss.item())
                    # print(testing_loss.item())

        stop = timeit.default_timer()
        print (f"\n ### Finished Training in {stop - start} ### \n")
        
        min_training_loss = min(losses)
        min_testing_loss = min(testing_losses)
        training_losses_minimal.append((lr, min_training_loss))
        testing_losses_minimal.append((lr, min_testing_loss))
        print(losses, "\n\n", testing_losses)

    table = pd.DataFrame(data={"Learning rate":[x[0] for x in training_losses_minimal], "Minimal training loss":[x[1] for x in training_losses_minimal], "Minimal testing loss":[x[1] for x in testing_losses_minimal]})
    table.to_csv(file_name)
    return min(training_losses_minimal, key=lambda x:x[1]), min(testing_losses_minimal, key=lambda x:x[1])

def find_max_lr(step:float=1) -> int:
    
    criterion = nn.MSELoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50000, shuffle=False)
    lr = 1
    while True:
        torch.manual_seed(1)
        net = SimpleNet().to(device)
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
        losses = []
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
            losses.append(loss.item())

        test_iter = iter(test_loader)
        with torch.no_grad():
            for x_test, y_test in test_iter:
                x_test, y_test = x_test.to(device), y_test.to(device)
                predictions = net(x_test)
                testing_loss = criterion(predictions, y_test)
                losses.append(testing_loss.item())

        if isnan(losses).any() == True or float("inf") in losses:
            return lr - 1
        
        lr += step


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

training_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))
test_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode))

entry_len = training_data[0][0].shape[0]

learning_rates = list(np.arange(0.005, 0.405, 0.005))
learning_rates = [[np.round(x, decimals=3)] for x in learning_rates]
print(learning_rates)

minimal_rate = train_and_test_model(learning_sets=learning_rates, batch_size=128, epochs=2, file_name="/home/jedrzej/Desktop/learning_rates_2_epochs.csv")[0][0]
print("Minimal rate: ", minimal_rate)
new_learning_rates = [[minimal_rate] + x for x in learning_rates if x[0] <= minimal_rate]
print(new_learning_rates)
print(train_and_test_model(learning_sets=new_learning_rates, batch_size=128, epochs=4, file_name="/home/jedrzej/Desktop/learning_rates_4_epochs.csv"))

