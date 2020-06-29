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
from itertools import cycle


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


def adjust_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def one_hot_encode(data: torch.utils.data.Dataset):
    return keras.utils.to_categorical(data, 10)

def flatten_vector(data: torch.utils.data.Dataset):
    return torch.flatten(data)

def train_and_test_model(learning_sets:List[List[float]], batch_size:int, epochs_per_lr:List[List[float]], min_by_epochs:List[List[float]], path=str):
    criterion = nn.MSELoss(reduction="mean")
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=50000, shuffle=False)

    color_map = plt.cm.get_cmap('gist_ncar', len(learning_sets))
    color_map_index = 0
    figure, axes = set_up_plot("CIFAR-10 learning rates comparison", "Epochs", "Loss (log10)")

    start = timeit.default_timer()
    
    minimal_losses = []
    for rates, epoch_limits, epochs_of_interest in zip(learning_sets, epochs_per_lr, min_by_epochs):
        # print(rates, epoch_limits, epochs_of_interest)
        torch.manual_seed(1)
        learning_rates = cycle(rates)
        net = SimpleNet().to(device)
        optimizer = optim.SGD(net.parameters(), lr=next(learning_rates), momentum=0.9)
        losses = []
        testing_losses = []
        for epochs in epoch_limits:
            # print("Epochs: ", epochs)
            for epoch in range(epochs):
                losses = train(net, optimizer, train_loader, losses, epoch, criterion, batch_size)
                testing_losses = test(net, test_loader, testing_losses, epoch, criterion, batch_size)
            new_lr = next(learning_rates)
            adjust_lr(optimizer, new_lr)

        stop = timeit.default_timer()
        # print (f"\n ### Finished Training in {stop - start} ### \n")
        
        losses_of_interest = [losses[n] for n in epochs_of_interest]
        testing_losses_of_interest = [testing_losses[n] for n in epochs_of_interest]
        print("Losses: ", losses)
        # print("Testing losses: ", testing_losses)

        min_training_loss_no_extra, min_testing_loss_no_extra = calculate_min_losses(losses_of_interest, testing_losses_of_interest)
        minimal_losses.append((rates, min_training_loss_no_extra, min_testing_loss_no_extra))

        min_training_loss, min_testing_loss = calculate_min_losses(losses, testing_losses)
        min_training_local_losses, min_testing_local_losses = calculate_local_losses(losses, testing_losses, epoch_limits)
        
        # axes.plot(range(sum(epoch_limits)), losses, color=color_map(color_map_index), label=f"Lr={rates}, Training min={min_training_loss}, Local mins={min_training_local_losses}")
        # axes.plot(range(sum(epoch_limits)), testing_losses, color=np.array(color_map(color_map_index)) * 0.6, label=f"Lr={rates}, Testing min={min(testing_losses)}, Local mins={min_testing_local_losses}")
        plot_training_testing_losses(axes, sum(epoch_limits), losses, testing_losses, color_map, color_map_index, min_training_loss, min_testing_loss, min_training_local_losses, min_testing_local_losses, rates)
        color_map_index += 1
    
    axes.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(minimal_losses)
    print("MIN: ", min(minimal_losses, key=lambda x:x[1]))
    return min(minimal_losses, key=lambda x:x[1])

def set_up_plot(title, x_label_title, y_label_title):
    figure, axes = plt.subplots(figsize=(12.8, 14.4))
    plt.title(title, fontsize=18)
    axes.set_xlabel(x_label_title, fontsize=15)
    axes.set_ylabel(y_label_title, fontsize=15)
    plt.yscale(value="log")
    plt.grid()
   
    return figure, axes

def train(model, optimizer, train_loader, losses, epoch, criterion, batch_size):
    # print(f"Epoch {epoch} training learning rate: ", get_lr(optimizer))
    running_loss_count = 0.0
    running_loss_sum = 0.0
    train_iter = iter(train_loader)
    for x_batch, y_batch in train_iter:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        # forward pass
        preds = model(x_batch)

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
    # print(f"Epoch {epoch}: Training loss mean: {running_loss_mean}")
    return losses

def test(model, test_loader, testing_losses, epoch, criterion, batch_size):
    running_loss_test_count = 0.0
    running_loss_test_sum = 0.0
    test_iter = iter(test_loader)
    with torch.no_grad():
        for x_test, y_test in test_iter:
            x_test, y_test = x_test.to(device), y_test.to(device)
            predictions = model(x_test)
            testing_loss = criterion(predictions, y_test)
            running_loss_test_count += batch_size
            running_loss_test_sum += testing_loss.item() * batch_size
        running_loss_test_mean = running_loss_test_sum / running_loss_test_count 
        testing_losses.append(running_loss_test_mean)

    # print(f"Epoch {epoch}: Testing loss mean: {running_loss_test_mean}")
    return testing_losses

def calculate_min_losses(losses, testing_losses):
    min_training_loss = min(losses)
    min_testing_loss = min(testing_losses)
    return min_training_loss, min_testing_loss

def calculate_local_losses(losses, testing_losses, epochs):
    min_training_local_losses = []
    min_testing_local_losses = []
    first_index = 0
    for epoch_cycle in epochs:
        min_training_local_losses.append(min(losses[first_index:first_index + epoch_cycle]))
        min_testing_local_losses.append(min(testing_losses[first_index:first_index + epoch_cycle]))
        first_index = epoch_cycle

    return min_training_local_losses, min_testing_local_losses

def plot_training_testing_losses(axes, epochs, losses, testing_losses, color_map, color_map_index, min_training_loss, min_testing_loss, min_training_local_losses, min_testing_local_losses, rates):
    print(f"Rates: {rates}, Losses: {losses}")
    axes.plot(range(epochs), losses, color=color_map(color_map_index), label=f"Lr={rates}, Training min={min_training_loss}, Local mins={min_training_local_losses}")
    axes.plot(range(epochs), testing_losses, color=np.array(color_map(color_map_index)) * 0.6, label=f"Lr={rates}, Testing min={min(testing_losses)}, Local mins={min_testing_local_losses}")

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

basic_rates = [18,10,1,0.1]
rate_modifiers = [0.03, 0.1, 0.3]
modified_rates = [[a,a*r,a*r*r] for a in basic_rates for r in rate_modifiers]


# Step 1 - find minimal learning rate for first epoch:
learning_rates = [[x] for x in np.arange(0.1, 1.1, 0.05)]
epochs = [[3] for x in learning_rates]
min_by_epochs = [[0] for x in learning_rates]
minimal_loss_part_one = train_and_test_model(learning_sets=learning_rates, batch_size=128, epochs_per_lr=epochs, min_by_epochs=min_by_epochs, path="/home/jedrzej/Desktop/wykres.png")[0][0]
print(minimal_loss_part_one)

new_learning_rates = [[minimal_loss_part_one, x] for x in np.arange(minimal_loss_part_one, 0, -0.05)]
new_epochs = [[3, 2] for x in new_learning_rates]
min_by_epochs = [[3] for x in new_learning_rates]
minimal_loss_part_two = train_and_test_model(learning_sets=new_learning_rates, batch_size=128, epochs_per_lr=new_epochs, min_by_epochs=min_by_epochs, path="/home/jedrzej/Desktop/wykres5.png")
print(minimal_loss_part_two)