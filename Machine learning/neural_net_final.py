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
from math import ceil


class Model():
    def __init__(self, criterion, batch_size):
        # Model parameters
        self.net = SimpleNet()
        self.optimizer = None
        self.criterion = criterion
        self.batch_size = batch_size
        # Different losses containers
        self.losses = []
        self.testing_losses = []
        self.min_training_loss = None
        self.min_testing_loss = None
        self.min_training_local_losses = []
        self.min_testing_local_losses = []
    
    def train(self, train_loader):
        running_loss_count = 0.0
        running_loss_sum = 0.0
        train_iter = iter(train_loader)
        batch_number = 0
        for x_batch, y_batch in train_iter:
            batch_number += 1
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            # forward pass
            preds = self.net(x_batch)

            # backward pass
            loss = self.criterion(preds, y_batch)
            self.optimizer.zero_grad()
            loss.backward()

            # update parameters
            self.optimizer.step()

            # print progress
            running_loss_count += self.batch_size
            running_loss_sum += loss.item() * self.batch_size

            if batch_number % 20 == 0:
                running_loss_mean = running_loss_sum/running_loss_count
                self.losses.append(running_loss_mean)
                running_loss_sum = 0.0
                running_loss_count = 0.0
                batch_number = 0
        if batch_number != 0:
            running_loss_mean = running_loss_sum/running_loss_count
            self.losses.append(running_loss_mean)

        return self.losses

    def adjust_lr(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
    
    def test(self, test_loader):
        running_loss_test_count = 0.0
        running_loss_test_sum = 0.0
        test_iter = iter(test_loader)
        batch_number = 0
        with torch.no_grad():
            for x_test, y_test in test_iter:
                batch_number += 1
                x_test, y_test = x_test.to(device), y_test.to(device)
                predictions = self.net(x_test)
                testing_loss = self.criterion(predictions, y_test)
                running_loss_test_count += self.batch_size
                running_loss_test_sum += testing_loss.item() * self.batch_size
                
                if batch_number % 20 == 0:
                    running_loss_test_mean = running_loss_test_sum/running_loss_test_count
                    self.testing_losses.append(running_loss_test_mean)
                    running_loss_test_sum = 0.0
                    running_loss_test_count = 0.0
                    batch_number = 0

        if batch_number != 0:
            running_loss_test_mean = running_loss_test_sum/running_loss_test_count
            self.testing_losses.append(running_loss_test_mean)

        return self.testing_losses
        
    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


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
    

def one_hot_encode_digits(data: torch.utils.data.Dataset):
    return keras.utils.to_categorical(data, 10)

def flatten_vector(data: torch.utils.data.Dataset):
    return torch.flatten(data)

def train_and_test_model(learning_sets:List[List[float]], batch_size:int, epochs_per_lr:List[List[float]], min_by_epochs:List[List[float]], path=str):
    train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    color_map = plt.cm.get_cmap('gist_ncar', len(learning_sets))
    color_map_index = 0
    figure_both, axes_both = set_up_plot("CIFAR-10 learning rates comparison", "Epochs", "Loss (log10)")
    figure_training, axes_training = set_up_plot("CIFAR-10 learning rates comparison", "Epochs", "Loss log(10)")
    figure_testing, axes_testing = set_up_plot("CIFAR-10 learning rates comparison", "Epochs", "Loss log(10)")

    start = timeit.default_timer()
    
    losses_per_epoch = ceil(len(train_loader.dataset)/batch_size/20)
    
    deciding_losses = []
    for rates, epoch_limits, epochs_of_interest in zip(learning_sets, epochs_per_lr, min_by_epochs):
        print(f"Learning rates: {rates}\nEpochs per lr: {epoch_limits}\nBest lr chosen by epochs: {epochs_of_interest}")
        torch.manual_seed(1)
        learning_rates = cycle(rates)
        criterion = nn.MSELoss(reduction="mean")
        model = Model(criterion=criterion, batch_size=batch_size)
        model.optimizer = optim.SGD(model.net.parameters(), lr=next(learning_rates), momentum=0.9)
        model.net = model.net.to(device)
        
        for epochs in epoch_limits:
            for epoch in range(epochs):
                losses = model.train(train_loader)
                testing_losses = model.test(test_loader)
            new_lr = next(learning_rates)
            model.adjust_lr(new_lr)

        stop = timeit.default_timer()
        print (f"\n ### Finished Training in {stop - start} ### \n")
        
        starting_index = [((n-1) * losses_per_epoch,n * losses_per_epoch) for n in epochs_of_interest]
        losses_of_interest = [x for n in epochs_of_interest for x in losses[(n-1) * losses_per_epoch:n * losses_per_epoch]]
        testing_losses_of_interest = [x for n in epochs_of_interest for x in testing_losses[(n-1) * losses_per_epoch:n * losses_per_epoch]]
        
        # Minimal losses only by chosen epochs of interest
        deciding_min_training_loss, deciding_minimal_testing_loss = calculate_min_losses(losses_of_interest, testing_losses_of_interest)
        deciding_losses.append((rates, deciding_min_training_loss, deciding_minimal_testing_loss))
        
        # Minimal losses by all epochs
        model.min_training_loss, model.min_testing_loss = calculate_min_losses(losses, testing_losses)

        # Minimal local losses, meaning minimal loss for every learning rate used
        model.min_training_local_losses, modelmin_testing_local_losses = calculate_local_losses(losses, testing_losses, epoch_limits)
        
        # Plot training and testing losses on one plot
        all_epochs = sum(epoch_limits)
        color = color_map(color_map_index)
        plot_training_losses(axes_both, all_epochs, model, color, rates)
        plot_testing_losses(axes_both, all_epochs, model, color, rates)
        
        # Plot only training losses
        plot_training_losses(axes_training, all_epochs, model, color, rates)
        
        # Plot only testing losses
        plot_testing_losses(axes_testing, all_epochs, model, color, rates)
        color_map_index += 1
    
    save_plot(figure_both, axes_both, path + "_both.png")
    save_plot(figure_training, axes_training, path + "_training.png")
    save_plot(figure_testing, axes_testing, path + "_testing.png")
    
    write_to_csv(path, deciding_losses)
    return min(deciding_losses, key=lambda x:x[1])

def set_up_plot(title, x_label_title, y_label_title):
    figure, axes = plt.subplots(figsize=(12.8, 14.4))
    plt.title(title, fontsize=18)
    axes.set_xlabel(x_label_title, fontsize=15)
    axes.set_ylabel(y_label_title, fontsize=15)
    axes.set_yscale(value="log")
    axes.grid(axis="both", which="both")
   
    return figure, axes

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

def plot_training_losses(axes, epochs, model, color, rates):
    axes.plot(np.arange(0, epochs, 1/(len(model.losses)/epochs)), model.losses, color=color, label=f"Lr={rates}, Training min={model.min_training_loss}, Local mins={model.min_training_local_losses}")

def plot_testing_losses(axes, epochs, model, color, rates):
    axes.plot(np.arange(0, epochs, 1/(len(model.testing_losses)/epochs)) , model.testing_losses, color=np.array(color) * 0.6, label=f"Lr={rates}, Testing min={model.min_testing_loss}, Local mins={model.min_testing_local_losses}")

def save_plot(figure, axes, path):
    axes.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    figure.savefig(path, bbox_inches="tight")
    plt.close(fig=figure)

def write_to_csv(path, minimal_losses):
    table = pd.DataFrame(data={"Learning rates":[x[0] for x in minimal_losses], "Minimal training loss":[x[1] for x in minimal_losses], "Minimal testing loss":[x[2] for x in minimal_losses]})
    table.to_csv(path + ".csv")

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

training_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode_digits))
test_data = list(cifar.CIFAR10("/home/jedrzej/Desktop/Machine_learning/", download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode_digits))

entry_len = training_data[0][0].shape[0]

# Step 1 - find minimal learning rate for first epoch:
learning_rates = [[round(x,2)] for x in np.arange(0.1, 1.05, 0.05)]
epochs = [[3] for x in learning_rates]
min_by_epochs = [[1] for x in learning_rates]
best_lr_part_one = train_and_test_model(learning_sets=learning_rates, batch_size=128, epochs_per_lr=epochs, min_by_epochs=min_by_epochs, path="/home/jedrzej/Desktop/etap_pierwszy")[0][0]
print(best_lr_part_one)

new_learning_rates = [[best_lr_part_one, round(x,2)] for x in np.arange(best_lr_part_one, 0, -0.05)]
new_epochs = [[3, 2] for x in new_learning_rates]
min_by_epochs = [[4] for x in new_learning_rates]
best_lr_part_two = train_and_test_model(learning_sets=new_learning_rates, batch_size=128, epochs_per_lr=new_epochs, min_by_epochs=min_by_epochs, path="/home/jedrzej/Desktop/etap_drugi")[0]
print(f"Best learning rates: {best_lr_part_two}")