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
