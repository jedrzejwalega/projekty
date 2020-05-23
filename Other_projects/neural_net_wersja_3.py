import numpy as np
import pandas as pd
import csv
import keras
import torch 
import torch.nn as nn
import torch.nn.functional as F
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
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x

training_data = mnist.MNIST("/home/jedrzej/Desktop/Machine_learning/", train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode)
test_data = mnist.MNIST("/home/jedrzej/Desktop/Machine_learning/", download=True, transform=transforms.Compose([transforms.ToTensor(), flatten_vector]), target_transform=one_hot_encode)

# x_train = torch.rand(60000, 784)
# y_train = torch.randint(low=0, high=10, size=(60000, 1))

# Change labels into binary categorical vectors
# y_train = keras.utils.to_categorical(y_train, 10)

# # Create observation and label pairs for each of the datasets
# training_data = list(zip(x_train,y_train))


net = SimpleNet()
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 128
epochs = 500

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=False)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
times = []
losses = []
for epoch in range(5):
    
    n = 0
    running_loss = 0.0
    start = timeit.default_timer()

    for x_batch, y_batch in iter(train_loader):
    
        # print("X_batch type, shape, data type: ", type(x_batch), x_batch.shape, x_batch.dtype)
        # print("Y_batch type, shape, data type: ", type(y_batch), y_batch.shape, y_batch.dtype)

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
        
        n += 1
        
    print(n)
    if epoch % 10 == 0:
        print("Epoch: {epoch}, Loss: {loss}".format(epoch=epoch, loss=running_loss))
        losses.append(running_loss)

    stop = timeit.default_timer()
    print(f"TIME in batch {n}: ", stop - start)
    times.append(stop - start)

print ("\n ### Finished Training ### \n")

# with torch.no_grad():
#     imgid = []
#     labels = []
#     image_num = 1
#     for x, y in test_data:
#         prediction = net(x)
#         max_value, predicted_label = torch.max(prediction.data,0)
#         imgid.append(image_num)
#         labels.append(int(predicted_label))
#         image_num += 1

# figure, axes = plt.subplots(figsize=(12.8, 14.4))
# axes.plot(range(0,epochs,10), losses, color="red")
# plt.title("Loss change in model training", fontsize=18)
# axes.set_xlabel("Epochs", fontsize=15)
# axes.set_ylabel("Loss", fontsize=15)
# plt.show()

figure, axes = plt.subplots(figsize=(12.8, 14.4))
axes.bar(x=range(5), height=times)
plt.title("Wersja zla random no shuffle", fontsize=18)
axes.set_xlabel("Batches", fontsize=15)
axes.set_ylabel("Time [s]", fontsize=15)
plt.show()