import numpy as np
import pandas as pd
import csv
import keras
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt



train_data_path = "/home/jedrzej/Desktop/mnist/train.csv"
test_data_path = "/home/jedrzej/Desktop/mnist/test.csv"

training_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

y_train = np.array(training_data["label"])
x_train = np.array(training_data.drop(labels = ["label"], axis=1))

# po co zamieniac z float64 na float32?
x_train = x_train.astype("float32")

x_test = np.array(test_data)
x_test = x_test.astype("float32")

y_train = keras.utils.to_categorical(y_train, 10)

training_data = list(zip(x_train,y_train))
test_data = x_test

def create_mini_batches(data, batch_size):
    batches = [data[k:k + batch_size] for k in range(0, len(data), batch_size)]
    torch_batches = []
    for mini_batch in batches:
        mini_batch_obs = []
        mini_batch_labels = []
        for pair in mini_batch:
            x,y = pair
            mini_batch_obs.append(x)
            mini_batch_labels.append(y)
        torch_batches.append((torch.from_numpy(np.array(mini_batch_obs)), torch.from_numpy(np.array(mini_batch_labels))))
        return torch_batches

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

net = SimpleNet()
criterion = nn.MSELoss(reduction="mean")
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 128
epochs = 500

losses = []
for epoch in range(epochs):
    running_loss = 0.0
    np.random.shuffle(training_data)
    batched_training_data = create_mini_batches(training_data, batch_size)
    for x_batch, y_batch in batched_training_data:
        # forward pass
        preds = net(x_batch)

        # backward pass
        loss = criterion(preds, y_batch)
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()

        # Print progress
        running_loss += loss.item()

    if epoch % 10 == 0:
        print("Epoch [%d] Loss: %.3f" % (epoch, running_loss))
        losses.append(running_loss)

print ("\n ### Finished Training ### \n")

torch_test_data = []
for x in test_data:
    torch_test_data.append(torch.from_numpy(x))
        
with torch.no_grad():
    imgid = []
    labels = []
    image_num = 1
    for x in torch_test_data:
        prediction = net(x)
        max_value, predicted_label = torch.max(prediction.data,0)
        imgid.append(image_num)
        labels.append(int(predicted_label))
        image_num += 1

figure, axes = plt.subplots(figsize=(12.8, 14.4))
axes.plot(range(0,epochs,10), losses, color="red")
plt.title("Loss change in model training", fontsize=18)
axes.set_xlabel("Epochs", fontsize=15)
axes.set_ylabel("Loss", fontsize=15)
plt.show()