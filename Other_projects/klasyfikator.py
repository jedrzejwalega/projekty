
# importing the libraries
import pandas as pd
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential
from torch.optim import Adam


train = pd.read_csv('/home/jedrzej/Desktop/train.csv')
test = pd.read_csv('/home/jedrzej/Desktop/test.csv')

sample_submission = pd.read_csv('/home/jedrzej/Desktop/sample_submission_I5njJSF.csv')

train_img = []
for img_name in train['id']:
    image_path = '/home/jedrzej/Desktop/train/' + str(img_name) + '.png'
    img = imread(image_path, as_gray=True)
    img = img.astype('float32')
    train_img.append(img)

train_x = np.array(train_img)
train_x = train_x.reshape(-1, 28*28).astype("float32")

train_y = train['label'].values

# create validation set
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 0.1, stratify = train_y)

input_num_units = 28*28
hidden_num_units = 500
output_num_units = 10

# set remaining variables
epochs = 20
learning_rate = 0.0005

model = Sequential(Linear(input_num_units, hidden_num_units),
                   ReLU(),
                   Linear(hidden_num_units, output_num_units))
# loss function
loss_fn = CrossEntropyLoss()

# define optimization algorithm
optimizer = Adam(model.parameters(), lr=learning_rate)

X = torch.Tensor([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

#Output
y = torch.Tensor([[1],[1],[0]])

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + torch.exp(-x))

#Derivative of Sigmoid Function/
def derivatives_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

#Variable initialization
epoch=7000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layer neurons
output_neurons = 1 #number of neurons in output layer

#weight and bias initialization
wh=torch.randn(inputlayer_neurons, hiddenlayer_neurons).type(torch.FloatTensor)
bh=torch.randn(1, hiddenlayer_neurons).type(torch.FloatTensor)
wout=torch.randn(hiddenlayer_neurons, output_neurons)
bout=torch.randn(1, output_neurons)

#Forward Propogation
hidden_layer_input1 = torch.mm(X, wh)
# hidden_layer_input = hidden_layer_input1 + bh
# hidden_layer_activations = sigmoid(hidden_layer_input)

# output_layer_input1 = torch.mm(hidden_layer_activations, wout)
# output_layer_input = output_layer_input1 + bout
# output = sigmoid(output_layer_input)
