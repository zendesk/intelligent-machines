
# coding: utf-8

# # A simple neural network

# Neural networks are amazing. When used well, they seem simulate intellegent behavior to accomplish a wide variety of tasks ranging from image recognition to image generation, from language comprehension to language translation, and much more. The thing about neural networks, though, is that they are typically designed with a specific task in mind. 
# 
# Clever people have been spent a lot of time trying to discover ways to implement neural networks in order to optimize for specific tasks. This has lead to the discovery of a wide diversity of architectures that tend to be suited for their specific tasks. You may be familiar with some of these architectures. One such artchitecture is the convolutional nerual network (ConvNet). Another is the recurrent neural network (RNN). ConvNets tend to be suited for image recognition tasks, whereas RNNs tend to perform well on sequence based tasks, such as language processing.
# 
# The simplest kind of neural network is called a feed forward network, which is what we'll focus on in this notebook. When you break down some of the more advanced architectures, it becomes apparent that they are basically just clever ways to combine various feed forward networks that use the same mathematical operations we'll learn about by building a feed forward network. For example, the LSTM architecture uses a system of feed forward networks to gate information as it passes through the network.
# 
# So using our understanding of vector and matrix operations, lets implement a feed forward network from scratch using numpy!

# ### The feed forward network 
# 
# Any neural network needs to be designed around a task. In the previous session we talked about some of the mathematical operations that occur inside a feed forward network, but haven't talked about any of the operations that happen on either end.
# 
# On the input end, we typically need to consider getting our data cleaned and organized in such a way that that we can feed it in as a vector. On the other end, we need to pose the learning task in the form of a loss function, the goal that we are optimizing for.

# We will use a standard dataset that does not require much preprocessing and we'll choose a simple loss function that doesn't require much work to understand for now. We can focus on these things in a later session. Input preprocessing tends to be different for each network you create, and loss functions will be discussed when we cover the the training process in greater depth.
# 
# How can we implement a feed forward network without understanding how to train it? We can't! We'll gloss over backpropogation in this session and save that for the next session.

# #### Imports



import numpy as np
from sklearn.datasets import load_boston
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# #### Load the dataset
# 
# For this network, we'll use the boston housing dataset. The goal for this dataset is to learn a model that helps us predict the median market value of the house based on features gathered assocated with the property. This is idea for a simple neural network since we want to use a simple loss function. Since we're predicting a price, we can frame this as a regression problem and use a sum of squares loss function.



boston = load_boston(return_X_y=False)




print(boston.DESCR)


# #### Visualize the data



boston.data




boston.target




df = pd.DataFrame(boston.data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                                        'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                        'B', 'LSTAT'])
df.loc[:, 'target'] = boston.target
df.head()


# With any dataset, there is the opportunity to explore the data to find feature that are going to likely be the most helpful when training the model. This is the case for any model and it is generally recommended. Since we're focusing on the network implementation, we'll skip the data exploration for now. 

# #### Define a data batcher
# When we send data through the network, we can send samples through one at a time or in batches. So its useful to have a function that handle that for us.



def data_batcher(inputs, targets, batch_size=5):
    assert len(inputs) == len(targets)
    
    inputs = inputs[: -(len(inputs) % batch_size)].reshape(-1, batch_size, inputs.shape[1])
    targets = targets[: -(len(targets) % batch_size)].reshape(-1, batch_size)
    
    batched_data = list()
    for x_inputs, y_targets in zip(inputs, targets):
        batched_data.append([x_inputs, y_targets.reshape(-1, 1)])
    
    return batched_data


# #### Define a loss function



def loss_function(pred, targ):
    return 0.5 * np.sum((targ - pred)**2)




def loss_function_deriv(pred, targ):
    return (targ - pred)


# ### Implement a linear network



# Set how many times to iterate through the dataset
epochs = 50000

# Set hyperparamters
lr = 0.00001
batch_size = 5

#Set dimensions of hidden layer
input_size = 13
hidden_size = 10

samples_seen = 0

# Set learnable weight matrices
hidden_weights = np.random.normal(loc=0.0, scale=0.01, size=(13, 10))
output_weights = np.random.normal(loc=0.0, scale=0.01, size=(10, 1))  # a single output!

# Train the network
for each_epoch in range(epochs):
    
    #create
    batched_data = data_batcher(boston.data, boston.target, batch_size=batch_size)
    epoch_loss = 0
    
    for batch_input, batch_target in batched_data:

        samples_seen += 1 * batch_size

        # Forward Pass
        hidden_layer_out = np.dot(batch_input, hidden_weights) # No activation function
        pred = np.dot(hidden_layer_out, output_weights) # No activation function

        # loss
        epoch_loss += loss_function(pred, batch_target)

        # backprop
        l2delta = loss_function_deriv(pred, batch_target) # No derivative
        l1delta = np.dot(l2delta, output_weights.T)  # No derivative since no nonlinear activation!

        # gradient descent
        output_weights += np.dot(hidden_layer_out.T, l2delta) * lr
        hidden_weights += np.dot(batch_input.T, l1delta) * lr

    if each_epoch % 5000 == 0:
        print("epoch {} ... loss: {} ... samples seen: {}".format(each_epoch, round(epoch_loss, 2), samples_seen))


# ## Implement a non-linear network



def sigmoid(x):  # Non-linearity
    return 1. / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1. - x)




# Set how many times to iterate through the dataset
epochs = 50000

# Set hyperparamters
lr = 0.00001
batch_size = 5

#Set dimensions of hidden layer
input_size = 13
hidden_size = 10

# Set learnable weight matrices
hidden_weights = np.random.normal(loc=0.0, scale=0.01, size=(13, 10))
output_weights = np.random.normal(loc=0.0, scale=0.01, size=(10, 1))  # a single output!

samples_seen = 0
# Train the network
for each_epoch in range(epochs):
    
    #create
    batched_data = data_batcher(boston.data, boston.target, batch_size=batch_size)
    epoch_loss = 0
    
    for batch_input, batch_target in batched_data:
        samples_seen += 1 * batch_size
        
        # Forward pass
        hidden_layer_out = sigmoid(np.dot(batch_input, hidden_weights))
        pred = np.dot(hidden_layer_out, output_weights) # No activation function

        # loss
        epoch_loss += loss_function(pred, batch_target)

        # backprop
        l2delta = loss_function_deriv(pred, batch_target) * 1
        l1delta = np.dot(l2delta, output_weights.T)  * sigmoid_deriv(hidden_layer_out)

        # gradient descent
        output_weights += np.dot(hidden_layer_out.T, l2delta) * lr
        hidden_weights += np.dot(batch_input.T, l1delta) * lr

    if each_epoch % 5000 == 0:
        print("epoch {} ... loss: {} ... samples seen: {}".format(each_epoch, round(epoch_loss, 2), samples_seen))


# #### Create a layer class and do it again



class Layer(object):
    
    def __init__(self, 
                 input_dim,
                 output_dim,
                 learning_rate,
                 name,
                 activation=None):

        self.weights = np.random.normal(0.0, 0.01, (input_dim, output_dim))
        self.name = name
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        activations = {'sigmoid': self.sigmoid,
                       'relu': self.relu,
                        None: self.non_activation}

        derivs = {'sigmoid': self.sigmoid_deriv,
                  'relu': self.relu_driv,
                   None: self.non_activation_deriv}


        self.activation = activations[activation]
        self.activation_deriv = derivs[activation]

    def forward_pass(self, input_x):
        self.input = input_x
        self.output = self.activation(np.dot(self.input, self.weights))
        return self.output

    def backward_pass(self, output_delta):
        self.weight_output_delta = output_delta * self.activation_deriv(self.output)
        return np.dot(self.weight_output_delta, self.weights.T)

    def update_weights(self):
        self.weights += np.dot(self.input.T, self.weight_output_delta) * self.learning_rate

    def sigmoid(self, x):
        return 1. / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (1. - x)
    
    def non_activation(self, x):
        return x
    
    def non_activation_deriv(self, x):
        return 1
    
    def relu(self, x):
        return 0 if x <= 0 else x
    
    def relu_deriv(self, x):
        return 1




# Set how many times to iterate through the dataset
epochs = 50000

# Set hyperparamters
lr = 0.00001
batch_size = 5


# init layers
hidden_layer = Layer(input_dim=13, output_dim=10, learning_rate=lr, name='hidden_layer')
output_layer = Layer(input_dim=10, output_dim=1, learning_rate=lr, name='output_layer')

samples_seen = 0

# Train the network
for each_epoch in range(epochs):

    #create batches
    batched_data = data_batcher(boston.data, boston.target, batch_size=batch_size)

    epoch_loss = 0
    for batch_input, batch_target in batched_data:
        samples_seen += 1 * batch_size

        # Forward pass
        hidden_out = hidden_layer.forward_pass(batch_input)
        pred = output_layer.forward_pass(hidden_out)

        # loss
        epoch_loss += loss_function(pred, batch_target)

        # backprop
        output_delta = loss_function_deriv(pred, batch_target)
        hidden_delta = output_layer.backward_pass(output_delta)
        hidden_layer.backward_pass(hidden_delta)

        # gradient descent
        output_layer.update_weights()
        hidden_layer.update_weights()

    if each_epoch % 5000 == 0:
        print("epoch {} ... loss: {} ... samples seen: {}".format(each_epoch, round(epoch_loss, 2), samples_seen))




layers = [hidden_layer.weights, output_layer.weights,
          hidden_layer.weights.flatten(), output_layer.weights.flatten()]

fig, axs = plt.subplots(figsize=(18, 12), nrows=2, ncols=2)
bar = False
for ix, (layer, ax) in enumerate(zip(layers, axs.flatten())):
    if ix < 2:
        if ix == 2:
            bar = True
        sns.heatmap(layer, ax=ax, cbar=bar)
        ax.set_title('layer {} weights'.format(ix+1), fontdict={'fontsize': 18})
    else:
        sns.distplot(layer, ax=ax);


