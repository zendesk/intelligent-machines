
# coding: utf-8

# # A simple feed forward neural network

# Neural networks are amazing. When properly constructed, they can be used to perform specific tasks such as image recognition and language translation. They are complex machines however, grounded in mathematics and require diligent tuning efforts to get right. Typically, neural networks are designed with a specific task in mind, and the network architecture reflects characteristics of the problem.
# 
# The search for architectures suitable for specific tasks is an ongoing one, however you may be familiar with some of the architectures that have been discovered so far. One such architecture is the convolutional neural network (ConvNet). Another is the recurrent neural network (RNN). ConvNets have enjoyed much success in the domain of image recognition, where a characteristic of the problem is the correlative nature of pixel values when forming images, whereas RNNs tend to perform well on sequence based tasks, where signal is embedded in the order of sequence elements.
# 
# The simplest kind of neural network is called a restricted bolztmann machine, also called the feed-forward network, which is what we'll focus on in this notebook. When you break down some of the more advanced architectures, it becomes apparent that they are basically just clever ways to combine various feed-forward networks that use the same mathematical operations we'll learn about by building a simple feed-forward network.
# 
# So using our understanding of vector and matrix operations, let's implement a feed-forward network from scratch using numpy!

# ## What we'll cover

#  - how to build a simple neural network to approximate a mathematical function
#  - explore the limitations of a purely linear network
#  - explore how using non-linear functions in the network all us to model non-linear mathematical functions
#  - how to set up some data for training
#  - how to implement the forward pass of a neural network
#  - how to predict housing prices using a NN
# 
# #### and if we have time:
#  - how modern NN libraries attempt to abstract layers by implementing a layer class
#  - how we can implement a NN using keras
#  - how we can implement a NN using TF

# #### What we'll gloss over

# We won't spend too much time discussing the mechanics of training; backpropogation and gradient descent. This will be covered in the next session.

# ### The operations at the ends of the network
# 
# 
# Neural networks essentially learn to generalize a data distribution, so to train a network we need some input data and the target distribution. On the input end of the network, we typically need to consider getting our data cleaned and organized in such a way that that we can feed it in as a vector (but as a single row in a 2d matrix). On the other end of the network (i.e. the output end), we need a loss function to compute the loss and drive the network towards the goal that we'd like to  optimize for.

# #### The data

# For this tutorial, we will use a standard dataset that does not require much preprocessing. This is to keep things as simple as possible. We can focus on these things in a later session. Input preprocessing tends to be different for each network you create, and loss functions will be discussed when we cover the the training process in greater depth.

# #### Imports



import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# ### Implement a purely linear network
# 
# Neural networks devoid of non-linear funcitons can, at best, model linear functions. For a  

# #### Define a few helper functions



def batch_data_gen(batch_size):
    " Generate inputs and outputs in batches "
    xs = np.random.randint(-10, 10, size=[batch_size, 1]).astype(np.float32)
    ys = (xs ** 2).astype(np.float32)
    return xs, ys

def loss_function(pred, targ):
    return 0.5 * np.sum((pred - targ)**2)

def loss_function_deriv(pred, targ):
    return (pred - targ)

def logistic(x, use=True, deriv=False):
    if use:
        if deriv is True:
            return x * (1 - x)
        return 1./(1. + np.exp(-x))
    else:
        return x if deriv is False else 1 #return 1 if computing the deriv (same as removing)

def predict(x, use_nonlinearity, hidden_weights, output_weights, hidden_bias, output_bias):
    out = logistic(np.dot(x, hidden_weights) + hidden_bias, use=use_nonlinearity)
    pred = np.dot(out, output_weights) + output_bias
    return pred


# #### Fit the data to approximate $f(x) = x^2$ using a linear network
# 
# We can flip the 'use' switch in the logistic function and its derivative to see what happens when we have non-linearities in the network vs not having non-linearities.



lr = 0.0001
num_iterations = 15000
n_input_features = 1
hidden_size = 32
batch_size = 25
samples_seen = 0
progress = list()

use_nonlinearity = True

# Set learnable weight matrices
hidden_weights = np.random.normal(loc=0.0, scale=0.01, size=(n_input_features, hidden_size))
hidden_bias = np.zeros((1, hidden_size))

output_weights = np.random.normal(loc=0.0, scale=0.01, size=(hidden_size, 1))  # a single output!
output_bias = np.zeros((1, 1))

losses = list()

# Train the network
for _ in range(num_iterations):
    xs, ys = batch_data_gen(batch_size)
    samples_seen += 1 * batch_size

    # Forward Pass
    hidden_layer_out = logistic(np.dot(xs, hidden_weights) + hidden_bias, use=use_nonlinearity)
    pred = np.dot(hidden_layer_out, output_weights) + output_bias

    # error
    err = loss_function(pred, ys)
    losses.append(np.sum(err))

    # backprop
    out_delta = loss_function_deriv(pred, ys) # pred - ys
    hidden_delta = np.dot(out_delta, output_weights.T) * logistic(hidden_layer_out, use=use_nonlinearity, deriv=True)

    # gradient descent - use minus-equals due to order of (pred - ys), if switched (ys - pred), use plus-equals
    output_weights -= np.dot(hidden_layer_out.T, out_delta) * lr
    output_bias -= np.sum(out_delta, axis=0) * lr

    hidden_weights -= np.dot(xs.T, hidden_delta) * lr
    hidden_bias -= np.sum(hidden_delta, axis=0) * lr

    if _ % 500 == 0:
        print(f"Current batch loss: {round(losses[-1], 2)} ... samples seen: {samples_seen}")
        x = np.linspace(-9, 9, 30).reshape(-1, 1)
        progress.append(predict(x, use_nonlinearity, hidden_weights, output_weights, hidden_bias, output_bias))

# Print out a cool plot of the learning progress
x = np.linspace(-9, 9, 30).reshape(-1, 1)
fig, axes = plt.subplots(ncols=5, nrows=3, figsize=(15,12))
step = 0
for out, ax, los in zip(progress, axes.flatten(), losses):
    ax.plot(x, x **2, alpha=0.3)
    ax.scatter(x, out)
    ax.set_title('Step: {}\nloss: {}'.format(str(step), str(round(los, 2))))
    step += 1000
plt.tight_layout()


# ## Use a neural network to approximate a regression function that predicts housing prices

# #### Load the dataset
# 
# For this network, we'll use the Boston housing dataset. The goal for this dataset is to learn a model that helps us predict the median market value of the house based on features gathered associated with the property. This is ideal for a simple neural network since we want to use a simple loss function. Since we're predicting a price, we can frame this as a regression problem and use a sum of squares loss function.



boston = load_boston(return_X_y=False)




print(boston.DESCR)


# #### Note for those who are wondering about the race based feature
# This appears to be a controversial feature related to race in the context of a toy dataset where we would like to perform regression. The original purpose of this feature was to rule out race as a confounding factor in the original study. For perspective on both sides of the debate as to whether or not this should be included, you can follow up by reading: https://mail.python.org/pipermail/scikit-learn/2017-July/001683.html
# 
# Since this tutorial does not require this feature, we will exclude it.

# #### Visualize the data



boston.data




boston.target




df = pd.DataFrame(boston.data, columns=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX',
                                        'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO',
                                        'B', 'LSTAT']).drop('B', axis=1)
df.loc[:, 'target'] = boston.target
df.head()


# With any dataset, there is the opportunity to explore the data to find features that are going to likely be the most helpful when training the model. There are a variety of techniques available to do this and it is generally recommended. However, since we're focusing on the network implementation, we'll skip the data exploration for now. 

# #### Define a data batcher
# When we send data through the network, we can send samples through one at a time or in batches. So it's useful to have a function that handles that for us.

# #### Fit



def data_batcher(input_df, batch_size=5):    
    " Create some batches of data for training "

    inputs = input_df.drop('target', axis=1).values
    targets = input_df['target'].values

    # split_data
    X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2, random_state=42)

    # reshape to batch dims
    train_cutoff = (len(X_train) % batch_size)
    test_cutoff = (len(X_test) % batch_size)

    train_x_batches = X_train[: -train_cutoff].reshape(-1, batch_size, X_train.shape[1])
    train_y_batches = y_train[: -train_cutoff].reshape(-1, batch_size, 1)
    test_x_batches  = X_test[ : -test_cutoff].reshape(-1, batch_size, X_test.shape[1])
    test_y_batches  = y_test[ : -test_cutoff].reshape(-1, batch_size, 1)
    
    assert len(train_x_batches) == len(train_y_batches)
    assert len(test_x_batches) == len(test_y_batches)
    
    batched_input_data = zip(train_x_batches, train_y_batches)
    batched_test_data = zip(test_x_batches, test_y_batches)

    return batched_input_data, batched_test_data




lr = 0.000001
n_input_features = 12
hidden_size = 128
batch_size = 25
samples_seen = 0
progress = list()
epochs = 50

epoch_loss = list()
eps = list()

use_nonlinearity = False

# Set learnable weight matrices
hidden_weights = np.random.normal(loc=0.0, scale=0.01, size=(n_input_features, hidden_size))
hidden_bias = np.zeros((1, hidden_size))

output_weights = np.random.normal(loc=0.0, scale=0.01, size=(hidden_size, 1))  # a single output!
output_bias = np.zeros((1, 1))

# Train the network - need epochs to go through data multiple times
for each_epoch in range(epochs):

    losses = list()

    batched_data, _ = data_batcher(df, batch_size=batch_size)
    
    for batch_input, batch_target in batched_data:
        samples_seen += 1 * batch_size

        # Forward Pass
        hidden_layer_out = logistic(np.dot(batch_input, hidden_weights) + hidden_bias, use=use_nonlinearity)
        pred = np.dot(hidden_layer_out, output_weights) + output_bias

        # error
        err = loss_function(pred, batch_target)
        losses.append(np.sum(err))

        # backprop
        out_delta = loss_function_deriv(pred, batch_target) # pred - ys
        hidden_delta = np.dot(out_delta, output_weights.T) * logistic(hidden_layer_out, use=use_nonlinearity, deriv=True)

        # gradient descent - use minus-equals due to order of (pred - ys), if switched (ys - pred), use plus-equals
        output_weights -= np.dot(hidden_layer_out.T, out_delta) * lr
        output_bias -= np.sum(out_delta, axis=0) * lr

        hidden_weights -= np.dot(batch_input.T, hidden_delta) * lr
        hidden_bias -= np.sum(hidden_delta, axis=0) * lr

    if each_epoch % 5 == 0:
        print('batch_loss: {}'.format(loss_function(pred, batch_target)))

    if each_epoch % 1 == 0:
        epoch_loss.append(loss_function(pred, batch_target))
        eps.append(each_epoch)




plt.plot(eps, epoch_loss)
plt.xlabel('step'), plt.ylabel('loss'), plt.title("Loss");
# plt.ylim((0, 50000))
plt.xlim((0, 50))


# #### Root Mean Squared Error



from sklearn.metrics import mean_squared_error #(y_true, y_pred)

# OR

def rmse(target, pred):
    assert len(target) == len(pred)
    return np.sqrt(np.sum(np.square(pred - target)) / float(len(target)))




test_error = list()
residuals = list()

test_X = np.vstack([x[0] for x in list(data_batcher(df, batch_size=batch_size)[1])])
targets = np.vstack([x[1] for x in list(data_batcher(df, batch_size=batch_size)[1])])

predictions = predict(test_X, use_nonlinearity, hidden_weights, output_weights, hidden_bias, output_bias)[:, 0]

print('Root Mean Squared Error: {}'.format(mean_squared_error(targets, predictions)))
print('Our rmse: {}'.format(rmse(targets, predictions)))




plt.hist(np.vstack(predictions)[:, 0]);


# #### Residuals



plt.scatter(np.vstack(targets), np.vstack(targets.reshape(-1, 1)) - np.vstack(predictions))
plt.axhline(0);
plt.axvline(0);


# #### $R^2$ result



r2_score(targets, np.vstack(predictions))


# There is a quite a bit we can do to improve the results we obtained in this tutorial, and we've covered only some of most basic techniques involved with implementing a neural network from scratch.
# 
# Ideas to explore are:
#  - normalizing the data during preprocessing
#  - the consequence of using different types of activation functions
#  - activation function saturation (see optimization notebook)
#  - unwanted variable correlation (linear dependance across columns)
#  

# ## Visualizing the weights of the network



layers = [hidden_weights, output_weights,
          hidden_weights.flatten(), output_weights.flatten()]

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


# ### Implement with Keras



from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf




tf.reset_default_graph()
model = Sequential()
model.add(Dense(32, input_dim=12, activation='relu', use_bias=True))
model.add(Dense(1))
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mse'])




model.summary()


# ### What might it take to assess a model?



# Split data
inputs = df.drop('target', axis=1).values
targets = df['target'].values




model.fit(df.drop('target', axis=1), df['target'], epochs=20);


# # Summary

# Building a model is actually the easy part! If we implement a network from scratch, it can be a bit tedious, however modern deep learning libraries such as keras, and even tensorflow, contain abstractions that make implementing simple models relatively easy.
# 
# The difficult part is figuring out how to frame the problem, which model architecture to use, which of the countless hyperparameters possible to use, and how to determine/when to decide that the performance of the model is good enough. 
# 
# For a short list of examples demonstrating the diversity of metrics available, you can check out: 
# https://www.analyticsvidhya.com/blog/2016/02/7-important-model-evaluation-error-metrics/
# 
# The list doesn't even stop there! In fact, when we approach a unique problem, we may need to use less common evaluation metrics (for example, we use MRR with answerbot) or perhaps one may need to invent an entirely new metric!
# 
# This is what makes the world of machine learning so exciting!

