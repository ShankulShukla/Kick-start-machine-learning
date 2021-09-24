
# cnn architecture used in this implementation -
# conv layer -> max-pool layer -> Fully connected layer -> softmax

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# using mnist data to create a number classifier
df = pd.read_csv(os.getcwd()+r"\mnist_test.csv")

print(df.head())

# visualizing the image
plt.imshow(df.iloc[100,1:].values.reshape(28,28), cmap='gray')
plt.show()


# convolution layer
class ConvLayer:
    # filter_size: a 2-elements tuple (width, height ) of the filter. 
    # num_filters: an integer specifies number of filter in the layer.
    # stride =1
    # xavier_init : can choose weight inilization
    def __init__(self, num_filters, filter_size, xavier_init = True):
        self.num_filters = num_filters
        self.filter_size = filter_size
        
        if xavier_init:
            # Initialize weights according `Xavier normal` distribution. With mean = 0, std = sqrt(2 / (num_input + num_output))
            weight_shape = (num_filters, filter_size, filter_size)
            self.filter_weight = np.random.normal(0, np.sqrt(2 / (filter_size**2 + num_filters)), weight_shape)
        else:
            # Initialize weights according standard normal distribution
            self.filter_weight = np.random.randn(num_filters, filter_size, filter_size)/(filter_size ** 2)
    
    # get patch of input for convolution with filter
    def extractWindow(self, input):
        height, width = input.shape
        for x in range(height - self.filter_size + 1):
            for y in range(width - self.filter_size + 1):
                window = input[x: (x+ self.filter_size), y: (y+ self.filter_size)]
                yield window, x, y
    
    def forward(self, X):
        #Forward propagation of the convolutional layer.
        #If padding is 'SAME', we must solve this equation to find appropriate number p:
        #    oH = (iH - fH + 2p)/s + 1
        #    oW = (iW - fW + 2p)/s + 1
        
        height, width = X.shape
        output = np.zeros((height - self.filter_size + 1, width - self.filter_size + 1, self.num_filters))
        for window, x, y in self.extractWindow(X):
            output[x,y] = np.sum(window * self.filter_weight, axis = (1,2))
        return output
    
    def backward(self, d_prev, X, lr):
        #Backward propagation of the convolutional layer.
        d_filter = np.zeros(self.filter_weight.shape)
        for window, x, y in self.extractWindow(X):
            for k in range(self.num_filters):
                d_filter += window*d_prev[x, y, k]
        
        # update filter weights
        self.filter_weight -= lr*d_filter
        return d_filter
                


# max-pool layer
class MaxPoolLayer:
    
    def __init__(self, filter_size):
        self.filter_size = filter_size
        
     # get patch of input for max pool 
    def extractWindow(self, input):
        pooled_height = input.shape[0]//self.filter_size 
        pooled_width = input.shape[1]//self.filter_size
        self.inp = input
        
        for x in range(pooled_height):
            for y in range(pooled_width):
                window = input[(x*self.filter_size): (x*self.filter_size+ self.filter_size), (y*self.filter_size): (y*self.filter_size + self.filter_size)]
                yield window, x, y
                
    def forward(self, X):
        #Pooling layer forward propagation. Through this layer, the input dimension will reduce:
        #  oH = floor((iH - fH)/stride + 1)
        #  oW = floor((iW - fW)/stride + 1)
        height, width, num_filters = X.shape
        output = np.zeros((height//self.filter_size, width//self.filter_size, num_filters))
        
        for window, x, y in self.extractWindow(X):
            output[x,y] = np.amax(window, axis=(0,1))
        return output
    
    def backward(self, d_prev, X):
        d_maxpool = np.zeros(self.inp.shape)
        for window, x, y in self.extractWindow(self.inp):
            height, width, num_filters = window.shape
            max_val = np.amax(window, axis=(0,1))
            
            for i in range(height):
                for j in range(width):
                    for k in range(num_filters):
                        if window[i, j, k] == max_val[k]:
                            d_maxpool[x*self.filter_size + i, y*self.filter_size +j, k] = d_prev[x, y, k]
        return d_maxpool
                            


# softmax layere

class SoftmaxLayer:
    # combination of fuly connected layer and softmax layer
    # Softmax activation function. Use at the output layer.
    #    g(z) = e^z / sum(e^z)
    def __init__(self, input_size, output_size):
        self.weight = np.random.randn(input_size, output_size)/input_size
        self.bias = np.zeros(output_size)
    
    def forward(self, X):
        self.inp_shape = X.shape
        X_flat = X.flatten()
        self.inp_flat = X_flat
        
        output = np.dot(X_flat, self.weight) + self.bias
        self.output = output
        
        # softmax transformation
        exp = np.exp(output)
        return exp/np.sum(exp, axis = 0)
    
    def backward(self, d_prev, X, lr):
        #Performs a backward pass of the softmax layer.
        #Returns the loss gradient for this layers inputs.
        for i, grad in enumerate(d_prev):
            if grad == 0:
                continue
            exponent = np.exp(self.output)
            exponent_sum = np.sum(exponent)
            
            d_out = -exponent[i]*exponent/(exponent_sum ** 2)
            d_out[i] = exponent[i]*(exponent_sum- exponent[i])/(exponent_sum ** 2)
            
            # weight gradient
            d_w = self.inp_flat
            d_b = 1
            d_w_inp = self.weight
            
            # loss gradient
            d_l = grad * d_out

            # loss gradieint w.r.t softmax layer 
            d_l_w = np.matmul(d_w[np.newaxis].T,  d_l[np.newaxis])
            d_l_b = d_l * d_b
            d_l_inp = np.matmul(d_w_inp, d_l)
            #print(self.weight.shape, lr, d_l_w.shape)
            # update weight and bias
            self.weight -= lr * d_l_w
            self.bias -= lr * d_l_b
        
            return d_l_inp.reshape(self.inp_shape)
            
            
            
        


# importing dataset
X,  y = df.iloc[:,1:].values, df['label'].values

X = X.reshape(X.shape[0],28,28)

# Ih this implementation we will be using 5000 images-
# 1000 test
# 4000 train

# diving dataset into approx train - 85% and test 15%
x_train, y_train, x_test, y_test = X[:4000,:], y[:4000], X[4000:5000,:], y[4000:5000]


# Normalize Dataset
x_train = x_train / 255.0
x_test = x_test / 255.0

# creating cnn arhitecture
num_class = 10

conv_layer = ConvLayer(64, 3)      # 28x28x1 -> 26x26x64
max_pool_layer = MaxPoolLayer(2)   # 26x26x64 -> 13x13x64

# This also work as dense layer
softmax_layer = SoftmaxLayer(13*13*64, num_class)


# forward propagation through network
def CNNForward(X, y):
    outcnn = conv_layer.forward(X)
    outpool = max_pool_layer.forward(outcnn)
    out_pred = softmax_layer.forward(outpool)

    # Cross entropy loss
    loss = -np.log(out_pred[y])
    pred = 1 if np.argmax(out_pred) == y else 0
    
    return out_pred, loss, pred



# back propagation through the network
def CNNBackward(grad_loss, X, lr):
    grad_softmax = softmax_layer.backward(grad_loss, X, lr)
    grad_pool = max_pool_layer.backward(grad_softmax, X)
    grad_conv = conv_layer.backward(grad_pool, X, lr)


# training CNN
def training(X, y, lr=0.01):
    # forward propagation
    out_pred, loss, pred = CNNForward(X, y)
    
    grad_loss = np.zeros(num_class)
    grad_loss[y] = -1/out_pred[y]
    
    # backward propagation
    CNNBackward(grad_loss, X, lr)
    
    return loss, pred


# hyperparameters
epochs = 3
learning_rate = 0.005


# training the CNN model
for epoch in range(epochs):
    loss = 0
    rightClassify = 0
    for i, (x, y) in enumerate(zip(x_train, y_train)):
        _loss, _pred = training(x, y, lr =learning_rate)
        loss += _loss
        rightClassify += _pred
        if i%100 ==0:
            print("Epoch - {} Step- {} Average loss- {}".format(epoch, i+1, loss/100))

print("Accuracy after training - {}".format(rightClassify/x_train.shape[0]))
