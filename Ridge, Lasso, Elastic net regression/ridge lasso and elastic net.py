
# Linear regression suffers from overfitting drastically, to avoid this we use regularisation, in this we penalize our model for doing something wrong
# Regularization = loss function + penalty(regularization)

# Ridge regression- apply l2 regularization

# Lasso regression- apply l1 regularization

# In both lambda is used as strength of regularization

# Elastic net applies both l1 and l2 regularization,
# we can control its behaviour of regularisation w.r.t its procilivity to which regularisation more using l1 ratio
 
# we will be using gradient decent to find weights

import numpy as np
import matplotlib.pyplot as plt


# ridge regression
class RidgeRegression:
    
    def __init__(self, lr=0.01, alpha =1.0, epoch=20):
        self.lr = lr
        # it is lambda parameter 
        self.alpha = alpha
        self.epoch = epoch
    
    def train(self, X, Y):
        self.weight = np.random.uniform(0,1, X.shape[1])
        self.bias = 0.
        
        for i in range(self.epoch):
            y_pred = np.dot(X, self.weight) + self.bias
            
            d_weight = (-(2*(X.T).dot(Y - y_pred)) + (2*self.alpha*self.weight))/X.shape[0]
            d_bias = -2*np.sum(Y - y_pred)/X.shape[0]
            
            # updating weights and bias
            self.weight -= self.lr*d_weight
            self.bias -= self.lr*d_bias
    
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias



# lasso regression
class LassoRegression:
    
    def __init__(self, lr=0.01, alpha =1.0, epoch=20):
        self.lr = lr
        # it is lambda parameter 
        self.alpha = alpha
        self.epoch = epoch
    
    def train(self, X, Y):
        self.weight = np.random.uniform(0,1, X.shape[1])
        self.bias = 0.
        
        for i in range(self.epoch):
            y_pred = np.dot(X, self.weight) + self.bias
            
            d_weight = (-(2*(X.T).dot(Y - y_pred)) + (self.alpha))/X.shape[0]
            d_bias = -2*np.sum(Y - y_pred)/X.shape[0]
            
            # updating weights and bias
            self.weight -= self.lr*d_weight
            self.bias -= self.lr*d_bias
    
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias


# Elastic net
class ElasticNet:
    
    def __init__(self, lr=0.01, alpha =1.0, epoch=20, l1_ratio=0.4):
        self.lr = lr
        # it is lambda parameter 
        self.alpha = alpha
        self.epoch = epoch
        self.l1_ratio = l1_ratio
    
    def train(self, X, Y):
        self.weight = np.random.uniform(0,1, (X.shape[1]))
        self.bias = 0.
        
        for i in range(self.epoch):
            y_pred = np.dot(X, self.weight) + self.bias
            
            d_weight = (-(2*(X.T).dot(Y - y_pred)) + (self.alpha*self.l1_ratio) + (self.alpha*(1-self.l1_ratio)*self.weight))/X.shape[0]
            d_bias = -2*np.sum(Y - y_pred)/X.shape[0]
            
            # updating weights and bias
            self.weight -= self.lr*d_weight
            self.bias -= self.lr*d_bias
    
    def predict(self, X):
        return np.dot(X, self.weight) + self.bias


# result should be approximate intercept = 1 and slope = 4
x = np.array([-1,-0.63,1.6,2.6,3.8,4.056,5.5,6.1,7,8.9,9.1]).reshape(-1,1)
y = np.array([-3,1,5,9,13,17,21,25,29,33,37])


# create model
model1 = RidgeRegression()
model2 = LassoRegression()
model3 = ElasticNet()


# training model
model1.train(x,y)
model2.train(x,y)
model3.train(x,y)


# Ridge regression
plt.scatter(x,y)
plt.ylim(-5, 45)
plt.title("Ridge Regression")
x_test = np.linspace(-1., 9., num=20)
y_pred = []
for i in x_test:
    y_pred.append(model1.predict(i)[0])
plt.plot(x_test, y_pred)
plt.show()


# Lasso regression
plt.scatter(x,y)
plt.ylim(-5, 45)
plt.title("Lasso Regression")
x_test = np.linspace(-1., 9., num=20)
y_pred = []
for i in x_test:
    y_pred.append(model2.predict(i)[0])
plt.plot(x_test, y_pred)
plt.show()


# Elastic net
plt.scatter(x,y)
plt.ylim(-5, 45)
plt.title("Elastic net")
x_test = np.linspace(-1., 9., num=20)
y_pred = []
for i in x_test:
    y_pred.append(model3.predict(i)[0])
plt.plot(x_test, y_pred)
plt.show()

