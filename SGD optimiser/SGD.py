
import numpy as np
import matplotlib.pyplot as plt

# SGD model
class SGD:
    #   sgd: will estimate the parameters w0 and w1 (here it uses least square cost function)
    #   model: the model we are trying to optimize using sgd
    #   xs: all point on the plane
    #   ys: all response on the plane
    #   learning_rate: the learning rate for the step that weights update will take
    #   max_num_iteration: the number of iteration before we stop updating

    def __init__(self):
        # Linear Model with two weights: bias (intercept) and weight (slope)
        self.weight = np.random.uniform(0,1,1)
        self.bias = 0.0
    
    def derivate(self, x, y):
        y_pred = self.weight*x + self.bias
        # dx_w: partial derivative of the weight 
        dx_w = 2*x*(y_pred - y)
        # dx_b: partial derivative of bias
        dx_b = 2*(y_pred - y)
        return dx_w, dx_b
    
    def train(self, X, Y, lr = 0.01, epoch = 10):
        for i in range(epoch):
            for x, y in zip(X, Y):
                dx_w, dx_b = self.derivate(x, y)
                self.weight -= lr*dx_w
                self.bias -= lr*dx_b
                
    def predict(self, x):
        return self.weight * x + self.bias 


# Creating sgd model 
model = SGD()


# result should be approximate intercept = 1 and slope = 4
x = [-1,0,1.6,2,3.8,4,5,6.1,7,8,9]
y = [-3,1,5,9,13,17,21,25,29,33,37]


# model training

model.train(x,y)


# plotting the model
plt.scatter(x,y)
plt.ylim(-5, 45)
x_test = np.linspace(-1., 9., num=20)
y_pred = []
for i in x_test:
    y_pred.append(model.predict(i)[0])
plt.plot(x_test, y_pred)
plt.show()

