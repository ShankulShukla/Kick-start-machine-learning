
import numpy as np
import matplotlib.pyplot as plt
import random


#         Adam: This is the adam optimizer that build upon adadelta and RMSProp
#         model: The model we want to optimize the parameter on
#         x: the feature of my dataset
#         y: the continous value (target)
#         learning_rate: the amount of learning we want to happen at each time step (default is 0.1 and will be updated by the optimizer)
#         beta_1: this is the first decaying average with proposed default value of 0.9 (deep learning purposes)
#         beta_2: this is the second decaying average with proposed default value of 0.999 (deep learning purposes)
#         epsilon: a variable for numerical stability during the division
#         epoch: the number of gd round we want to do before stopping the optimization


class Adam:
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
    
    def train(self, X, Y, lr=0.01, beta_1=0.9, beta_2=0.999, epoch = 500, eps=0.000001):
        
        # Variable initialization for weight and bias
        m_w, m_b = 0, 0
        v_w, v_b = 0, 0
        
        # time steps 
        t = 1
        
        for _ in range(epoch):
            # randomly sample data
            idx = random.randint(0,len(X)-1)
            x, y = X[idx], Y[idx]
            # gradients
            g_w, g_b = self.derivate(x, y)
                
            # update forst moment m parameter for weight and bias
            m_w = beta_1*m_w + (1-beta_1)*g_w
            m_b = beta_1*m_b + (1-beta_1)*g_b
                
            # update second moment v parameter for weight and bias
            v_w = beta_2*v_w + (1-beta_2)*(g_w**2)
            v_b = beta_2*v_b + (1-beta_2)*(g_b**2)
                
            # bias-correction for m parameter
            m_corr_w = m_w/(1-(beta_1**t))
            m_corr_b = m_b/(1-(beta_1**t))
                
            # bias-correction for v parameter
            v_corr_w = v_w/(1-(beta_2**t))
            v_corr_b = v_b/(1-(beta_2**t))
                
            # updating the parameter
            self.weight -= (lr / (np.sqrt(v_corr_w) + eps))*m_corr_w
            self.bias -= (lr / (np.sqrt(v_corr_b) + eps))*m_corr_b
            t=t+1
                
    def predict(self, x):
        return self.weight*x + self.bias



model = Adam()


# result should be approximate intercept = 1 and slope = 4
x = [-1,0,1.6,2,3.8,4,5,6.1,7,8,9]
y = [-3,1,5,9,13,17,21,25,29,33,37]


model.train(x, y)

# plotting the result from weights trained by optimiser
plt.scatter(x,y)
plt.ylim(-5, 45)
x_test = np.linspace(-1., 9., num=20)
y_pred = []
for i in x_test:
    y_pred.append(model.predict(i)[0])
plt.plot(x_test, y_pred)
plt.show()

