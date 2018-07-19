#can download the mnist dataset and convert into .npz file to extract the images and corresponding labels
import numpy as np
import matplotlib.pyplot as plt
class NN_classifier:

    def __init__(self, n_iter, l_rate, b_size, lr, n_hidden, o_unit):

        self.n_iteration = n_iter
        self.learning_rate = l_rate
        self.batch_size = b_size
        self.l2 = lr
        self.hidden_unit = n_hidden
        self.output_unit = o_unit

    def fit(self,X,y):
        # we have to define the weights (including the bias term's weight) of two layers i.e, from input to hidden layer and from hidden to output layer
        self.weight_h = np.random.rand(X.shape[1],self.hidden_unit)
        self.weight_h_bias = np.zeros((1,self.hidden_unit))
        self.weight_o = np.random.rand(self.hidden_unit,self.output_unit)
        self.weight_o_bias = np.zeros((1,self.output_unit))

        x_train, y_train, x_val, y_val = X[:-125], y[:-125], X[-125:], y[-125:]
        y_train = self.one_hot(y_train)
        self.error = []
        for i in range(self.n_iteration):
            for batch in range(0,x_train.shape[0]+((x_train.shape[0]+self.batch_size)//2),self.batch_size):
                x_batch, y_batch = x_train[batch: batch+self.batch_size], y_train[batch: batch+self.batch_size]
                act_o, act_h = self.forward_propagation(x_batch)
                self.back_propagation(x_batch, y_batch, act_o, act_h)
            pred, _ = self.forward_propagation(x_train)
            e = self.cost_function(y_train, pred)
            pred = clf.predict(x_val)
            self.error.append(e)
            acc = np.sum(y_val == pred).astype(np.float) / x_val.shape[0]
            print('Accuracy for iteration {} on validation set- {}%'.format(i,acc))

    def one_hot(self,y):
        classes = len(np.unique(y))
        y_ = np.zeros((y.shape[0], classes))
        for n, i in enumerate(y):
            y_[n, i] = 1
        return y_

    def cost_function(self, y, pred):
        J = -(np.sum(y* np.log(pred) + (1-y)* np.log(1-pred)))
        J += self.l2*(np.sum(self.weight_h**2)+np.sum(self.weight_o**2))
        return J

    def predict(self, X):
        pred, _ = self.forward_propagation(X)
        y_pred = np.argmax(pred, axis=1)
        return y_pred

    def sigmoid(self, z):
        # for the exploding gradients
        z = np.clip(z, -250, 250)
        return 1./(1.+np.exp(-z))

    def forward_propagation(self, X):
        act_i = X
        z_h = np.dot(act_i, self.weight_h) + self.weight_h_bias
        act_h = self.sigmoid(z_h)
        z_o = np.dot(act_h, self.weight_o) + self.weight_o_bias
        act_o = self.sigmoid(z_o)
        return act_o, act_h

    def back_propagation(self, X, y, act_o, act_h):
        # we are not going to calculate the delta at the input layer
        delta_o = np.subtract(act_o, y)
        diff_sigmoid = act_h*(1-act_h)
        delta_h = np.dot(delta_o, self.weight_o.T) * diff_sigmoid
        par_derivative_h = np.dot(X.T, delta_h)
        par_derivative_o = np.dot(act_h.T, delta_o)
        # weight update part
        self.weight_h -= self.learning_rate*(par_derivative_h + self.l2 * self.weight_h)
        self.weight_h_bias -= self.learning_rate*(delta_h.sum(axis=0))
        self.weight_o -= self.learning_rate*(par_derivative_o + self.l2 * self.weight_o)
        self.weight_o_bias -= self.learning_rate*(delta_o.sum(axis=0))


data = np.load('mnist.npz')
x_train, x_test, y_train, y_test = data['x_train'], data['x_test'], data['y_train'], data['y_test']
x_train = ((x_train/255.0) - 0.5)*2
clf = NN_classifier(n_iter=200,l_rate=0.005,b_size=100, lr=0.1,n_hidden=100,o_unit=10)
clf.fit(x_train, y_train)
plt.plot(range(len(clf.error)),clf.error)
plt.xlabel('n_iterations')
plt.ylabel('Error')
plt.show()
pred = clf.predict(x_test)
acc = np.sum(y_test == pred).astype(np.float)/x_test.shape[0]
print('Accuracy (on test set)- ',acc*100)