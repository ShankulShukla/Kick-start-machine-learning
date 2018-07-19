import numpy as np
import matplotlib.pyplot as plt

class linear_regression:

    def __init__(self, lr, n_iter, r_state):
        self.learning_rate = lr
        self.n_iter = n_iter
        self.random_state = r_state

    def fit(self, X, y):
        rs = np.random.RandomState(self.random_state)
        self.weight = rs.normal(loc=0.0,scale=0.01,size=((X.shape[1],1)))
        for _ in range(self.n_iter):
            h = np.dot(X, self.weight)
            err = np.sum((h - y)**2)/X.shape[0]
            print('error - ', err)
            # can visualize the error also
            # gradient descent
            gra = np.dot(X.T,(h-y))
            self.weight -= self.learning_rate * gra

    def predict(self, X):
        return np.dot(X, self.weight)

# creating linear data with gaussian noise
noise=np.random.normal(0,1,100)
X = np.linspace(0, 5, 100)
Y = 3.6*X + 4.2
Y = Y + noise
plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
X = X.reshape(X.shape[0],1)
Y = Y.reshape(Y.shape[0],1)
clf = linear_regression(lr=0.0005,n_iter=20,r_state=25)
clf.fit(X,Y)
print('plotting the predicted curve to the scatter plot-')
plt.scatter(X,Y)
plt.xlabel('X')
plt.ylabel('Y')
print(clf.weight)
plt.plot(X,clf.predict(X),color='red')
plt.show()
