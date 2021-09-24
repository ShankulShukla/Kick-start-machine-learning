
# svm- maximum margin classifier other failed to do so (linear regression, logistic etc) 
# main idea is to maximize the minimum distance of any data point from the best fit line using support vectors

from sklearn.datasets import make_classification

from matplotlib import pyplot as plt
import numpy as np


# store this x, y in text file
x, y = make_classification(n_classes=2, n_samples=500, n_clusters_per_class=1, random_state = 1, n_features=2, n_informative=2, n_redundant=0)

# mapping label

y[y==0] = -1


# plotting the data


plt.scatter(x[:,0], x[:,1], c=y)
plt.plot()

# svm model
class SVM:
    def __init__(self, C=1.0):
        # svm model
        self.C = C
        self.w = 0
        self.b = 0
        
    def hingeloss(self, W, b, X, y):
        loss = 0.
        loss +=np.dot(W, W.T)/2
        
        # going through all training samples
        for i in range(X.shape[0]):
            ti = y[i] * (np.dot(W, X[i].T)+b)
            loss += self.C * max(0, 1-ti)
        return loss[0][0]
    
    def fit(self, X, y, batch_size=64, lr=0.01, epochs=50):
        num_features = X.shape[1]
        num_samples = X.shape[0]
        W = np.zeros((1, num_features))
        b = 0
        losses = []
        for i in range(epochs):
            loss = self.hingeloss(W, b, X, y)
            losses.append(loss)
            for start in range(0, num_samples, batch_size):
                # gradient of weight
                d_w = 0
                # gradient of bias
                d_b = 0
                for j in range(start, start+batch_size):
                    if j < num_samples:
                        ti = y[j] * (np.dot(W, X[j].T)+b)
                        if ti <= 1:
                            d_w += self.C * y[j] * X[j]
                            d_b += self.C * y[j]
                # updating the weights
                W -= lr*(W-(d_w/batch_size))
                b += lr*(d_b/batch_size)
                            
        self.W = W
        self.b = b
        return losses


# creating svm model

svmModel = SVM(1000)


# fitting the model
loss = svmModel.fit(x, y, lr = .0001,epochs=1000)



# training loss
print("training loss -",loss)

# plotting the svm model
X1 = np.linspace(-2.5,3,10)
X2 = svmModel.b + svmModel.W[0][1]*X1
plt.scatter(x[:, 0], x[:, 1], c=y, cmap= plt.cm.Accent)
plt.plot(X1, X2, c='red')




def compute_cost(W, X, Y):
    # calculate hinge loss
    N = X.shape[0]
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0  # equivalent to max(0, distance)
    hinge_loss = 1 * (np.sum(distances) / N)

    # calculate cost
    cost = 1 / 2 * np.dot(W, W) + hinge_loss
    return cost

print(compute_cost([0., 0.], x, y))

