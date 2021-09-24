# inspired from sebastian raschka post
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
class LDA:

    def __init__(self,d):

        self.discriminant = d

    def fit(self,X,y):

        # scale the features
        sub = np.subtract(X, X.mean(axis=0))
        X_std = sub / X.std(axis=0)

        # first find the scatter matrices
        # 1. with in scatter matrix
        labels = np.unique(y)
        feature_mean = [np.mean(X_std[y == i,:],axis=0) for i in labels]
        self.scatterW = np.zeros([X_std.shape[1],X_std.shape[1]])

        for i,mean in zip(labels,feature_mean):
            # scatter = np.zeros([X_std.shape[1],X_std.shape[1]])
            # for row in X_std[y==i]:
            #     sub = np.subtract(row,mean)
            #     scatter = scatter + sub.dot(sub.T)
            # scatter = (1/len(X_std[y==i]))*scatter
            scatter = np.cov(X_std[y==i].T)
            self.scatterW += scatter

        # 2. in between scatter matrix

        self.scatterB = np.zeros([X_std.shape[1], X_std.shape[1]])
        total_mean = np.mean(X_std,axis=0)
        for i,mean in zip(labels,feature_mean):
            n = X_std[y==i,:].shape[0]
            mean = mean.reshape(13,1)
            total_mean = total_mean.reshape(13, 1)
            sub = np.subtract(mean,total_mean)
            self.scatterB += n*sub.dot(sub.T)
        # creating the convariance

        sigma = np.dot(np.linalg.inv(self.scatterW),self.scatterB)
        self.e_val,self.e_vec = np.linalg.eig(sigma)
        eigen_pairs = [(self.e_val[i], self.e_vec[:, i]) for i in range(len(self.e_vec))]
        eigen_pairs.sort(key=lambda x: x[0], reverse=True)
        self.w_ = np.hstack((eigen_pairs[i][1][:,np.newaxis]) for i in range(self.discriminant))

    def visualize(self):
        total = self.e_val.sum()
        e_val = [i / total for i in sorted(self.e_val, reverse=True)]
        c_e_val = np.cumsum(e_val)
        plt.plot(range(1, len(self.e_val) + 1), e_val, marker='o', color='yellow', label='eigen value')
        plt.plot(range(1, len(self.e_val) + 1), c_e_val, marker='o', color='red', label='cummulative eigen value')
        plt.xlabel('number of principal components')
        plt.ylabel('expected variance ratio')
        plt.legend()
        plt.show()

    def transform(self, X):
        sub = np.subtract(X, X.mean(axis=0))
        X_std = sub / X.std(axis=0)
        Z = np.dot(X_std, self.w_)
        return Z

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None,skiprows=1).values
X = df[:,1:]
y=df[:,0]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=25)
pc = LDA(2)
pc.fit(x_train,y_train)
pc.visualize()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
print('Accuracy without lda-',clf.score(x_test,y_test))
x_train_lda = pc.transform(x_train)
x_test_lda = pc.transform(x_test)
clf.fit(x_train_lda,y_train)
print('Accuracy with lda-',clf.score(x_test_lda,y_test))