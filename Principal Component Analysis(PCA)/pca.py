import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

class PCA:
    def __init__(self,p_comp):

        self.principal_component = p_comp


    def fit(self,X):

        sub = np.subtract(X,X.mean(axis=0))
        X_std = sub/X.std(axis=0)
        #covariance matrix
        cov = np.cov(X_std.T)
        self.e_val, self.e_vec = np.linalg.eig(cov)
        eigen_pairs = [(self.e_val[i],self.e_vec[:,i]) for i in range(len(self.e_vec))]
        eigen_pairs.sort(key=lambda x:x[0],reverse=True)
        self.w = np.hstack((eigen_pairs[i][1][:,np.newaxis]) for i in range(self.principal_component))


    def visualize(self):
        total = self.e_val.sum()
        e_val = [ i/total for i in sorted(self.e_val,reverse=True) ]
        c_e_val = np.cumsum(e_val)
        plt.plot(range(1,len(self.e_val)+1), e_val,marker='o',color='yellow',label='eigen value')
        plt.plot(range(1,len(self.e_val)+1), c_e_val, marker='o', color='red',label='cummulative eigen value')
        plt.xlabel('number of principal components')
        plt.ylabel('expected variance ratio')
        plt.legend()
        plt.show()

    def transform(self,X):
        sub = np.subtract(X, X.mean(axis=0))
        X_std = sub / X.std(axis=0)
        Z = np.dot( X_std,self.w)
        return Z

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None,skiprows=1).values
X = df[:,1:]
y=df[:,0]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=25)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
pc = PCA(2)
# 2 represent the number of principal components
pc.fit(x_train)
pc.visualize()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
print('Accuracy without pca-',clf.score(x_test,y_test))
x_train_pca = pc.transform(x_train)
x_test_pca = pc.transform(x_test)
clf.fit(x_train_pca,y_train)
print('Accuracy with pca-',clf.score(x_test_pca,y_test))
