# gaussian kernel
# inspired by sebastian raschka post and andrew ng lectures
from scipy.spatial.distance import pdist,squareform
import numpy as np
from scipy import exp
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
class Kernel_PCA:

    def __init__(self,g,n):

        self.gamma = g
        self.components = n

    def fit(self,X):

        dist = pdist(X,'sqeuclidean')
        sq_dist = squareform(dist)
        K = exp(-self.gamma * sq_dist)
        # standardize the kernel
        one_n = np.ones((K.shape[0],K.shape[0]))/K.shape[0]
        K_std = K - one_n.dot(K) - K.dot(one_n) +one_n.dot(K).dot(one_n)
        self.e_val , self.e_vec = eigh(K_std)
        e_pair = [(self.e_val[i],self.e_vec[:,i]) for i in range(len(self.e_val))]
        e_pair.sort(key=lambda x: x[0], reverse=True)
        self.w_ = np.hstack((e_pair[i][1][:,np.newaxis]) for i in range(self.components))
        return self.w_

    def transform(self,x,X):

        e_val, e_vec = self.e_val[::-1], self.e_vec[:, ::-1]
        alpha = np.column_stack((e_vec[:, i]) for i in range(self.components))
        lambdas = [e_val[i] for i in range(self.components)]
        result = []
        for x_ in x:
            pdist = np.array([np.sum((x_ - i) ** 2) for i in X])
            K = exp(-self.gamma * pdist)
            result.append(np.array(K.dot(alpha / lambdas)))

        return np.array(result)

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None,skiprows=1).values
X = df[:,1:]
y = df[:,0]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=25)
pc = Kernel_PCA(0.00001,15)
x_train_pca = pc.fit(x_train)
plt.scatter(x_train_pca[y_train==1,0],x_train_pca[y_train==1,1])
plt.scatter(x_train_pca[y_train==2,0],x_train_pca[y_train==2,1])
plt.scatter(x_train_pca[y_train==3,0],x_train_pca[y_train==3,1])
plt.show()
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(x_train,y_train)
print('Accuracy without pca-',clf.score(x_test,y_test))
x_test_pca = pc.transform(x_test,x_train)
clf.fit(x_train_pca,y_train)
print('Accuracy with kernel-pca-',clf.score(x_test_pca,y_test))