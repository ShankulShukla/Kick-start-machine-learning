import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
# Function to find majority element
from collections import Counter

class KNN_Classfier:

    def __init__(self,X,y,k):
        self.train = X
        self.train_label = y
        self.k = k

    def k_Euclidean_distance(self,X_test):
        result = []
        for index,row in X_test.iterrows():
            row=np.array(row)
            row.reshape(row.shape[0],1)
            error = np.subtract(self.train,row)
            sqr = error**2
            pre = sqr.sum(axis=1)
            pre = np.c_[pre.index, pre]
            pre = pre[pre[:,1].argsort()]
            result.append(pre[:self.k,0])
        return result

    def majority(self,y):
        max = 0
        maj = None
        dict = Counter(y)
        for i in dict:
             if dict[i] > max:
                 max = dict[i]
                 maj = i
        return maj

    def fit(self,X_test):
        pre = self.k_Euclidean_distance(X_test)
        result = []
        for i in pre:
            result.append(self.majority(self.train_label[i]))
        return result

    def accuracy(self,predict,actual):
        true = 0
        for i,j in zip(predict,actual):
            if i == j:
                true = true+1

        return float(true/len(predict))

import os
df = pd.read_csv(os.getcwd()+"\iris.csv")
x_train,x_test,y_train,y_test = train_test_split(df.loc[:,["SepalLengthCm",  "SepalWidthCm"]],df.loc[:,"Species"],test_size=0.4,random_state=25)
clf = KNN_Classfier(x_train,y_train,3)
pre = clf.fit(x_test)
print("Accuracy-",clf.accuracy(pre,y_test))

from matplotlib.colors import ListedColormap
# contour differentiation line
color = ['red','cyan','magenta','blue','black']
marker  = ['*','-','+','o','^']
y = []
for i in df.loc[:,"Species"]:
    if i == "Iris-versicolor":
        y.append(2)
    elif i == "Iris-setosa":
        y.append(1)
    else:
        y.append(3)
y= np.array(y)
cmap = ListedColormap(color[:len(np.unique(y))])
X=np.vstack((x_train,x_test))
y=np.hstack((y_train,y_test))
x_max,x_min=X[:,0].max()+1 ,X[:,0].min()-1
y_max,y_min=X[:,1].max()+1 ,X[:,1].min()-1
xx1,xx2=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
Z=clf.fit(pd.DataFrame(np.array([xx1.ravel(),xx2.ravel()]).T))
y = []
for i in Z:
    if i == "Iris-versicolor":
        y.append(2)
    elif i == "Iris-setosa":
        y.append(1)
    else:
        y.append(3)
Z =np.array(y)
Z=Z.reshape(xx1.shape)
import matplotlib.pyplot as plt
plt.contourf(xx1,xx2,Z,alpha=0.3,cmap=cmap)
for i in np.unique(y_test):
    index = i == y_test
    plt.scatter(x_test['SepalLengthCm'][index],x_test['SepalWidthCm'][index],label=i)
plt.legend(loc='best')
plt.show()
