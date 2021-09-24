# logistic regression with regularization
import numpy as np
import os
import pandas as pd

class logistic_classification:
    def __init__(self, lr, rs, n_iter, rl):
        self.learning_rate = lr
        self.random_state = rs
        self.n_iteration = n_iter
        self.regularization =rl

    def fit(self, X, y):
        r = np.random.RandomState(self.random_state)
        self.weight = r.normal(loc=0.0,scale=0.01,size=(X.shape[1]+1,1))
        # adding the bias term in the weight vector only ..just by stacking that in the input X
        X = np.hstack((np.ones((X.shape[0],1)),X))
        self.errors = []
        for _ in range(self.n_iteration):
            t = np.dot(X,self.weight)
            h = self.sigmoid(t)
            self.weight -= self.learning_rate*(np.dot(X.T,np.subtract(h,y)))
            init = self.weight[1]
            self.weight -=  self.learning_rate*(self.regularization*self.weight)
            self.weight[1] = init
            error = -np.sum(y*np.log(h)+(1-y)*np.log(1-h)) + self.regularization*np.sum(self.weight**2)
            self.errors.append(error)
            print(error)


    def sigmoid(self, x):
        return 1./(1.+np.exp(-1*x))

    def perdict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        pred = self.sigmoid(np.dot(X, self.weight))
        return np.where(pred>=0.5,1,0)

df = pd.read_csv(os.getcwd()+'/titanic_dataset.csv')

# Applying preprocessing
df=df.drop(['Ticket','Cabin','PassengerId'],axis=1)
#encoding the sex and embarked as in integers form
df['Sex'] = df['Sex'].map({'male':1,'female':0})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
df['tot_par'] = df['SibSp']+df['Parch']

#we can drop the SibSp and Parch column
df = df.drop(['SibSp','Parch'],axis=1)

# encoding the titles
df['titles'] = df.Name.str.extract(' ([\w]+)\.',expand = False)
df['titles']=df['titles'].map({'Miss':4,'Mrs':3,'Mr':2,'Master':1})
df['titles'] = df['titles'].fillna(0)
df = df.drop('Name',axis=1)
df = df.dropna(subset=['Age','Embarked'])

#normalize the fare
df['Fare'] = (df['Fare'] - df['Fare'].mean()) / (df['Fare'].std())
df['Age'] = (df['Age'] - df['Age'].mean()) / (df['Age'].std())

df = df.values
X, y = df[:, 1:], df[:, 0].reshape(-1, 1)
clf = logistic_classification(lr=0.00005,rs=25,n_iter=100,rl=0.01)
clf.fit(X,y)
pred = clf.perdict(X)
acc = (pred == y).sum()/y.shape[0]
print('accuracy on the train set - ' , acc)
