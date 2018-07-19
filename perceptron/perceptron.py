import numpy as np
import os
import pandas as pd
class perceptron:

    def __init__(self, lr, rs, n_iter):
        self.learning_rate = lr
        self.random_state = rs
        self.n_iteration = n_iter

    def fit(self, X, y):
        r = np.random.RandomState(self.random_state)
        self.weight = r.normal(loc=0.0,scale=0.001,size=(X.shape[1]+1))
        # adding the bias term in the weight vector only ..just by stacking that in the input X
        X = np.hstack((np.ones((X.shape[0],1)),X))
        self.errors = []
        for _ in range(self.n_iteration):
            for i in range(len(X)):
                h = np.dot(X[i],self.weight)
                self.weight -= self.learning_rate*(np.dot(X[i],h-y[i]))
            # print(np.sum(h-y))
            self.errors.append(np.sum(h-y))

    def perdict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        pred = np.dot(X, self.weight)
        return np.where(pred>0.0,1,0)

os.chdir('C:/Users\shankul\Desktop/all')
df = pd.read_csv('titanic_dataset.csv')
df=df.drop(['Ticket','Cabin','PassengerId'],axis=1)
# encoding the sex and embarked as in integers form
df['Sex'] = df['Sex'].map({'male':1,'female':0})
df['Embarked'] = df['Embarked'].map({'S':0,'C':1,'Q':2})
df['tot_par'] = df['SibSp']+df['Parch']
#we can drop the SibSp and Parch column
df = df.drop(['SibSp','Parch'],axis=1)
df['titles'] = df.Name.str.extract(' ([\w]+)\.',expand = False)
df['titles']=df['titles'].map({'Miss':4,'Mrs':3,'Mr':2,'Master':1})
df['titles'] = df['titles'].fillna(0)
df = df.drop('Name', axis=1)
df = df.dropna(subset=['Age', 'Embarked'])
#normalize the fare
df['Fare'] = (df['Fare'] - df['Fare'].mean()) / (df['Fare'].std())
df = df.values
X, y = df[:, 1:], df[:, 0]
clf = perceptron(lr=0.0011, rs=25, n_iter=100)
clf.fit(X,y)
pred = clf.perdict(X)
# for i in range(len(pred)):
#     print(pred[i],y[i])
acc = (pred == y).sum()/y.shape[0]
print('accuracy on the train set - ', acc)
