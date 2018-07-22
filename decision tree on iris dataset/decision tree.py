# implementation inspired by ml recipe by google implemented taking entropy as error function and on iris dataset
import pandas as pd
import math
import numpy as np
from sklearn.cross_validation import train_test_split


def unique(data,col):
    return set([row[col] for index,row in data.iterrows()])

def is_numeric(value):
    """Test if a value is numeric."""
    return isinstance(value, int) or isinstance(value, float)


class leaf:

    def __init__(self,data):
        self.prediction = unique(data,data.shape[1]-1)


class question:

    def __init__(self,col,val):
        # print(col,val)
        self.column = col
        self.value = val

    def match(self,row):
        # print(row)
        if isinstance(row[self.column],int) or isinstance(row[self.column],float):
            return row[self.column] >= self.value
        else:
            return row[self.column] == self.value

    def __repr__(self):
        # This is just a helper method to print
        # the question in a readable format.
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

header=['SepalLengthCm' ,'SepalWidthCm' , 'PetalLengthCm' , 'PetalWidthCm','Species']

def count_class(data):
    temp = {}
    for index,row in data.iterrows():
        if row[data.shape[-1]-1] not in temp:
            temp[row[data.shape[-1]-1]] = 1
        else:
            temp[row[data.shape[-1]-1]] += 1
    return temp



def best_question(data):
    initial_impurity = entropy(data)
    max_IG = 0.0
    b_ques = None
    # print(data)
    data=pd.DataFrame(data)
    for col in range(data.shape[1]-1):
        row = unique(data,col)
        for r in row:
            ques = question(col,r)
            true_tree, false_tree = partition(ques,data)
            if len(true_tree) == 0 or len(false_tree) == 0:
                continue
            total = len(data)
            IG = info_gain(initial_impurity, true_tree , false_tree ,total)
            if IG >= max_IG:
                max_IG = IG
                b_ques = ques

    return max_IG , b_ques



def entropy(data):
    impurity = 0
    data = pd.DataFrame(data)
    counts = count_class(data)
    for i in counts:
        prob = float(counts[i]/(len(data)))
        if prob != 0.0 and prob != 1.0:
            impurity = impurity - ((prob*math.log(prob,2) + (1-prob)*math.log((1-prob),2)))
    return impurity


def info_gain(impurity, true_tree , false_tree, total):
    return impurity - float(len(true_tree)/total)*entropy(true_tree) - float(len(false_tree)/total)*entropy(false_tree)

def partition(ques,data):

    true, false = [],[]
    for index,row in data.iterrows():
        if ques.match(row):
            true.append(row)
        else:
            false.append(row)
    return true ,false

def build_tree(data,depth):
    data = pd.DataFrame(data)
    gain,ques = best_question(data)

    if gain == 0.0 or depth == 0:
        return leaf(data)

    true_tree, false_tree = partition(ques,data)
    true = build_tree(true_tree,depth-1)
    false = build_tree(false_tree,depth-1)

    return Decision_store(ques,true,false)

class Decision_store:

    def __init__(self,q,t,f):
        self.ques = q
        self.true = t
        self.false = f


def predict(x_test,tree):
    #by recursion
    if isinstance(tree, leaf):
        return tree.prediction

    if tree.ques.match(x_test):
        return predict(x_test, tree.true)
    else:
        return predict(x_test, tree.false)

def print_tree(node, spacing=""):

    # Base case: we've reached a leaf
    if isinstance(node, leaf):
        print (spacing + "Predict", node.prediction)
        return

    # Print the question at this node
    print (spacing + str(node.ques))

    # Call this function recursively on the true branch
    print (spacing + '--> True:')
    print_tree(node.true, spacing + "  ")

    # Call this function recursively on the false branch
    print (spacing + '--> False:')
    print_tree(node.false, spacing + "  ")
import os
os.chdir('C:/Users\shankul\Desktop/all')
df = pd.read_csv("iris.csv",header=None,skiprows=1).values
x_train,x_test,y_train,y_test = train_test_split(df[:,[1,2,3,4]],df[:,5],test_size=0.3,random_state=25)
x_train = np.insert(x_train,4,y_train,axis=1)
tree = build_tree(x_train,4)
print_tree(tree, " ")
a=0
for i in range(len(x_test)):
    if y_test[i] in list(predict(x_test[i],tree)) :
        a=a+1
print(a/len(x_test))


