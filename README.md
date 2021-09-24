# Popular machine learning algorithms implemented from scratch 

Implementing important machine learning algorithms from bare bones just using NumPy as primary dependency. Goal for this repository is to cover extensively all major algortihms and code simple implementations so that anyone can kick start their machine learning journey.

## Regression

- ### Linear Regression
  Goal of linear regression is to model the realtionaship between explanatory feature *x* and continous values response *y*. [Implementation](Perceptron)

- ### Ridge Regression
  Ridge regression is an L2 penalized model where we simply add the suqred sum of weights to our cost funstion of regression. [Imple]

- ### Lasso regression
  Lasso Regression includes an L1 penalty to our cost function. This penalty allows some coefficient values to go to the value of zero, allowing input variables to be effectively removed from the model, providing a type of automatic feature selection.
- ### Elastic Net
  Elastic net linear regression uses the penalties from both the lasso and ridge techniques to regularize regression models.
  
## Gradient descent optimization algorithms

- ### Stochastic Gradient Descent (SGD)
  Stochastic gradient descent (SGD) in gradient descent optimiser that performs parameter update for each training example and label.
  
- ### SGD with momentum
  SGD has trouble navigating ravines, i.e. areas where the surface curves much more steeply in one dimension than in another, which are common around local optima. Momentum is a method that helps accelerate SGD in the relevant direction and dampens oscillation.
  
- ### RMSProp
  Root Mean Squared Propagation, or RMSProp, is an extension of gradient descent and the AdaGrad version of gradient descent that uses a decaying average of partial gradients in the adaptation of the step size for each parameter.
  
- ### Adam
  Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
  
## Neural network

- ### Perceptron
  A perceptron is the simplest form of an ANN. The perceptron consists of one neuron with two inputs and one output. It is just a simplified model of a biological neuron

- ### Artificial Neural Network or Multilayer perceptron
  Artificial Neural Network, or ANN, is a group of multiple perceptrons/ neurons at each layer. ANN is also known as a Feed-Forward Neural network because inputs are processed only in the forward direction
 
- ### Convolutional Neural Network
  A convolutional neural network is a feed-forward neural network that is generally used to analyze visual images by processing data with grid-like topology. A convolutional neural network is used to detect and classify objects in an image.
  
## Principal Component Analysis(PCA)

- ### Linear PCA
  Principal Component Analysis (PCA) - which is a linear dimensionality reduction technique that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.
  
- ### Kernel PCA
  Kernel Principal Component Analysis (KPCA) is a non-linear dimensionality reduction technique. It is an extension of Principal Component Analysis (PCA), kernel PCA uses a kernel function to project dataset into a higher dimensional feature space, where it is linearly separable
  
## Support Vector Machine

SVM algorithm is a supervised learning algorithm that find the best line or decision boundary for classification. It works on the concept of maximum margin classification. SVM algorithm finds the closest point of the lines from both the classes, called support vectors. The distance between the vectors and the hyperplane is called as margin. And the goal of SVM is to maximize this margin. 

## Decision tree

A Decision tree is a supervised learning technique which follows flowchart like tree structure, where each internal node denotes a test on an attribute, each branch represents an outcome of the test, and each leaf node (terminal node) holds a class label. 

## K-Means Clustering

K-Means Clustering is an Unsupervised Learning algorithm, which groups the unlabeled dataset into different clusters. Here K defines the number of pre-defined clusters that need to be created in the process. It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

## Naive Bayes

Naive Bayes algorithm is a classification technique based on Bayes’ Theorem with an assumption of independence among predictors. The name *naive* is used because it assumes the features that go into the model is independent of each other.

## Linear Discriminant Analysis

Linear Discriminant Analysis (LDA) is the commonly used dimensionality reduction technique in supervised learning. LDA, does the separation by computing the directions (“linear discriminants”) that represent the axis that enhances the separation between multiple classes. 

## K-nearest neighbors

K-nearest neighbors (KNN) algorithm is a type of supervised ML algorithm which can be used for both classification as well as regression predictive problems. K-nearest neighbors (KNN) algorithm uses ‘feature similarity’ to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set.

## Logistic Regression

Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.  In this fit an "S" shaped logistic function, which predicts two maximum values (0 or 1).
