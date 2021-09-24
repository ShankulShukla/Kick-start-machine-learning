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
