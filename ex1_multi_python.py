import numpy as np

import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider, Button

%matplotlib inline

################################################################ Here we will do linear regression with multivariables #################################################################


################################################################ Loading Data #################################################################


dataset_Multi = 'C:/Users/tuico/Desktop/Courseca_Machine_Learning/Week_two_Coding _Exercises/ex1data2.txt' #This is the data-set we are extracting our examples From


data_Multi = np.loadtxt(dataset_Multi, delimiter=',', skiprows=0, dtype=float)


data_Multi[0,:]

X = data_Multi[:, 0:2]


X


y = data_Multi[:, 2].reshape(len(data_Multi[:,1]),1)

### Here you can y.size ###

m = len(y)
################################# Printing out some data points #################################################################

print('The first 10 examples from the dataset: ')
print('x = [%.0f, %.0f], y = %.0f', [X[0:9,:], y[0:9,:]])

############### Feature Normalization ###############

def featureNormalize(X_input):

    X_norm = X

    X_norm_2 = X

    μ = np.zeros((1, m))

    σ = np.zeros((1, m))

    μ = X.mean(axis= 0 )*np.ones((1,m)).T

    σ = X.std(axis= 0 )*np.ones((1,m)).T


    X_norm = (X-μ)/σ

    X_norm

    return(μ, σ, X_norm)

[μ,σ, X] = featureNormalize(X)
###These will be diagonal matrices containing the desired values
### just call out the first element of each or change code as desired ###

print("Computed μ = ",μ[0])
print("Computed σ = ",σ[0])







####################### Adds 1 to the incercept term to X #################
X = np.hstack((np.ones((m,1)), X))


X
## Simple-gradient descent alg ##


## Introduce compute-cost function ##
def ComputeCost(X, Y, θ):
    prediction =  X @ θ

    sqrErrors = (prediction -y)
    sqrErrors_2 = (prediction -y).T  @ (prediction-y)

    J = 1/(2*m)*np.sum(sqrErrors_2)

    return(J)


### This version of the GD-Alg is correct ###

def GradientDescent(x, y, θ,α , num_iters):
    J_history = np.zeros((num_iters,1))
    for iter in range(num_iters):

        error = X @ θ -y ##dim(hyp) = 400 x 1

        θ = θ - ((α/m)*X.T@error) # dim(θ) = 3 x 1

        J_history[iter] = ComputeCost(X, y, θ)
    return(θ, J_history)
## Choose arbitrary data here ##

α = 1
num_iters = 400
θ = np.zeros((3,1))


GradientDescent(X, y, θ, α , num_iters)
np.shape(GradientDescent(X, y, θ,α , num_iters))


[θ, J_history] = GradientDescent(X, y, θ, α, num_iters)

θ
### Ploting of cost curve function here ###

plt.plot(np.arange(len(J_history)), J_history)
plt.xlabel("Number of Iterations")
plt.ylabel("Cost-J($\Theta$)")
plt.show()


### Plotting the GD-Alg theta values ###

print('θ computed from the GD-Alg: \n')
print(θ.T)
print("\n")

### Here we will do Feature Normalization along with predicting the price of a house
### of 1650ft^2

### The first thing that we need to note is that the price of the house is the hypothesisi

### i.e price =  X @ θ

## We  need to define a new array of X that is formated as
###@ X_proj = [Stand = 1, House Size, θ(size)]

X_proj = [1, 1650, 3]

#Now we need to normalize the data that we just got, excluding the first column,

# as it is already normalized - This is the same formula as normal distributions.


X_proj[1:3] = (X_proj[1:3] - μ[0])/σ[0]


X_proj
price = X_proj @ θ
print("Therefore the predicted price for a house of 1650ft^2 is: ", price)


### Last, we can use the normal equation to find a closed form solution
###for ### θ

### \Theta =(X@X.T)^(-1)X.T@y

### This can be obtained by calling a function
def NormalEqn(X,y):

    θ = np.zeros(X.shape[1])


    θ = np.linalg.inv(X.T@X)@X.T@y

    return(θ)


θ = NormalEqn(X,y)

### Now w can print off the theta from teh subsequent normal equations ###

print("Thus, Θ when computed from the Normal-Equation form is θ = ",θ)


# Again we predict the price for a house of 1650ft^2 ###


X_proj_norm = [1, 1650, 3]

#Now we need to normalize the data that we just got, excluding the first column,

# as it is already normalized - This is the same formula as normal distributions.


X_proj_norm[1:3] = (X_proj_norm[1:3] - μ[0])/σ[0]


X_proj_norm
price_norm = X_proj_norm @ θ
print("Therefore the predicted price for a house of 1650ft^2 is: ", price_norm, " Dollars, when obtained from normal equations")
