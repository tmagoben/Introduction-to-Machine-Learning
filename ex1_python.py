import numpy as np

import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Slider, Button

%matplotlib inline

################################################################  This is my Journery of converting Andrew Ng's Coursera Intro to Machine Learning Course ################################################################
################################################################  From MATLAB langauge to python ################################################################

################################################################  I will use bare-bones basic python syntax to re-interperate the exercises ################################################################





################################################################  Part 1 ################################################################
################################################################  Extracting and ploting the data ################################################################
################################################################  x refers to the populatin in 10,000's ################################################################
################################################################  y refers to the profit in $10,000's ################################################################



dataset = 'C:/Users/tuico/Desktop/Courseca_Machine_Learning/Week_two_Coding _Exercises/ex1data1.txt' #This is the data-set we are extracting our examples From
# Just change this to the path where your dataset(s) are located
################################################################  We will now lpad the text-set using an np.loadtxt function. ################################################################
data = np.loadtxt(dataset, delimiter=',', skiprows=0, dtype=float)

################################################################  Check to view dataset ################################################################
print(data)

################################################################  Extracting the X values ################################################################
X = data[:,0].reshape(len(data),1) # dim(X) = 97 x 1


################################################################  Extracting the y values ################################################################
y = data[:,1].reshape(len(data),1) #dim(y) = 97 x 1

y
m = len(y) ################################################################  This is the number of training examples





################################################################ Here we are plotting the data that we've been given ################################################################
plt.plot(X,y, 'x', label = 'Training Data')
plt.xlabel('X-data')
plt.ylabel('y-data')
plt.legend()
plt.show()

################################################################ Next, the goal is to compute the cost and Gradient Descent of our function ################################################################


################################################################  The firt thing we must do is add a column of 1's to the training examples ################################################################
################################################################  To atribute to the x0 training examples ################################################################
X = np.append(np.ones((1,m)).T,X, axis = 1)  # dim(X) =  97 x 2

################################################################  Now we create the intialize theta to zeros ################################################################

θ = np.zeros((2,1))  # dim(θ)

np.shape(θ)

θ
################################################################ Here are the parameters for gradient Descent ################################################################

iterations = 1500

α = 0.01

################################################################  Now we must generate the compute cost function ################################################################

################################################################  This is achieved using a SSE estimator in vectorized form ################################################################


def ComputeCost(X,Y, θ):
    prediction =  X @ θ

    sqrErrors = (prediction -y)**2

    J = 1/(2*m)*np.sum(sqrErrors)

    return(J)


J = ComputeCost(X,y,θ)
################################################################  Run a first check on the operation of ComputeCost ################################################################
print('The cost function with θ = [0;0],is J = ',J)
print('This value should be 32.07')




################################################################  Further Testing ################################################################
print('The cost function with θ = [-1;2], is J = ', ComputeCost(X,y,np.array([[-1],[2]])))

print('This value should be (approx) 54.24')


################################################################ Gradient Descent Algorithim in a vectorized fahsion ################################################################
def GradientDescent(x, y, θ,α , num_iters):
    J_history = np.zeros((num_iters,1))
    for iter in range(num_iters):

        error = X @ θ -y ##dim(hyp) = 97 x 1

        θ = θ - ((α/m)*X.T@error) # dim(θ) = 2 x 1

    J_history = ComputeCost(X, y, θ)

    return(θ)


################################################################  We now should have new values of optimized θ(hypothesis) parameters for our dataset ################################################################

#Setting θ equal to the grad-desc of our data and paramete
θ_GD = GradientDescent(X, y, θ, α, iterations)

################################################################ Printing optimized θ ################################################################

print('theta found by gradient descent is:', θ_GD)
print('This agrees with the MATLAB model!')

################################################################ Ploting the linearly fit model with our data set ################################################################

plt.plot(X[:,1],y, 'x', label = 'Training Data')
plt.plot(X[:,1], X@ θ_GD, label = 'Linear model')
plt.legend(loc = 4 )
plt.title('$\\theta_{GD}$ $Optimized$')
plt.xlabel('$\\theta_{0}$')
plt.ylabel('$\\theta_{1}$')
plt.show()

################################################################  We will now predivt values for population sizes od 35,000 and 70,000 ################################################################

predict_1 = [1, 3.5]@ θ_GD

print('For population = 35,000, we predict a profit of: ', predict_1*10000)


predict_2 = [1,7]@ θ_GD

print('For population = 70,000, we predict a profit of: ', predict_2*10000)



################################################################ Visualiizing J(θ_0, θ_1) ################################################################
################################################################  Grid over which we will calculate J_history ################################################################
θ0_vals = np.linspace(-10,10,100)

θ1_vals = np.linspace(-1, 4, 100)

################################################################  Initialize J_vals to a matrix of 0's ################################################################
J_vals  = np.zeros((len(θ0_vals), len(θ1_vals)))

################################################################  Fill out J_Vals ################################################################

for i in range(len(θ0_vals)):
    for j in range(len(θ1_vals)):
        t = np.array([[θ0_vals[i]],[θ1_vals[j]]])
        J_vals[i,j] = ComputeCost(X, y, t)


################################################################  Check J_vals to make sure it is proper ################################################################

J_vals

################################################################  Surface Plot of 2D cost function ################################################################

################################################################  Mesh grid of θ values ################################################################
X_θ,Y_θ = np.meshgrid(θ0_vals, θ1_vals)
ax = plt.axes(projection='3d')
surf =ax.plot_surface(X_θ, Y_θ, J_vals.T, cmap = cm.inferno)
ax.view_init(25, 50)
plt.xlabel('$\\theta_0$')
plt.ylabel('$\\theta_1$')
plt.title('$J(X_\\theta,Y_\\theta)$')
plt.show()

################################################################  Counter Plot ################################################################
## Plot J_vals as 15 contours apced logarithmically betweenn 0.01 and 100 ####


plt.contour(J_vals,np.logspace(-2,3,20))
plt.xlabel('$\\theta_0$')
plt.ylabel('$\\theta_1$')
plt.title('$J(X_\\theta,Y_\\theta)$ $Contour$ $Plot$')
plt.show()
