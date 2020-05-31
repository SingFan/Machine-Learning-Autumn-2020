# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 21:04:15 2019

@author: Brian Chan
"""
# Lesson 3: Demonstration of optmization algorithm and simple least square method

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
# 1. Import data
# =============================================================================

data=pd.read_csv("Uni_linear.txt", header=None)

data.head()

data.describe()

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

# =============================================================================
# 2. Cost function
# =============================================================================
def computeCost(X,y,theta):
    """
    Take in a numpy array X,y, theta and generate the cost function of using theta as parameter
    in a linear regression model
    """
    m           = len(y)
    predictions = X.dot(theta)
    square_err  = (predictions - y)**2
    
    return 1/(2*m) * np.sum(square_err)

data_n = data.values
m      = data_n[:,0].size
X      = np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y      = data_n[:,1].reshape(m,1)
theta  = np.zeros((2,1))

computeCost(X,y,theta)

# =============================================================================
# 3. Gradient descent algorithm
# =============================================================================
def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each iteration
    """
    m         = len(y)
    J_history = []
    
    for i in range(num_iters):
        predictions = X.dot(theta)
        error   = np.dot(X.transpose(),(predictions -y))
        descent = alpha * 1/m * error
        theta  -= descent
        J_history.append(computeCost(X,y,theta))
    
    return theta, J_history

theta,J_history = gradientDescent(X, y, theta, 0.01, 1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

# =============================================================================
# 4. 3D plot
# =============================================================================
from mpl_toolkits.mplot3d import Axes3D

#Generating values for theta0, theta1 and the resulting cost value

theta0_vals=np.linspace(-5,5,100)
theta1_vals=np.linspace(-1,4,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t=np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i,j]=computeCost(X,y,t)

theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
#Generating the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")

#rotate for better angle
ax.view_init(30,120)

# =============================================================================
# 5. History of optimization iteration
# =============================================================================

plt.figure()
plt.plot(J_history[1:])
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

# =============================================================================
# 6. Scatter with best fit line
# =============================================================================
plt.figure()
plt.scatter(data[0],data[1])
x_value = [x for x in range(25)]
y_value = [y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")

# =============================================================================
# residuals plot
# =============================================================================

x_value = [x for x in range(len(y))]
y_value = [data.iloc[i,1] - (data.iloc[i,0]*theta[1]+theta[0]) for i in np.arange(len(data))]
plt.figure()
plt.scatter(x_value, y_value)
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Residual error ($10,000)")
plt.title("Residual error of Profit Prediction")


# =============================================================================
# 7. Prediction demonstration
# =============================================================================
def predict(x,theta):
    """
    Takes in numpy array of x and theta and return the predicted value of y based on theta
    """
    predictions= np.dot(theta.transpose(),x)
    
    return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))


predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))





















