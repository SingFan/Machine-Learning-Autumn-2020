# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:16:02 2019

@author: Brian Chan
"""

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
# 1. Simulate data
# =============================================================================
N_sample = 20


X = np.random.normal(1, 1, N_sample)
e = np.random.normal(0, 0.2, N_sample)

y = 1 + 1*X + e

data = pd.DataFrame(np.stack((X,y),axis = 1), columns = ['X','y'])

data.describe()

plt.figure()
plt.scatter(data['y'],data['X'])
plt.xticks(np.arange(data['X'].min(),data['X'].max(),step=5))
plt.yticks(np.arange(data['y'].min(),data['y'].max(),step=5))
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
        error = np.dot(X.transpose(),(predictions -y))
        descent=alpha * 1/m * error
        theta-=descent
        J_history.append(computeCost(X,y,theta))
    
    return theta, J_history

theta,J_history = gradientDescent(X, y, theta, 0.01, 1500)
print("h(x) = "+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

# =============================================================================
# 4. 3D plot
# =============================================================================
from mpl_toolkits.mplot3d import Axes3D

#Generating values for theta0, theta1 and the resulting cost value

theta0_vals = np.linspace(-5.0,5.0,100)
theta1_vals = np.linspace(-2.0,2.0,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t           = np.array([theta0_vals[i],theta1_vals[j]])
        J_vals[i,j] = computeCost(X,y,t)
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
# 
# =============================================================================
import statsmodels.api as sm

X     = sm.add_constant(X)
model = sm.OLS(y,X).fit()

print(model.summary())

















