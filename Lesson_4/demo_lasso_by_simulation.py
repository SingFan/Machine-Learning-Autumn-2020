# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:16:02 2019

@author: Brian Chan
"""
# Lesson 3: Demonstration of optmization algorithm and simple least square method

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm
# =============================================================================
# Case 1: quadratic form
# =============================================================================
def quadratic_form(center, theta, COV):
    return np.matmul(np.matmul((theta - center).T,COV),(theta - center))

# =============================================================================
# 1. 3D plot
# =============================================================================

from mpl_toolkits.mplot3d import Axes3D

#Generating values for theta0, theta1 and the resulting cost value
COV    = np.eye(2)
center = np.array([[4],[0.5]])# center = np.ones((2,1))*2
theta0_vals = np.linspace(-2.0,8.0,100)
theta1_vals = np.linspace(-3.0,5.0,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t           = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[j,i] = quadratic_form(center,t,COV)
        
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
#Generating the surface plot
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
cset = ax.contour(theta0_vals,theta1_vals,J_vals, zdir='z', levels=np.arange(0,len(theta0_vals)-1,2), offset=-5.0, cmap=cm.RdYlBu)
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.set_title('Plot quadratic form')
#rotate for better angle
ax.view_init(30,120)


# =============================================================================
# Case 2: quadratic form with norm 1 (L1)
# =============================================================================
def quadratic_form_L1(center,theta,COV, Lambda):
    return np.matmul(np.matmul((theta - center).T,COV),(theta - center)) + Lambda*np.sum(np.abs(theta))

#Generating values for theta0, theta1 and the resulting cost value
COV    = np.eye(2)
center = np.array([[4],[0.5]])# center = np.ones((2,1))*2
Lambda = 5

theta0_vals = np.linspace(-2.0,8.0,100)
theta1_vals = np.linspace(-3.0,5.0,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t           = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[j,i] = quadratic_form_L1(center,t,COV,Lambda)
        
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
#Generating the surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
cset = ax.contour(theta0_vals,theta1_vals,J_vals, zdir='z', levels=np.arange(0,len(theta0_vals)-1,2), offset=-5.0, cmap=cm.RdYlBu) # coolwarm
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.set_title('Plot quadratic form with $L_1$')
#rotate for better angle
ax.view_init(30,120)

# =============================================================================
# Case 3: norm 1 (L1)
# =============================================================================
def L1(theta, Lambda):
    return Lambda*np.sum(np.abs(theta))


#Generating values for theta0, theta1 and the resulting cost value
Lambda = 5

theta0_vals = np.linspace(-2.0,8.0,100)
theta1_vals = np.linspace(-3.0,5.0,100)
J_vals=np.zeros((len(theta0_vals),len(theta1_vals)))

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t           = np.array([[theta0_vals[i]],[theta1_vals[j]]])
        J_vals[j,i] = L1(t, Lambda)
        
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
#Generating the surface plot
fig  = plt.figure()
ax   = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0_vals,theta1_vals,J_vals,cmap="coolwarm")
cset = ax.contour(theta0_vals,theta1_vals,J_vals, zdir='z', levels=np.arange(0,len(theta0_vals)-1,5), offset=-5.0, cmap=cm.RdYlBu)
#fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("$\Theta_0$")
ax.set_ylabel("$\Theta_1$")
ax.set_zlabel("$J(\Theta)$")
ax.set_title('Plot norm 1 ($L_1$)')
#rotate for better angle
ax.view_init(30,120)

#L1(np.array([[0],[0]]), Lambda)

