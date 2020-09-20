# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 22:16:23 2019

@author: Brian Chan
"""

from numpy import arange
import matplotlib.pyplot as plt
from scipy.stats import norm
# x-axis for the plot
x_axis = arange(-3, 3, 0.001)

# =============================================================================
# 
# =============================================================================
std_list = list([0.2,0.4,0.6,1.0])

plt.figure()
for i in std_list:
    plt.plot(x_axis, norm.pdf(x_axis, 0, i))

plt.show()
plt.title('Difference std parameters')
plt.legend(std_list)

# =============================================================================
# 
# =============================================================================
mean_list = list([0.2,0.4,0.6,1.0])

plt.figure()
for i in std_list:
    plt.plot(x_axis, norm.pdf(x_axis, i, 1))

plt.show()
plt.title('Difference mean parameters')
plt.legend(mean_list)