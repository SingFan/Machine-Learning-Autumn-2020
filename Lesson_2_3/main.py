# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:57:28 2019

@author: Brian Chan
"""

# =============================================================================
# Data Types (integer, float, string, list, set, dict)
# =============================================================================

# 1. integer and float
x = 1
type(x)

x = 1.0
type(x)

y = x + 1

x = True
type(x)

int(x)

print(x,y)
print(x,'\n',y)

# 2. string
text = 'Python strings are sliceable.'
text[0]

text[10]

L = len(text)

split_text = text.split(' ')

' '.join(split_text) 

#text[L]  # Error
#text[:10]

text = '0123456789'
text[:5]
text[5:]


print(y,x)

# 3. list

x=[1,2,3,4]

# 2 dimensional
# [1,2,3,4]
# [5,6,7,8]

x = [[1,2,3,4], [5,6,7,8]]

x[0]
x[0][0]

x = [[1,2,3,4] , [5,6,7]]
x = [0,1,2,3,4,5,6,7,8,9]
x[0]
x[5]
#x[10] # Error

# 4. slicing for list

x[4:] # [4, 5, 6, 7, 8, 9]
x[:4] # [0, 1, 2, 3]
x[1:4] # [1, 2, 3]
x[0]
x[1]

x = [[1,2,3,4] , [5,6,7]]
x[0][1:4] # [2, 3, 4]

x = [0,1,2,3,4,5,6,7,8,9]
len(x)
x.append(0)
len(x)

x.extend([11,12,13]) # similar to .append, but .extend conccatenate from the input, which needs not to be a list

x = range(10)
len(x)
   
# 5. dict
data = {'age': 34, 'children' : [1,2], 1: 'apple'}

data['age']

# 6. set
x = set(['MSFT','GOOG','AAPL','HPQ','MSFT']) # from list to set
x.add('CSCO')

y = set(['XOM', 'GOOG'])
x.intersection(y)

x = x.union(y)
len(y)

# Example: Merge two lists and drop duplicated entries in the merged list

x = ['MSFT','GOOG','AAPL','HPQ','MSFT']
y = ['XOM', 'GOOG']
x = list(set(x).union(set(y)))   

# 7. for loop
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

print(len(animals))

for i in range(10):
    print(i)

# =============================================================================
# Exercise: create a list with ascending order integer from 0 to 10
# =============================================================================

x = list()
for i in range(10):
    x.append(i)


# =============================================================================
# Exercise: Find smallest number in a list
# =============================================================================

import numpy as np
Test = list(np.floor(np.random.rand(10)*100))

minimium = Test[0]
for x in Test:
    if x < minimium:
        minimium = x

print(minmium)



# =============================================================================
# array
# =============================================================================
import numpy as np

# 1. array creation
x = [0.0, 1, 2, 3, 4]
y = np.array(x)

type(y)

y = np.array([[0.0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

# 2. shape
np.shape(y)
y.shape


y = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
y.shape


x = np.array([[1.0,2.0],[3.0,4.0]])
y = np.array([[5.0,6.0],[7.0,8.0]])

# 3. concatenate
z = np.concatenate((x,y),axis = 0)
z = np.concatenate((x,y),axis = 1)

x = np.reshape(np.arange(4.0),(2,2))

# 4. some manipulation

x.T # transpose

np.arange(1.0, 4.0, 0.5)

x = np.arange(4.0) - 1 

x = np.arange(1.0,5.0)

x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

# element-wise multiplication
print(x * y)
print(np.multiply(x, y))

# element-wise division
print(x / y)
print(np.divide(x, y))

print(np.divide(x, y)*y)

# matrix multiplication
print(np.matmul(x,y))

# 5. special
x = np.ones((10,1))
x = np.zeros((10,1))

# 6. logic
x = np.arange(-1,3)
x <= 0

# for loop

animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print(animal)

print(len(animals))

for x in range(10):
    print(x)


# function
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

for x in [-1, 0, 1]:
    print(sign(x))

# lambda
lambda a, b: (a + 1, b * 1)
#_(1,3)

two_returns = lambda a, b: (a + 1, b * 1)
two_returns(1,3)

def two_returns(a, b):
    # logic is here
    c = a + 1
    d = b * 1
    return c, d

two_returns(1,3)

# =============================================================================
# List comprehension
# =============================================================================

# conventional way to create list array
odds = [x for x in range(50) if x%2]
odds = [x%2 for x in range(50)]

# comparison
odds = []
for x in range(50):
    if x%2:
        odds.append(x)

odds = []
for x in range(50):
    odds.append(x%2)
        
# =============================================================================
# functionn: arbitrary arguments
# =============================================================================

# *args, **kwargs

# =============================================================================
# import/export
# =============================================================================

f = open("Ticker_list.txt", "r")
Data = list()
for line in f:
    Data.append(line)

len(Data)

import pandas as pd

Data = pd.read_excel("Ticker_list2.xlsx", index_col=1, header=1)
Data = pd.read_excel("Ticker_list2.xlsx", index_col=0, header=None)

# Data = pd.read_csv("Ticker_list.csv")

Data = pd.read_excel("data.xlsx", index_col=0, header=0)
Data.index = pd.to_datetime(Data.index, format='%Y-%m-%d %H:%M:%S')
Data.index
Data.iloc[0,1]
Data.iloc[0,1:10]
Data.head()
Data.shape
print(Data.describe())
Data.sum(axis = 0)


Data['2018']
Data['2019Q1']
Data.index.quarter

# nan entries filling in
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
		[3, 4, np.nan, 1],
		[np.nan, np.nan, np.nan, 5],
		[np.nan, 3, np.nan, 4]],
		columns=['A','B','C','D'])

df.fillna(0)
df.fillna(method='ffill') # forward fill
values = {'A': 0, 'B': 1, 'C': 2, 'D': 3} 
df.fillna(value=values)
# =============================================================================
# plot
# =============================================================================
import matplotlib.pyplot as plt

r = np.random.randn(1000) # rand
y = np.cumsum(r)
plt.figure()
plt.plot(y)
plt.xlabel("Step")
plt.ylabel("$\sum^N_{t = 1} r_t$")
plt.title("Simulation path of random walk")

# simulate multiple time series
plt.figure()

for i in np.arange(10):
    r = np.random.randn(1000) # rand
    y = np.cumsum(0.5*r + 5e-2)
    plt.plot(y)
    
plt.xlabel("Time Step")
plt.ylabel("$\sum^N_{t = 1} r_t$")
plt.title("Simulation path of random walk")


# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]

# Create plots with pre-defined labels.
plt.figure()
plt.plot(a, c, 'k--', label='Model length')
plt.plot(a, d, 'k:', label='Data length')
plt.plot(a, c + d, 'k', label='Total message length')

plt.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()

# =============================================================================
# Exercise 1: Plot y = x^2
# =============================================================================

N = 100
x = np.arange(10)
plt.figure()
plt.plot(x**2)
    
plt.xlabel("x")
plt.ylabel("$y=x^2$")
plt.title("Plot of y = $x^2$")


# =============================================================================
# Exercise 2: write a function of y = a*x^2 + b*x + c and plot graph
# =============================================================================

def function(x, a, b, c):
    y = a*x**2 + b*x + c
    return y

x_start = -10
x_end   = 10
y = []
a = 1
b = 2
c = 1
for x in np.arange(x_start, x_end, 0.1):
    y.append(function(x, a, b, c))

plt.figure()
plt.plot(np.arange(x_start, x_end, 0.1),y)
    
plt.xlabel("x")
plt.ylabel("$y=x^2$")
plt.title("Plot of y = $x^2$")




# =============================================================================
# Exercise 3: Find average of a list array
# =============================================================================

N_sim = 1e3
data  = np.random.randn(int(N_sim)) # int only
average = 0
for i in np.arange(len(data)):
    average = average + data[i]

average = average/len(data)
print("Average: %.5f"%(average)) # decimal places


#plt.figure()
#plt.scatter(np.arange(len(data)), data)

plt.figure()
plt.hist(data, 500) # probability density function of standard normal distribution



# =============================================================================
# Exercise 4: Solve system of equations
# 2a + 1b = 3
# 3a + 2b = 4
# =============================================================================

# Hints: consider Ax = c

A = np.array([[2,1],[3,2]])
c = np.array([[3],[4]])

x = np.linalg.solve(A, c)

np.matmul(A,x)






# =============================================================================
# Reference:
# =============================================================================
#http://cs231n.github.io/python-numpy-tutorial/

#https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-slides-code/
