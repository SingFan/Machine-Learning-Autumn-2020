# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 21:03:19 2019

@author: Brian Chan
"""

from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
# =============================================================================
# plot of logit function
# =============================================================================
t = np.linspace(-10, 10, 100)
sig = 1 / (1 + np.exp(-t))
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1 + e^{-t}}$")
plt.xlabel("t")
plt.legend(loc="upper left", fontsize=20)
plt.axis([-10, 10, -0.1, 1.1])
#save_fig("logistic_function_plot")
plt.show()

from sklearn import datasets
iris = datasets.load_iris()
list(iris.keys())

# =============================================================================
# Example 1
# =============================================================================

print("Example 1: logistic (Iris or not, with one explanary variable)")


from sklearn import datasets
iris = datasets.load_iris()

print(list(iris.keys()))

# print(iris.DESCR)

X = iris["data"][:, 3:]  # petal width
y = (iris["target"] == 2).astype(np.int)  # 1 if Iris-Virginica, else 0

# =============================================================================
# # logistic regression
# =============================================================================

log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X, y)

print("logistic coeff", log_reg.coef_)
print("logistic intercept", log_reg.intercept_)


# plot of logit function
X_new   = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)

plt.figure()
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.show()

# =============================================================================
# # demonstration of classifying with logit function
# =============================================================================
X_new   = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]

plt.figure(figsize=(8, 3))
plt.plot(X[y==0], y[y==0], "bs")
plt.plot(X[y==1], y[y==1], "g^")
plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
plt.text(decision_boundary+0.02, 0.15, "Decision  boundary", fontsize=14, color="k", ha="center")
plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc='b', ec='b')
plt.arrow(decision_boundary, 0.92, 0.3, 0, head_width=0.05, head_length=0.1, fc='g', ec='g')
plt.xlabel("Petal width (cm)", fontsize=14)
plt.ylabel("Probability", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 3, -0.02, 1.02])
#save_fig("logistic_regression_plot")
plt.show()


# =============================================================================
# Example 2
# =============================================================================

print("Example 1: logistic (Iris or not, with two explanary variables)")

X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression(solver="liblinear", C=10**10, random_state=42)
log_reg.fit(X, y)

print("logistic coeff", log_reg.coef_)
print("logistic intercept", log_reg.intercept_)

x0, x1 = np.meshgrid( np.linspace(2.9, 7, 500).reshape(-1, 1), np.linspace(0.8, 2.7, 200).reshape(-1, 1))

X_new = np.c_[x0.ravel(), x1.ravel()]

y_proba = log_reg.predict_proba(X_new)

plt.figure(figsize=(10, 4))
plt.plot(X[y==0, 0], X[y==0, 1], "bs")
plt.plot(X[y==1, 0], X[y==1, 1], "g^")

zz = y_proba[:, 1].reshape(x0.shape)
contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)


left_right = np.array([2.9, 7])
boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]


plt.clabel(contour, inline=1, fontsize=12)
plt.plot(left_right, boundary, "k--", linewidth=3)
plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.axis([2.9, 7, 0.8, 2.7])
#save_fig("logistic_regression_contour_plot")
plt.show()


# =============================================================================
# Softmax
# =============================================================================

#X = iris["data"][:, (2, 3)]  # petal length, petal width
#y = iris["target"]
#
#softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
#softmax_reg.fit(X, y)

#x0, x1 = np.meshgrid(
#        np.linspace(0, 8, 500).reshape(-1, 1),
#        np.linspace(0, 3.5, 200).reshape(-1, 1),
#    )
#X_new = np.c_[x0.ravel(), x1.ravel()]
#
#
#y_proba = softmax_reg.predict_proba(X_new)
#y_predict = softmax_reg.predict(X_new)
#
#zz1 = y_proba[:, 1].reshape(x0.shape)
#zz = y_predict.reshape(x0.shape)
#
#plt.figure(figsize=(10, 4))
#plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris-Virginica")
#plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris-Versicolor")
#plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris-Setosa")
#
#from matplotlib.colors import ListedColormap
#custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])
#
#plt.contourf(x0, x1, zz, cmap=custom_cmap)
#contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
#plt.clabel(contour, inline=1, fontsize=12)
#plt.xlabel("Petal length", fontsize=14)
#plt.ylabel("Petal width", fontsize=14)
#plt.legend(loc="center left", fontsize=14)
#plt.axis([0, 7, 0, 3.5])
#save_fig("softmax_regression_contour_plot")
#plt.show()

#softmax_reg.predict([[5, 2]])
#softmax_reg.predict_proba([[5, 2]])




