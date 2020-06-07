# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 16:54:37 2019

@author: Brian Chan
"""

# chapter 5 : first SVM example

from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]

setosa_or_versicolor = (y == 0) | (y == 1)
X = X[setosa_or_versicolor]
y = y[setosa_or_versicolor]

# SVM Classifier model

svm_clf = SVC(kernel="linear", C=float("inf"))
svm_clf.fit(X, y)

#C = 0.01
#alpha = 1 / (C * len(X))
#svm_clf = SVC(kernel="linear", C=C)
#svm_clf.fit(X, y)

#svm_clf = SVC(kernel="rbf", C=float("inf"))
#svm_clf.fit(X, y)

# =============================================================================
# ## Bad models
# =============================================================================
x0 = np.linspace(0, 5.5, 200)
pred_1 = 5*x0 - 20
pred_2 = x0 - 1.8
pred_3 = 0.1 * x0 + 0.5


# =============================================================================
# display results
# =============================================================================
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)


plt.figure(figsize=(12,2.7))

plt.subplot(121)
plt.plot(x0, pred_1, "g--", linewidth=2)
plt.plot(x0, pred_2, "m-", linewidth=2)
plt.plot(x0, pred_3, "r-", linewidth=2)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", label="Iris-Versicolor")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", label="Iris-Setosa")
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 5.5, 0, 2])

plt.subplot(122)
plot_svc_decision_boundary(svm_clf, 0, 5.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo")
plt.xlabel("Petal length", fontsize=14)
plt.axis([0, 5.5, 0, 2])

#save_fig("large_margin_classification_plot")
plt.show()


# =============================================================================
# compare different solver
# =============================================================================

#from sklearn.svm import SVC, LinearSVC
#from sklearn.linear_model import SGDClassifier
#from sklearn.preprocessing import StandardScaler
#
#C = 5
#alpha = 1 / (C * len(X))
#
#lin_clf = LinearSVC(loss="hinge", C=C, random_state=42)
#svm_clf = SVC(kernel="linear", C=C)
#sgd_clf = SGDClassifier(loss="hinge", learning_rate="constant", eta0=0.001, alpha=alpha,
#                        max_iter=100000, tol=-np.infty, random_state=42)
#
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
#
#lin_clf.fit(X_scaled, y)
#svm_clf.fit(X_scaled, y)
#sgd_clf.fit(X_scaled, y)
#
#print("LinearSVC:                   ", lin_clf.intercept_, lin_clf.coef_)
#print("SVC:                         ", svm_clf.intercept_, svm_clf.coef_)
#print("SGDClassifier(alpha={:.5f}):".format(sgd_clf.alpha), sgd_clf.intercept_, sgd_clf.coef_)




