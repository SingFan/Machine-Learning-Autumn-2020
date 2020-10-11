# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:29:05 2019

@author: Brian Chan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()


df_cancer.shape
df_cancer.columns

# Let's plot out just the first 5 variables (features)

sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area'] )
sns.pairplot(df_cancer, hue = 'target', vars = ['mean radius', 'mean texture', 'mean perimeter','mean area'] )

# =============================================================================
# 
# =============================================================================

df_cancer['target'].value_counts()
sns.countplot(df_cancer['target'], label = "Count")

plt.figure(figsize=(20,12)) 
sns.heatmap(df_cancer.corr(), annot=True)

# =============================================================================
# prepare SVM
# =============================================================================

X = df_cancer.drop(['target'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()

y = df_cancer['target']
y.head()


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)


from sklearn.svm import SVC

svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix

print("Try first SVM ")

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
print(confusion)

sns.heatmap(confusion, annot=True)
print('/n /n')
print(classification_report(y_test, y_predict))

# =============================================================================
# Normalization
# =============================================================================

X_train_min = X_train.min()
X_train_max = X_train.max()

X_train_range = (X_train_max- X_train_min)

X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()

X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range



svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)

print("Try second SVM with normalization")


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
print(confusion)

print(classification_report(y_test,y_predict))



# =============================================================================
# grid search
# =============================================================================

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)

print (grid.best_params_)
print('/n /n')
print (grid.best_estimator_)


# =============================================================================
# performance of the best model
# =============================================================================

print("Try best SVM of having grid search with normalization")

grid_predictions = grid.predict(X_test_scaled)

cm = np.array(confusion_matrix(y_test, grid_predictions, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
print(confusion)
print('/n /n')
print(classification_report(y_test,grid_predictions))


# =============================================================================
# Remarks:
# =============================================================================


#    mean radius = mean of distances from center to points on the perimeter
#    mean texture = standard deviation of gray-scale values
#    mean perimeter = mean size of the core tumor
#    mean area =
#    mean smoothness = mean of local variation in radius lengths
#    mean compactness = mean of perimeter^2 / area - 1.0
#    mean concavity = mean of severity of concave portions of the contour
#    mean concave points = mean for number of concave portions of the contour
#    mean symmetry =
#    mean fractal dimension = mean for "coastline approximation" - 1
#    radius error = standard error for the mean of distances from center to points on the perimeter
#    texture error = standard error for standard deviation of gray-scale values
#    perimeter error =
#    area error =
#    smoothness error = standard error for local variation in radius lengths
#    compactness error = standard error for perimeter^2 / area - 1.0
#    concavity error = standard error for severity of concave portions of the contour
#    concave points error = standard error for number of concave portions of the contour
#    symmetry error =
#    fractal dimension error = standard error for "coastline approximation" - 1
#    worst radius = "worst" or largest mean value for mean of distances from center to points on the perimeter
#    worst texture = "worst" or largest mean value for standard deviation of gray-scale values
#    worst perimeter =
#    worst smoothness = "worst" or largest mean value for local variation in radius lengths
#    worst compactness = "worst" or largest mean value for perimeter^2 / area - 1.0
#    worst concavity = "worst" or largest mean value for severity of concave portions of the contour
#    worst concave points = "worst" or largest mean value for number of concave portions of the contour
#    worst fractal dimension = "worst" or largest mean value for "coastline approximation" - 1







#TERM 	DESCRIPTION
#TRUE POSITIVES 	The number of "true" classes correctly predicted to be true by the model.
#
#TP = Sum of observations predicted to be 1 that are actually 1
#
#The true class in a binary classifier is labeled with 1.
#TRUE NEGATIVES 	The number of "false" classes correctly predicted to be false by the model.
#
#TN = Sum of observations predicted to be 0 that are actually 0
#
#The false class in a binary classifier is labeled with 0.
#FALSE POSITIVES 	The number of "false" classes incorrectly predicted to be true by the model. This is the measure of Type I error.
#
#FP = Sum of observations predicted to be 1 that are actually 0
#
#Remember that the "true" and "false" refer to the veracity of your guess, and the "positive" and "negative" component refer to the guessed label.
#FALSE NEGATIVES 	The number of "true" classes incorrectly predicted to be false by the model. This is the measure of Type II error.
#
#FN = Sum of observations predicted to be 0 that are actually 1
#
#TOTAL POPULATION 	In the context of the confusion matrix, the sum of the cells.
#
#total population = tp + tn + fp + fn
#
#SUPPORT 	The marginal sum of rows in the confusion matrix, or in other words the total number of observations belonging to a class regardless of prediction.
#
#ACCURACY 	The number of correct predictions by the model out of the total number of observations.
#
#accuracy = (tp + tn) / total_population
#
#PRECISION 	The ability of the classifier to avoid labeling a class as a member of another class.
#
#Precision = True Positives / (True Positives + False Positives)
#
#A precision score of 1 indicates that the classifier never mistakenly classified the current class as another class. precision score of 0 would mean that the classifier misclassified every instance of the current class
#RECALL/SENSITIVITY 	The ability of the classifier to correctly identify the current class.
#
#Recall = True Positives / (True Positives + False Negatives)
#
#A recall of 1 indicates that the classifier correctly predicted all observations of the class. 0 means the classifier predicted all observations of the current class incorrectly.
#SPECIFICITY 	Percent of times the classifier predicted 0 out of all the times the class was 0.
#
#specificity = tn / (tn + fp)
#
#FALSE POSITIVE RATE 	Percent of times model predicts 1 when the class is 0.
#
#fpr = fp / (tn + fp)
#
#F1-SCORE 	The harmonic mean of the precision and recall. The harmonic mean is used here rather than the more conventional arithmetic mean because the harmonic mean is more appropriate for averaging rates.
#
#F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
#
#The f1-score's best value is 1 and worst value is 0, like the precision and recall scores. It is a useful metric for taking into account both measures at once.
#

