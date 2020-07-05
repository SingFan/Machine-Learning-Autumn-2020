# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 13:38:03 2019

@author: Brian Chan
"""

from six.moves import urllib
import numpy as np

try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
    
from sklearn.model_selection import train_test_split

X = mnist["data"]
y = mnist["target"]

#Reshaped_X = np.array(X[0,:]).reshape(28,28)

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(cumsum)
plt.title('Cumulative variance explained along n')
plt.show()

d = np.argmax(cumsum >= 0.95) + 1

pca       = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)


pca.n_components_
np.sum(pca.explained_variance_ratio_)

# List_n_components = np.array([3,5,6,9,10,11,20,30,40,50,60,70,90,120,150,200])
List_n_components = np.array([3,20,200])


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")


for i in List_n_components:
    pca = PCA(n_components = i)
    X_reduced   = pca.fit_transform(X_train)
    X_recovered = pca.inverse_transform(X_reduced)
    
    import matplotlib as mpl
    
    plt.figure(figsize=(7, 4))
    plot_digits(X_recovered[::2100])
    plt.title("Compressed" + str(i), fontsize=16)
    
    plt.savefig("mnist_compression_plot"+str(i))





