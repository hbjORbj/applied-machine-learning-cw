
##########################################################
#  Python script template for Question 1 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q1.py', which is the file name you should use when you submit your code for this question.
#  - You should write code for the functions defined below. Do not change their names.
#  - You can define function arguments (parameters) and returns (attributes) if necessary.
#  - In case you define additional functions, do not define them here, but put them in a separate Python module file, "iaml01cw2_my_helpers.py", and import it in this script.
#  - For those questions requiring you to show results in tables, your code does not need to present them in tables - just showing them with print() is fine.
#  - You do not need to include this header in your submission.
##########################################################

#--- Code for loading modules and the data set and pre-processing --->
# NB: You can edit the following and add code (e.g. code for loading sklearn) if necessary.

import os
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from iaml01cw2_helpers import *
from iaml01cw2_my_helpers import *

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

#<----

# Q1.1
def iaml01cw2_q1_1():
    Xtrn_nm, Xtst_nm = get_normalised_data()

    print(Xtrn_nm[0, :4])
    print(Xtrn_nm[-1,:4])
# iaml01cw2_q1_1()   # comment this out when you run the function

# Q1.2
def iaml01cw2_q1_2():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    # samples[i] is an array of the indices of samples whose label is i
    samples = [[] for i in range(10)]

    for i in range(0, 60000):
        label = Ytrn[i]
        samples[label].append(i)

    # mean_vectors[i] is the mean vector of class i
    mean_vectors = []

    for label in range(0, 10):
        mean_vector = [0 for i in range(784)]
        for idx in samples[label]:
            mean_vector = mean_vector + Xtrn_nm[idx]
        mean_vectors.append(mean_vector / len(samples[label]))
    mean_vectors = np.array(mean_vectors)

    # image_indices[i] is an array of the indices of nearest, second_nearest, second_farthest, farthest samples 
    # from the mean vector of class i

    image_indices = [[] for i in range(10)]

    for i in range(0, 10):
        distances = [get_distance(Xtrn_nm[idx], mean_vectors[i]) for idx in samples[i]]
        min_idx = np.argmin(distances)
        nearest_idx = samples[i][min_idx]

        max_idx = np.argmax(distances)
        farthest_idx = samples[i][max_idx]

        del samples[i][min_idx] # delete the position of the index for nearest sample
        del samples[i][max_idx] # delete the position of the index for farthest sample

        distances = [get_distance(Xtrn_nm[idx], mean_vectors[i]) for idx in samples[i]]
        min_idx = np.argmin(distances)
        second_nearest_idx = samples[i][min_idx]

        max_idx = np.argmax(distances)
        second_farthest_idx = samples[i][max_idx]

        image_indices[i].append(nearest_idx)
        image_indices[i].append(second_nearest_idx)
        image_indices[i].append(second_farthest_idx)
        image_indices[i].append(farthest_idx)

    # plot the images: each image is 28 by 28 pixels

    fig = plt.figure(figsize=(5,10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for label in range(0, 10):
        ax = fig.add_subplot(10, 5, 1+label*5, xticks=[], yticks=[])
        ax.imshow(np.reshape(mean_vectors[label], (28,28)), cmap=plt.cm.gray_r)
        for i in range(0,4):
            idx = image_indices[label][i]
            ax = fig.add_subplot(10, 5, i+2+label*5, xticks=[], yticks=[])
            ax.imshow(np.reshape(Xtrn_nm[idx], (28,28)), cmap=plt.cm.gray_r)
    df = pd.DataFrame(image_indices, columns=["nearest","second nearest", "second farthest", "farthest"])     
    print(df)
# iaml01cw2_q1_2()   # comment this out when you run the function

# Q1.3
    pca = PCA()
    pca.fit(Xtrn_nm)
    # Row number represents which principal component is used to project the data (e.g. 3 means the third principal component)
    # and each cell represents its explained variance 
    variances = pca.explained_variance_[0:5]
    df = pd.DataFrame(variances, ['1','2','3','4','5'], ["variance"])     
    print(df)
# iaml01cw2_q1_3()   # comment this out when you run the function


# Q1.4
def iaml01cw2_q1_4():
    Xtrn_nm, Xtst_nm = get_normalised_data()

    pca = PCA().fit(Xtrn_nm)
    plt.figure(figsize=(12,8))
    plt.plot([i for i in range(1,785)], np.cumsum(pca.explained_variance_ratio_), color='green')
    plt.xlabel('number of principal components')
    plt.ylabel('cumulative explained variance ratio')
    plt.title('cumulative explained variance ratio vs number of components')
    plt.show()
# iaml01cw2_q1_4()   # comment this out when you run the function


# Q1.5
def iaml01cw2_q1_5():
    Xtrn_nm, Xtst_nm = get_normalised_data()

    pca = PCA(10)
    X_proj = pca.fit_transform(Xtrn_nm)

    fig = plt.figure(figsize=(5,2))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the images: each image is 28 by 28 pixels
    for i in range(0,10):
        ax = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        ax.imshow(np.reshape(pca.components_[9-i,:], (28,28)), cmap=plt.cm.gray_r)
# iaml01cw2_q1_5()   # comment this out when you run the function


# Q1.6
def iaml01cw2_q1_6():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    # samples[i] stores the index of the first sample of class i 
    samples = [None for i in range(10)]
    count = 0

    for i in range(0, 60000):
        label = Ytrn[i]
        if samples[label] == None:
            samples[label] = i
            count += 1
        if count == 10:
            break

    # table[x][y] represents the RMSE between the original sample in Xtrn_nm 
    # and reconstructed one using K_values[y] principal components for the first sample of class x       
    table = [[None for i in range(4)] for i in range(10)]
    K_values = [5, 20, 50, 200]

    for label in range(0, 10):
        for i in range(0, 4):
            idx = samples[label]
            K = K_values[i]
            pca = PCA(n_components=K)
            X_proj = pca.fit_transform(Xtrn_nm)[idx]

            # we inverse transform the reduced dimensionality back into the original dimensionality
            X_inv_proj = pca.inverse_transform(X_proj)

            original = Xtrn_nm[idx]
            reconstructed = X_inv_proj
            rmse = mean_squared_error(original, reconstructed, squared=False)
            table[label][i] = rmse

    df = pd.DataFrame(table, columns=K_values)     
    print(df)
# iaml01cw2_q1_6()   # comment this out when you run the function


# Q1.7
def iaml01cw2_q1_7():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()
    Xmean = Xtrn - Xtrn_nm

    # samples[i] stores the index of the first sample of class i 
    samples = [None for i in range(10)]
    count = 0

    for i in range(0, 60000):
        label = Ytrn[i]
        if samples[label] == None:
            samples[label] = i
            count += 1
        if count == 10:
            break

    # table[x][y] represents a reconstructed sample using K_values[y] principal components for the first sample of class x
    # plus the first sample of class x of Xmean
    table = [[None for i in range(4)] for i in range(10)]
    K_values = [5, 20, 50, 200]

    for label in range(0, 10):
        for i in range(0, 4):
            idx = samples[label]
            K = K_values[i]
            pca = PCA(n_components=K)
            X_proj = pca.fit_transform(Xtrn_nm)[idx]

            # we inverse transform the reduced dimensionality back into the original dimensionality
            reconstructed = pca.inverse_transform(X_proj)
            table[label][i] = reconstructed + Xmean[idx]

    fig = plt.figure(figsize=(4,10))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the images: each image is 28 by 28 pixels
    for label in range(0, 10):
        for i in range(0,4):
            ax = fig.add_subplot(10, 4, i+1+label*4, xticks=[], yticks=[])
            ax.imshow(np.reshape(table[label][i], (28,28)), cmap=plt.cm.gray_r)
# iaml01cw2_q1_7()   # comment this out when you run the function

# Q1.8
def iaml01cw2_q1_8():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(Xtrn_nm)

    # each color represents an image label (0 - 9)
    for i in range(0,10):
        px = X_pca[:, 0][Ytrn == i]
        py = X_pca[:, 1][Ytrn == i]
        plt.scatter(px, py, cmap=plt.cm.coolwarm)
    plt.tight_layout()
    plt.legend([0,1,2,3,4,5,6,7,8,9], prop={'size': 6})
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
# iaml01cw2_q1_8()   # comment this out when you run the function
