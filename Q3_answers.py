
##########################################################
#  Python script template for Question 3 (IAML Level 10)
#  Note that:
#  - You should not change the name of this file, 'iaml01cw2_q3.py', which is the file name you should use when you submit your code for this question.
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
from helper_function import *
from helper_function_2 import *

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

#<----

# Q3.1
def iaml01cw2_q3_1():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data_speech()

    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)

    print("The sum of squared distances of samples to their closest cluster centre: " + str(kmeans.inertia_))

    # clusters[x] is an array of the indices of samples whose cluster is a label x
    clusters = [[] for i in range(22)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    for label in range(0, 22):
        print("Number of samples for cluster " + str(label+1) + ": " + str(len(clusters[label])))
# iaml01cw2_q3_1()   # comment this out when you run the function

# Q3.2
def iaml01cw2_q3_2():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data_speech()

    kmeans = KMeans(n_clusters=22, random_state=1).fit(Xtrn)

    # clusters[x] is an array of the indices of samples whose cluster is a label x
    clusters = [[] for i in range(22)]
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    mean_vectors = []
    for label in range(0, 22):
        mean_vector = [0 for i in range(26)]
        for idx in clusters[label]:
            mean_vector = mean_vector + Xtrn[idx]
        mean_vectors.append(mean_vector / len(clusters[label]))
    mean_vectors = np.array(mean_vectors)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(mean_vectors)
    X_pca_cc = pca.fit_transform(kmeans.cluster_centers_)

    # each color represents a language (0 - 21)
    colors = ['IndianRed','Salmon','Red','DarkRed','Pink','HotPink','DeepPink','SeaGreen','Green',
             'YellowGreen','Olive','Aqua','Gold','Khaki','SteelBlue','Blue','Orange','OrangeRed',
             'Lime','Teal','Navy','SpringGreen','LightYellow','Yellow','CadetBlue','DeepSkyBlue',
             'LimeGreen','PaleVioletRed','LightCoral','PeachPuff','OliveDrab']

    for i in range(0,22):
        plt.scatter(X_pca_cc[i][0], X_pca_cc[i][1], color='black') # plot cluster centres
        plt.scatter(X_pca[i][0], X_pca[i][1], color=colors[i]) # plot mean vectors
    plt.tight_layout()
    plt.legend(["Language " + str(i) + ": " + colors[i] for i in range(22)], prop={'size': 8}, loc="lower center", bbox_to_anchor=(1.2, -0.15))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()
# iaml01cw2_q3_2()   # comment this out when you run the function

# Q3.3
def iaml01cw2_q3_3():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data_speech()

    # samples[x] is an array of the indices of samples whose label is x
    samples = [[] for i in range(22)]
    for i in range(0, 22000):
        label = Ytrn[i]
        samples[label].append(i)

    mean_vectors = []
    for label in range(0, 22):
        mean_vector = [0 for i in range(26)]
        for idx in samples[label]:
            mean_vector = mean_vector + Xtrn[idx]
        mean_vectors.append(mean_vector / len(samples[label]))
    mean_vectors = np.array(mean_vectors)

    linkage_matrix = linkage(mean_vectors, "ward")
    dendrogram = dendrogram(linkage_matrix, orientation="right")

    plt.title("Hierarchical Clustering")
    plt.show()
# iaml01cw2_q3_3()   # comment this out when you run the function

# Q3.4
def iaml01cw2_q3_4():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data_speech()
    kmeans = KMeans(n_clusters=3, random_state=1).fit(Xtrn)
    centers = kmeans.cluster_centers_

    linkage_matrix_single = linkage(centers, "single")
    dendrogram_single = dendrogram(linkage_matrix_single)
    plt.title("Hierarchical Clustering")
    plt.show()

    linkage_matrix_complete = linkage(centers, "complete")
    dendrogram_complete = dendrogram(linkage_matrix_complete)
    plt.title("Hierarchical Clustering")
    plt.show()

    linkage_matrix_ward = linkage(centers, "ward")
    dendrogram_ward = dendrogram(linkage_matrix_ward)
    plt.title("Hierarchical Clustering")
    plt.show()
# iaml01cw2_q3_4()   # comment this out when you run the function

# Q3.5
def iaml01cw2_q3_5():
#
# iaml01cw2_q3_5()   # comment this out when you run the function

