
##########################################################
#  Python script template for Question 2 (IAML Level 10)
#  Note that
#  - You should not change the name of this file, 'iaml01cw2_q2.py', which is the file name you should use when you submit your code for this question.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
#<----

# Q2.1
def iaml01cw2_q2_1():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    model = LogisticRegression()
    model.fit(Xtrn_nm, Ytrn)
    predictions = model.predict(Xtst_nm)
    conf_matrix = confusion_matrix(Ytst, predictions)
    acc_score = accuracy_score(Ytst, predictions)
    print(conf_matrix)
    print(acc_score)
# iaml01cw2_q2_1()   # comment this out when you run the function

# Q2.2
def iaml01cw2_q2_2():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    model = svm.SVC()
    model.fit(Xtrn_nm, Ytrn)
    predictions = model.predict(Xtst_nm)
    conf_matrix = confusion_matrix(Ytst, predictions)
    mean_acc_score = model.score(Xtst_nm, Ytst)
    print(conf_matrix)
    print(mean_acc_score)
# iaml01cw2_q2_2()   # comment this out when you run the function

# Q2.3
def iaml01cw2_q2_3():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    pca = PCA(n_components=2)
    X = pca.fit_transform(Xtrn_nm)

    model = LogisticRegression()
    model.fit(X, Ytrn)
    std1 = np.std(X[:, 0])
    std2 = np.std(X[:, 1])
    x_min, x_max = -5*std1, 5*std1
    y_min, y_max = -5*std2, 5*std2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, std1),np.arange(y_min, y_max, std2))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.show()
# iaml01cw2_q2_3()   # comment this out when you run the function

# Q2.4
def iaml01cw2_q2_4():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    pca = PCA(n_components=2)
    X = pca.fit_transform(Xtrn_nm)

    model = svm.SVC()
    model.fit(X, Ytrn)
    std1 = np.std(X[:, 0])
    std2 = np.std(X[:, 1])
    x_min, x_max = -5*std1, 5*std1
    y_min, y_max = -5*std2, 5*std2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, std1),np.arange(y_min, y_max, std2))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
    plt.show()
# iaml01cw2_q2_4()   # comment this out when you run the function

# Q2.5
def iaml01cw2_q2_5():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    # Obtain Xsmall and Ysmall
    Xsmall, Ysmall = [], []

    # occurrence[i] stores the occurrence of class/label i
    occurrence = [0 for i in range(10)]

    for i in range(0, 60000):
        label = Ytrn[i]
        if occurrence[label] < 1000: 
            Xsmall.append(Xtrn_nm[i])
            Ysmall.append(label)
            occurrence[label] += 1
        if len(Xsmall) == 10000:
            break
    Xsmall = np.array(Xsmall)
    Ysmall = np.array(Ysmall)

    # Perform 3-fold cross validation
    kf = KFold(n_splits=3)
    for train_index, test_index in kf.split(Xsmall):
        X_train, X_test = Xsmall[train_index], Xsmall[test_index]
        y_train, y_test = Ysmall[train_index], Ysmall[test_index]

    # Train SVC classifier with varying C
    model = svm.SVC()
    C_values = np.logspace(-2, 3, num=10)
    param_grid = {'C': C_values,
                  'gamma': ['auto'],
                  'kernel': ['rbf']}

    grid = GridSearchCV(model, param_grid, refit=True)
    grid.fit(X_train, y_train)

    # estimate the classification accuracy
    grid_predictions = grid.predict(X_test)
    print("the classification accuracy: " + str(accuracy_score(y_test, grid_predictions)))

    # Plot the mean cross-validated classification accuracy against C
    mean_accuracies = []
    for C_value in C_values:
        temp_model = svm.SVC(gamma='auto', kernel='rbf', C=C_value)
        temp_model.fit(X_train, y_train)
        mean_acc_score = temp_model.score(X_test, y_test)
        mean_accuracies.append(mean_acc_score)

    C_values_log = [round(np.log10(C_value),2) for C_value in C_values]
    scores = pd.DataFrame(mean_accuracies, columns=["mean cross-validated classification accuracy"])

    sns.set(style="white", rc={"lines.linewidth": 3})
    sns.barplot(x=C_values_log,y="mean cross-validated classification accuracy",data=scores)
    plt.xlabel('log C')
    plt.title("mean classification accuracy w.r.t log C")
    plt.show()

    print("the highest obtained mean accuracy score: ")
    print(grid.best_score_)
    print("the optimal value of C is given by the following estimator: ")
    print(grid.best_estimator_)

# iaml01cw2_q2_5()   # comment this out when you run the function

# Q2.6 
def iaml01cw2_q2_6():
    Xtrn, Ytrn, Xtst, Ytst = get_og_data()
    Xtrn_nm, Xtst_nm = get_normalised_data()

    model = svm.SVC(gamma='auto', kernel='rbf', C=77.42636826811278)
    model.fit(Xtrn_nm, Ytrn)
    predictions_trn = model.predict(Xtrn_nm)
    predictions_tst = model.predict(Xtst_nm)
    acc_score_trn = accuracy_score(Ytrn, predictions_trn)
    acc_score_tst = accuracy_score(Ytst, predictions_tst)
    print(acc_score_trn)
    print(acc_score_tst)
# iaml01cw2_q2_6()   # comment this out when you run the function

