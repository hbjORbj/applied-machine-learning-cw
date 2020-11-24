##########################################################
#  Python module template for helper functions of your own (IAML Level 10)
#  Note that:
#  - Those helper functions of your own for Questions 1, 2, and 3 should be defined in this file.
#  - You can decide function names by yourself.
#  - You do not need to include this header in your submission.
##########################################################

import os
import numpy as np
from iaml01cw2_helpers import *

def get_og_data():
    data_path = os.path.join(os.getcwd(), '../data')
    Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(data_path)
    return Xtrn, Ytrn, Xtst, Ytst

def get_normalised_data():
    data_path = os.path.join(os.getcwd(), '../data')
    Xtrn, Ytrn, Xtst, Ytst = load_FashionMNIST(data_path)

    Xtrn_orig = Xtrn.copy()
    Xtst_orig = Xtst.copy()

    Xtrn = Xtrn / 255.0
    Xtst = Xtst / 255.0

    Xmean = np.cumsum(Xtrn, axis=0)[59999] / 60000

    Xtrn_nm = Xtrn - Xmean
    Xtst_nm = Xtst - Xmean
    return Xtrn_nm, Xtst_nm

def get_og_data_speech():
    data_path = os.path.join(os.getcwd(), '../data')
    Xtrn, Ytrn, Xtst, Ytst = load_CoVoST2(data_path)
    return Xtrn, Ytrn, Xtst, Ytst

# Return the Euclidean distance between two vectors
def get_distance(a, b):
    return np.sqrt(((a - b)**2).sum())