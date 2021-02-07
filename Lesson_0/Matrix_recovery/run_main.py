# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 10:41:30 2020

@author: brachan
"""

import pandas as pd
import numpy as np
# from utils import db_read, explode
# import cx_Oracle

import matplotlib.pyplot as plt
# =============================================================================
# import data
# =============================================================================

df = pd.read_csv('raw_data.csv', index_col =0)

# =============================================================================
# recommendation system by SVD
# =============================================================================


R_df = df.pivot(index = 'ID', columns ='ITEM_NAME', values = 'SCORE').fillna(0)

R_df.drop(columns=[np.nan],inplace = True)


# replace >1 to 1
R_df = (R_df > 0) * 1


R = R_df.values


user_ratings_mean = np.mean(R, axis = 1)

R_demeaned = R - user_ratings_mean.reshape(-1, 1)


# Singular Value Decomposition

from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 8)

sigma = np.diag(sigma)


# Making Predictions from the Decomposed Matrices

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Making Movie Recommendations

preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
preds_df.head()

# # =============================================================================
# # 
# # =============================================================================

# preds_df.columns = R_df.columns
# preds_df.index  = R_df.index

# R_df_head     = R_df.head(100)
# preds_df_head = preds_df.head(100)




# # =============================================================================
# # 
# # =============================================================================


# # see visiting history
# count_byitem = R_df.sum(axis = 1)

# A = preds_df.loc[count_byitem.index[count_byitem > 2],:]





































