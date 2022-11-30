# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:46:53 2022

@author: isabe
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:52:42 2022

@author: Isabel Thomas isabel.thomas@unige.ch
"""
##import modules 
import pandas as pd
import os
import numpy as np
import sklearn
import random

#%% date set-up
from datetime import datetime

# current dateTime
now = datetime.now()

# convert to string
date_time_str = now.strftime("%d%m")
print('DateTime String:', date_time_str)

#%% 1. set up file path
ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"

#%% 2. read in pre-prepared data
# where each instance has been assigned
# to a cell as descibed in methods

#data for 1985 contains data from 2 landsat images
#data for 1997 contains data from 5 landsat images

df = pd.read_csv(os.path.join(ml_dir, "S2/points_extract_2018_6.csv"))

#%%
l = {'code':["11","12","13","14","15","16","17","21","31","32","33","34","35","41","42","43","44","45","46","47","51","52","53","61","62","63","64"],
        'desc':['Consolidated surfaces','Buildings','Greenhouses','Gardens with border and patch structures',
               'Lawns','Trees in artificial areas','Mix of small structures','Grass and herb vegetation',
               'Shrubs','Brush meadows','Short-stem fruit trees','Vines','Permanent garden plants and brush crops',
               'Closed forest','Forest edges','Forest strips','Open forest','Brush forest','Linear woods',
               'Clusters of trees','Solid rock','Granular soil','Rocky areas', 'Water', 'Glacier, perpetual snow','Wetlands',
               'Reedy marshes']}
lkup=pd.DataFrame(data=l)
df = pd.merge(df,lkup,left_on='label', right_on='desc', how='left')

#%% 4. clean up data

df = df[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11','code','RELI']]

#%%
#NDWI = (B03 - B08) / (B03 + B08)
#NDBI = (B11 - B08) / (B11 + B08)

def getIndicesS2(df):
    #NDVI = (NIR - RED) / (NIR + RED)
    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"])

    #NDBI = (SWIR – NIR) / (SWIR + NIR)
    df["NDBI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])

    #NDWI = (G – NIR) / (G + NIR)
    df["NDWI"] = (df["B3"] - df["B8"]) / (df["B3"] + df["B8"])
    
    return df

df = getIndicesS2(df)

#%%
valid = "S2/validation/samplesOFS_gva_valid_27.csv"
valid_df = pd.read_csv(os.path.join(ml_dir, valid))
print(valid_df.shape)
df_valid = df[df['RELI'].isin(valid_df['RELI'])]
print(df_valid.shape)
df_train = df[~df['RELI'].isin(valid_df['RELI'])]
print(df_train.shape)

#%% 6. Split dataset into X (inputs) and y (labels)

#training/validation
X_tv = df_tv[var]
X_tv = X_tv.to_numpy()
y_tv = df_tv[['LC']]
y_tv = y_tv.values.ravel()
#test
X_test = df_test[var]
X_test = X_test.to_numpy()
y_test = df_test[['LC']]
y_test = y_test.values.ravel()

#%% 7. Print class distributions
print(pd.value_counts(y_tv))
print(pd.value_counts(y_test))

#%% 8. Define training/validation splits

# Grouped K-fold cross validation is used to maintain the predefined group 
# separation to avoid overestimation of model due to spatial autocorrelation
# from selecting training & validation data from neighbouring datapoints. 

from sklearn.model_selection import GroupKFold

#unique identifier for grid cell ('index_right' as grouping variable)
groups = df_tv['index_right'].values
#initiate GroupKFold with 7 folds
group_kfold = GroupKFold(n_splits=7) 

# Generator for the train/test indices
fold_kfold = group_kfold.split(X_tv, y_tv, groups)  

# Create a nested list of train and test indices for each fold
train_indices, val_indices = [list(trainval) for trainval in zip(*fold_kfold)]

fold_cv = [*zip(train_indices, val_indices)]

#%% Optional: check train/validation average class distribution
#y_tv = df_tv[['LC']]
#y_train_vals = ((pd.value_counts(y_tv.iloc[fold_cv[0][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[1][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[3][0]].values.ravel()))/7)
#y_valid_vals = ((pd.value_counts(y_tv.iloc[fold_cv[0][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[1][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][1]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[2][0]].values.ravel())+
                #pd.value_counts(y_tv.iloc[fold_cv[3][1]].values.ravel()))/7)

#print(y_train_vals)
#print(y_valid_vals)
#y_tv = y_tv.values.ravel()

#%% 10. Initiate RF classifer, train model whilst performing hyperparameter search
### currently takes ~6 hours ###
from sklearn.ensemble import RandomForestClassifier

import time

#initiate timer for whole classification
t0 = time.time()

#initiate classifier
rfc = RandomForestClassifier(random_state=42)

#run with accuracy as performance metric
#grid_search = HalvingGridSearchCV(rfc, param_grid, cv=fold_cv, verbose=3)

#to run with f1 score as performance metric (f1_weighted as multiclass classification)
grid_search = HalvingGridSearchCV(rfc, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)

grid_search.fit(X_tv, y_tv)

#print best parameters
print("The best parameters are %s with a score of %0.2f"
    % (grid_search.best_params_, grid_search.best_score_))

#calculate timer
t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  

#%% 9. Save RF results
results = grid_search.cv_results_
df = pd.DataFrame(results) 
df.to_csv(os.path.join(ml_dir,'grid_search_f1_',date_time_str, '.csv'))
#df.to_csv('C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/grid_search_f1-2705.csv')

#%% 10. Predict test set values
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score 

y_pred = grid_search.predict(X_test)

# Performance metrics on test set
test_acc = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred, average="weighted")

print(f'The accuracy of the model on the test set is {test_acc:.1%}')
print(f'The f1 score of the model on the test set is {test_f1:.1%}')

#%% 11. Return class-based metrics
from sklearn.metrics import confusion_matrix, classification_report

#(to load in pre-saved results)
#test_0409 = pd.read_csv(os.path.join(ml_dir, "results_for_test_2004-2009_2505.csv"))
#y_pred = test_0409['preds']

y_actu = pd.Series(y_test, name='Actual')
y_pred = pd.Series(y_pred, name='Predicted')

print(classification_report(y_test, y_pred))

#confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)
#precision
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
#recall
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
matrix = confusion_matrix(y_test, y_pred, normalize='pred')

#%% 12. save predictions and probabilities

all_probs = np.max(grid_search.predict_proba(X_test), axis=1)
df_test['prob'] = all_probs
df_test['predicted'] = y_pred

df_test.to_csv(os.path.join(ml_dir, "results_for_test_2004-2009_",date_time_str,".csv"))
#%% 13. Calculate kappa score
from sklearn.metrics import cohen_kappa_score
kscore = cohen_kappa_score(y_test, y_pred)
print(kscore)

#%% 14. Second test set (2004 Landsat image data)
test_0409 = pd.read_csv(os.path.join(ml_dir, "data/test_2004-2009.csv"))
X_test2 = df_test[var]
X_test2 = X_test2.to_numpy()
y_test2 = df_test[['LC09R_6']]
y_test2 = y_test2.values.ravel()

test_preds = grid_search.predict(X_test2)

test_acc = accuracy_score(y_test2, test_preds)
test_f1 = f1_score(y_test2, test_preds, average="weighted")

print(f'The accuracy of the model on the test set is {test_acc:.1%}')
print(f'The f1 score of the model on the test set is {test_f1:.1%}')
#%% 15. Calculate class-based performance metrics
from sklearn.metrics import confusion_matrix
y_actu = pd.Series(y_test2, name='Actual')
y_pred = pd.Series(test_preds, name='Predicted')

#confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)
#precision
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
#recall
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
matrix = confusion_matrix(y_test, y_pred, normalize='pred')
#%% 16. Run Multinomial Logistic Regression model with hyperparameter search
#parameter grid
param_grid = {'solver': ['saga', 'sag'],
               'C': [100, 10, 1.0, 0.1, 0.01]}

from sklearn.linear_model import LogisticRegression
import time
t0 = time.time()

log_mod = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=4000)

from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV 

grid_search = HalvingGridSearchCV(log_mod, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)
#grid_search = HalvingGridSearchCV(log_mod, param_grid, cv=fold_cv, scoring='f1_weighted', verbose=3)
grid_search.fit(X_tv, y_tv)
t1 = time.time()
results = grid_search.cv_results_
df = pd.DataFrame(results) 
df.to_csv(os.path.join(ml_dir,'grid_search_lm_f1', date_time_str, '.csv')
#df.to_csv(os.path.join(ml_dir,'grid_search_f1_',date_time_str, '.csv'))
#%%  MLR predictions on test set
test_preds1 = grid_search.predict(X_test)

test_acc = accuracy_score(y_test, test_preds1)
test_f1 = f1_score(y_test, test_preds1, average="weighted")

print(f'The accuracy of the model on the test set (1) is {test_acc:.1%}')
print(f'The f1 score of the model on the test set (1) is {test_f1:.1%}')

test_preds2 = grid_search.predict(X_test2)

test_acc = accuracy_score(y_test2, test_preds2)
test_f1 = f1_score(y_test2, test_preds2, average="weighted")

print(f'The accuracy of the model on the test set (2) is {test_acc:.1%}')
print(f'The f1 score of the model on the test set (2) is {test_f1:.1%}')

#%%
 
from sklearn.svm import SVC