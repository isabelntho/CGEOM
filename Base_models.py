# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 10:46:53 2022

@author: Isabel Thomas isabel.thomas@unige.ch
"""
#%% 1. import modules 
import pandas as pd
import os
import numpy as np
import sklearn
import random
import time

#%% 2. set up file path
ml_dir = "C:/Users/isabe/Documents/UNIGE/S4/Machine Learning/"

#%% 3. read in pre-prepared data
# where each instance has been assigned
# to a cell as descibed in methods
df = pd.read_csv(os.path.join(ml_dir, "S2/grid_extract_2018_6.csv"))
#%% 4. Join in LC label data
l27 = {'code':["11","12","13","14","15","16","17","21","31","32","33","34","35","41","42","43","44","45","46","47","51","52","53","61","62","63","64"],
        'desc':['Consolidated surfaces','Buildings','Greenhouses','Gardens with border and patch structures',
               'Lawns','Trees in artificial areas','Mix of small structures','Grass and herb vegetation',
               'Shrubs','Brush meadows','Short-stem fruit trees','Vines','Permanent garden plants and brush crops',
               'Closed forest','Forest edges','Forest strips','Open forest','Brush forest','Linear woods',
               'Clusters of trees','Solid rock','Granular soil','Rocky areas', 'Water', 'Glacier, perpetual snow','Wetlands',
               'Reedy marshes']}

l6 = {'code':["10","20","30","40","50","60"],
        'desc':['Artificial areas', 'Grass and herb vegetation', 
                'Brush vegetation', 'Tree vegetation', 'Bare land', 
                'Watery areas']}

lkup=pd.DataFrame(data=l6)
df = pd.merge(df,lkup,left_on='label', right_on='desc', how='left')

#%% 5. clean up data

df = df[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'code','RELI']]

test=df[df.isnull().any(axis=1)]
df = df[~df['RELI'].isin(test['RELI'])]
#%% 6. Calculate indices
def getIndicesS2(df):
    #NDVI = (NIR - RED) / (NIR + RED)
    df["NDVI"] = (df["B8"] - df["B4"]) / (df["B8"] + df["B4"])

    #NDBI = (SWIR – NIR) / (SWIR + NIR)
    df["NDBI"] = (df["B11"] - df["B8"]) / (df["B11"] + df["B8"])

    #NDWI = (G – NIR) / (G + NIR)
    df["NDWI"] = (df["B3"] - df["B8"]) / (df["B3"] + df["B8"])
    
    return df

df = getIndicesS2(df)

#%% 7. Split into valid and training
valid = "S2/validation/samplesOFS_gva_valid_6.csv"
df_valid_csv = pd.read_csv(os.path.join(ml_dir, valid))
#print(valid_df.shape)
df_valid = df[df['RELI'].isin(df_valid_csv['RELI'])]
print(df_valid.shape)
df_train = df[~df['RELI'].isin(df_valid_csv['RELI'])]
print(df_train.shape)

#%% 8. Split dataset into X (inputs) and y (labels)

var = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'NDVI', 'NDBI', 'NDWI']
#training/validation
X_train = df_train[var]
X_train = X_train.to_numpy()
y_train = df_train[['code']]
y_train = y_train.values.ravel()
#test
X_valid = df_valid[var]
X_valid = X_valid.to_numpy()
y_valid = df_valid[['code']]
y_valid = y_valid.values.ravel()

#%% 9. Print class distributions
print(pd.value_counts(y_train))
print(pd.value_counts(y_valid))

#%% 10. Run RF model
from sklearn.ensemble import RandomForestClassifier

#initiate timer for whole classification
t0 = time.time()

#initiate classifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
rfc_preds = rfc.predict(X_valid)
rfc_acc = accuracy_score(y_valid, rfc_preds)
from sklearn.metrics import f1_score 
rfc_f1 = f1_score(y_valid, rfc_preds, average="weighted")

print(rfc_acc)
print(rfc_f1)

t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  

#%% 11. Calculate feature importances
#RUN THIS ONE

from sklearn.inspection import permutation_importance
result = permutation_importance(
    rfc, X_train, y_train, n_repeats=10, random_state=42, n_jobs=2
)

#%% 12. Test model on 2021 data

df_21 = pd.read_csv(os.path.join(ml_dir, "S2/grid_extract_2021_6.csv"))
df_21 = pd.merge(df_21,lkup,left_on='label', right_on='desc', how='left')
df_21 = df_21[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'code','RELI']]

df_21 = getIndicesS2(df_21)
df_21 = df_21[~df_21['RELI'].isin(test['RELI'])]
#test
X_test = df_21[var]
X_test = X_test.to_numpy()
y_test = df_21[['code']]
y_test = y_test.values.ravel()

#%% 13. Predictions
test_preds = rfc.predict(X_test)
test_acc = accuracy_score(y_test, test_preds)
test_f1 = f1_score(y_test, test_preds, average="weighted")

print(test_acc)
print(test_f1)
pts = pd.read_csv(os.path.join(ml_dir, "S2/grid_extract_2021_6.csv"))
pts = pts[~pts['RELI'].isin(test['RELI'])]
pts['Predictions'] = test_preds

#%% 14. SVC model
from sklearn.svm import SVC

#initiate timer for whole classification
t0 = time.time()

#initiate classifier
svc = SVC(probability=True)
svc.fit(X_train, y_train)

svc_preds = svc.predict(X_valid)
svc_acc = accuracy_score(y_valid, svc_preds)
print(svc_acc)

from sklearn.metrics import f1_score 
svc_f1 = f1_score(y_valid, svc_preds, average="weighted")

t1 = time.time()
print(f"{(t1 - t0):.2f}s elapsed")  
#%% 15. Logistic regression model
from sklearn.linear_model import LogisticRegression

log_mod = LogisticRegression(random_state=0, multi_class='multinomial', max_iter=4000)

log_mod.fit(X_train, y_train)

log_preds = log_mod.predict(X_valid)
log_acc = accuracy_score(y_valid, log_preds)
log_f1 = f1_score(y_valid, log_preds, average="weighted")

print(log_acc)
print(log_f1)

#%% 16. Confusion matrix
from sklearn.metrics import confusion_matrix
y_actu = pd.Series(y_valid, name='Actual')
y_pred = pd.Series(rfc_preds, name='Predicted')

#confusion matrix
df_confusion = pd.crosstab(y_actu, y_pred)
#precision
df_conf_norm_a = df_confusion / df_confusion.sum(axis=1)
#recall
df_conf_norm_b = df_confusion / df_confusion.sum(axis=0)
#matrix = confusion_matrix(y_test, y_pred, normalize='pred')