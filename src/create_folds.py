# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:46:37 2020

@author: nahil
"""


import pandas as pd
from sklearn import model_selection

df = pd.read_csv("../input/mnist_train.csv")
print("Shape of MNIST:",df.shape,sep=" ")

df["kfold"] = -1

df = df.sample(frac = 1).reset_index(drop=True)

kf = model_selection.StratifiedKFold(n_splits=5)

for fold,(i,val) in enumerate(kf.split(X=df,y=df.label.values)):
    df.loc[val,"kfold"] = fold
    
df.to_csv("../input/mnist_train_folds.csv",index=True)