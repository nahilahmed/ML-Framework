# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:42:55 2020

@author: nahil
"""

import argparse
import config
import joblib
import os

import model_dispatcher

import pandas as pd
from sklearn import metrics
from sklearn import tree

def run(fold, model):
    
    df = pd.read_csv(config.TRAINING_FILE)
    
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    x_train = df_train.drop("label",axis=1).values
    y_train = df_train.label.values
    
    x_valid = df_valid.drop("label",axis=1).values
    
    clf = model_dispatcher.models[model]
    
    clf.fit(x_train, y_train)
    
    preds = clf.predict(x_valid)
    
    accuracy = metrics.accuracy_score(df_valid.label.values, preds)
    print("Fold: {}, Accuracy:{}%".format(fold,accuracy*100))
    
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT,f"dt_{fold}.bin")
    )
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--fold",type=int)
    parser.add_argument("--model", type=str)
    
    args = parser.parse_args()
    
    run(args.fold, args.model)
        