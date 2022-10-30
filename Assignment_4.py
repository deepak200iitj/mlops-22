## Ignore warnings

import warnings
warnings.filterwarnings("ignore")

## Importing the necessary Libraries


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, svm, metrics



## Defining SVM's parameters

gamma_list = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
c_list = [0.1, 0.2, 0.3, 0.5, 1, 10,100]

# Defining Decision Tree's classifier's hyperparameter
max_depth = [1,3,5,8,10]
min_samples_leaf = [1,3,5,10,20]
criterion =  ["gini", "entropy"]

# 5 different splits of train/test/validation set

train_frac = [0.8, 0.75, 0.7, 0.65, 0.6]
val_frac =   [0.1, 0.125, 0.15, 0.175, 0.2]
test_frac =  [0.1, 0.125, 0.15, 0.175, 0.2]


## Part 3: Loading the dataset
digits = datasets.load_digits()
print(f"\nImage shape: {digits.images.shape}")
print("\n Confusion Matrix of different splits and classifiers")

## Part 4: Data Pre-processing

acc_df = pd.DataFrame()
acc_svc = list()
acc_dt = list()

best_acc_svc = -1.0
best_model_svc = None
best_h_params_svc = dict()
best_acc_dt = -1.0
best_model_dt = None
best_h_params_dt = dict()

## Train, Test, Val set split

for i in range(5):
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X_train, X_val, y_train, y_val = train_test_split(data, digits.target, test_size = val_frac[i], shuffle = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = test_frac[i], shuffle = True)
    
    # For Support Vector Machines (SVM)
    
    for g in gamma_list:
        for c in c_list:            
            
            ## Defining the model
            
            clf = svm.SVC(gamma = g, C = c, random_state = 42)

            ## Training
            
            clf.fit(X_train, y_train)

            ## prediction
            
            predicted = clf.predict(X_val)
            cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)    
                
            ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
                
            if (cur_acc > best_acc_svc):
                best_acc_svc = cur_acc
                best_model_svc = clf
                best_h_params_svc[i] = {"gamma":g, "C": c}

    
    # For Decision Tree Clssifier    
    
    for d in max_depth:
        for l in min_samples_leaf:
            for c in criterion:
            
                ## Defining the model
                
                clf = DecisionTreeClassifier(max_depth = d, min_samples_leaf = l, criterion = c, random_state = 42)

                ## Training the model
                
                clf.fit(X_train, y_train)

                ## Prediction
                
                predicted = clf.predict(X_val)
                cur_acc = metrics.accuracy_score(y_pred = predicted, y_true = y_val)
                
                ## Identify the combination-of-hyper-parameter for which validation set accuracy is the highest
                
                if (cur_acc > best_acc_dt):
                    best_acc_dt = cur_acc
                    best_model_dt = clf
                    best_h_params_dt[i] = {"max_depth":d, "min_samples_leaf": l, "criterion": c}

    
    ## Get the test set prediction on best params i.e. using best model (SVM model) and calculate the accuracy of test set using best model

    pred_svc = best_model_svc.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_svc, y_true = y_test)
    acc_svc.append(acc)
    
    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_svc)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"\nConfusion matri SVM:\n\n{disp.confusion_matrix}\n")
    plt.savefig(f"run_{i}_SVM.png")

    
    ## Get the test set prediction on best params i.e. using best model (Decision Tree model) and calculate the accuracy of test set using best model
    
    pred_dt = best_model_dt.predict(X_test)
    acc = metrics.accuracy_score(y_pred = pred_dt, y_true = y_test)
    acc_dt.append(acc)

    ## Confusion Matrix

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, pred_dt)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"\nConfusion matrix DT:\n\n{disp.confusion_matrix}\n")
    plt.savefig(f"run_{i}_DT.png")
    

## calculate the mean and standard deviations of both the classifier's performances

acc_svc.append(np.mean(acc_svc))
acc_dt.append(np.mean(acc_dt))
acc_svc.append(np.std(acc_svc))
acc_dt.append(np.std(acc_dt))
acc_df['run'] = [1,2,3,4,5,'mean','std']
acc_df["Acc_SVC"] = acc_svc
acc_df["Acc_DT"] = acc_dt

##
print("\nBest hyper-params for different splits:\n")
print(best_h_params_svc, best_h_params_dt, sep='\n')
print("\n Accuracies of different split:\n")
print(acc_df)
print("\n")

## Write the performance metrics in the readme.md

acc_df.to_markdown("readme.md",index=False)