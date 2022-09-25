# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


#PART: library dependencies -- sklear, torch, tensorflow, numpy, transformers

# Standard scientific Python imports
import matplotlib.pyplot as plt
from skimage import transform
import numpy as np
import pandas as pd
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# 1. set the ranges of hyper parameters 
gamma_list = [0.1, 0.01, 0.02, 0.005]
c_list = [0.1, 0.2, 0.5, 1,10, 100] 

h_param_comb = [{'gamma':g, 'C':c} for g in gamma_list for c in c_list]

#This is to test if all the possible combination is present in the list.
assert len(h_param_comb) == len(gamma_list)*len(c_list)


train_frac = 0.8
test_frac = 0.1
dev_frac = 0.1

#PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()

#PART: sanity check visualization of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)


#PART: data pre-processing -- to remove some noise, to normalize data, format the data to be consumed by mode
# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#Qn2  Print the image size
print(f"the image shape of digit dataset : {digits.images.shape}")
print(f"the image shape of 1st image(other images will be similar) of digit  dataset : {digits.images[0].shape}")
print("\n")
print("\n")
data_list=[]
features=digits.images
data_list.append(data)
resized_16_16 = np.array(list(map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (16, 16), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             features)))
data_list.append(resized_16_16) 
resized_4_4 = np.array(list(map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (4, 4), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             features)))
data_list.append(resized_4_4) 
resized_2_2 = np.array(list(map
                         (lambda img: transform.resize(
                                        img.reshape(8,8),#old shape
                                          (2, 2), #new shape
                                          mode='constant',
                                         #flatten the resized image
                                          preserve_range=True).ravel(),
             features)))

data_list.append(resized_2_2)  #List contains three different sized datasets

print(f"Processing for four shapes (8*8(ie.original), 4*4, 16*16, 2*2)) : {data.shape}, {resized_4_4.shape},{resized_16_16.shape} and {resized_2_2.shape}")

#Lets iterate over all these three to get the remaining part of question 2
#WRITE THE LOOP HERE

#Qn2 SecondPart Resize images to 3 different resolutions of your choice.
#Lets say we resize to the image to half, quarter and double
#Different resolutions selected--> (image.shape[0]*0.5, image.shape[1]*0.5), (image.shape[0]*.25, image.shape[1]*.25) and (image.shape[0]*2, image.shape[1]* 2)


#PART: define train/dev/test splits of experiment protocol
# train to train model
# dev to set hyperparameters of the model
# test to evaluate the performance of the model
for data in data_list:
    #Create a new Dataframe everytime 
    df = pd.DataFrame(columns=['hyperparameter','train_acc','dev_acc','test_acc'])
    print(f"Starting Processing for data shape : {data.shape}")
    dev_test_frac = 1-train_frac
    X_train, X_dev_test, y_train, y_dev_test = train_test_split(
        data, digits.target, test_size=dev_test_frac, shuffle=True
    )
    X_test, X_dev, y_test, y_dev = train_test_split(
        X_dev_test, y_dev_test, test_size=(dev_frac)/dev_test_frac, shuffle=True, random_state=42
    )


    best_acc = -1.0
    best_model = None
    best_h_params = None
    #print(f"hyperparameters:\t\ttrain_accuracy:\tdev_accuracy:\ttest_accuracy")
    # 2. For every combination-of-hyper-parameter values
    for cur_h_params in h_param_comb:

        #PART: Define the model
        # Create a classifier: a support vector classifier
        clf = svm.SVC()

        #PART: setting up hyperparameter
        hyper_params = cur_h_params
        clf.set_params(**hyper_params)


        #PART: Train model
        # 2.a train the model 
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)
        train_accuracy=clf.score(X_train, y_train)
        train_accuracy=round(train_accuracy,2)
        # print(cur_h_params)
        #PART: get dev set predictions
        predicted_dev = clf.predict(X_dev)

        # 2.b compute the accuracy on the validation set
        cur_acc = metrics.accuracy_score(y_pred=predicted_dev, y_true=y_dev)
        cur_acc=round(cur_acc,2)

        predicted_test=clf.predict(X_test)
        test_accuracy= metrics.accuracy_score(y_pred=predicted_test, y_true=y_test)
        test_accuracy=round(test_accuracy,2)

        #3. identify the combination-of-hyper-parameter for which validation set accuracy is the highest. 
        if cur_acc > best_acc:
            best_acc = cur_acc
            best_model = clf
            best_h_params = cur_h_params
            #print("Found new best acc with :"+str(cur_h_params))
            #print("New best val accuracy:" + str(cur_acc))
    #Assignment 3 Qn 1.1
    #1) the table of results: columns = {train, dev, test accuracy}, rows = {each hyperparameter combination}:
        df_row_list=[cur_h_params,train_accuracy,cur_acc,test_accuracy]
       # print(f"{cur_h_params}\t\t:{train_accuracy}\t:{cur_acc}\t:{test_accuracy}")
        df.loc[len(df.index)] = df_row_list
    df = df.set_index('hyperparameter')

   

    print(df)

    #Calculating min  accuracies across hyperparameters
    min_train_acc=min(df['train_acc'].tolist())
    min_dev_acc=min(df['dev_acc'].tolist())
    min_test_acc=min(df['test_acc'].tolist())

    print(f"min_train_acc : {min_train_acc}")
    print(f"min_dev_acc : {min_dev_acc}")
    print(f"min_test_acc: {min_test_acc}")
   


    #Calculating max accuracies across hyperparameters
    max_train_acc=max(df['train_acc'].tolist())
    max_dev_acc=max(df['dev_acc'].tolist())
    max_test_acc=max(df['test_acc'].tolist())

    print(f"max_train_acc : {max_train_acc}")
    print(f"max_dev_acc : {max_dev_acc}")
    print(f"max_test_acc : {max_test_acc}")

    #Calculating mean accuracies across hyperparameters
    mean_train_acc=df['train_acc'].mean()
    mean_dev_acc=df['dev_acc'].mean()
    mean_test_acc=df['test_acc'].tolist()

    print(f"mean_train_acc : {mean_train_acc}")
    print(f"mean_dev_acc : {mean_dev_acc}")
    print(f"mean_test_acc : {mean_test_acc}")

 #Calculating median accuracies across hyperparameters
    median_train_acc=df['train_acc'].median()
    median_dev_acc=df['dev_acc'].median()
    median_test_acc=df['test_acc'].tolist()

    print(f"median_train_acc : {median_train_acc}")
    print(f"median_dev_acc : {median_dev_acc}")
    print(f"median_test_acc : {median_test_acc}")



    #Assignment 3 Qn 1.2
    #2) best hyperparams, and train, dev, test accuracy with them. Upload the screenshot to google form.


    #PART: Get test set predictions
    # Predict the value of the digit on the test subset
    predicted = best_model.predict(X_test)

#     #PART: Sanity check of predictions
#     _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
#     for ax, image, prediction in zip(axes, X_test, predicted):
#         ax.set_axis_off()
#         image = image.reshape(8, 8)
#         ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#         ax.set_title(f"Prediction: {prediction}")

    # 4. report the test set accurancy with that best model.
    #PART: Compute evaluation metrics
#     print(
#         f"Classification report for SVM {clf}:\n"
#         f"{metrics.classification_report(y_test, predicted)}\n"
#     )
   


    best_train_accuracy=best_model.score(X_train, y_train)
    best_train_accuracy=round(best_train_accuracy,2)
    best_predicted_dev = best_model.predict(X_dev)

    # 2.b compute the accuracy on the validation set
    best_cur_acc = metrics.accuracy_score(y_pred=best_predicted_dev, y_true=y_dev)
    best_cur_acc=round(best_cur_acc,2)

    best_predicted_test=best_model.predict(X_test)
    best_test_accuracy= metrics.accuracy_score(y_pred=best_predicted_test, y_true=y_test)
    best_test_accuracy=round(best_test_accuracy,2)


    print(f"\nbest_hyperparameters:\t\tbest_train_acc:\t\tbest_dev_acc:\t\tbest_test_acc")
    print(f"{best_h_params}\t\t:{best_train_accuracy}\t\t:{best_cur_acc}\t\t:{best_test_accuracy}\n")
    print("*"*100)
    print("*"*100)
    print("\n\n")
    



