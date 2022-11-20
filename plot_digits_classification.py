# Import datasets, classifiers and performance metrics
# from joblib import dump, load
# import pdb
from sklearn import datasets, svm, metrics
from sklearn import tree
import numpy as np
import argparse
from joblib import dump, load

from utils2 import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    h_param_tuning_dec,
    data_viz,
    get_all_h_param_comb,
    predict
)

parser = argparse.ArgumentParser()

parser.add_argument('--clf_name')           # positional argument
parser.add_argument('--random_state')      # option that takes a value
parser.add_argument('-v', '--verbose',
                    action='store_true')  # on/off flag

args = parser.parse_args()
# print(args.filename, args.count, args.verbose)

clf_name = args.clf_name
random_seed_value = args.random_state
print(f"Classifier name: {clf_name}, Random seed: {random_seed_value}")

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {"gamma": gamma_list, "C": c_list}

h_param_comb = get_all_h_param_comb(params)

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
del digits

if clf_name == 'svm':
    models = [[svm.SVC(), 'svm']]
elif clf_name == 'tree':
    models = [[tree.DecisionTreeClassifier(), 'decision_tree']]
else:
    print("Give valid argument")
    exit()
# models = [[tree.DecisionTreeClassifier(), 'decision_tree'], [svm.SVC(), 'svm']]
perf_test = {}
metric = metrics.accuracy_score
metric2 = metrics.f1_score
for model in models:
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(data, label, train_frac, dev_frac, int(random_seed_value))
    clf = model[0]
    if model[1] == 'svm':
        best_model, best_metric, best_h_params = h_param_tuning(h_param_comb, clf, x_train, y_train, x_dev, y_dev,
                                                                metric)
    else:
        best_model, best_metric, best_h_params = h_param_tuning_dec(clf, x_train, y_train, x_dev, y_dev, metric)

    result = predict(best_model, x_test, y_test, metric)
    print("test accuracy: ", str(result))

    result2 = predict(best_model, x_test, y_test, metric2, 'f1_macro')
    print("test macro-f1: ", str(result2))


    file = open("results/" + str(clf_name) + "_" + str(random_seed_value) + ".txt", "w")
    file.write("test accuracy: " + str(result) + "\n")
    file.write("test macro-f1: " + str(result2))
    file.close()

    best_param_config = "_".join([h + "=" + str(best_h_params[h]) for h in best_h_params])
    dump(best_model, "models/svm_" + best_param_config + ".joblib")

print("\n")