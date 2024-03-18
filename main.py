import pandas as pd
import numpy as np
from sklearn import preprocessing
import arff
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize, minmax_scale
from math import sqrt
from random import seed
from random import randrange
from sklearn.preprocessing import normalize

# Read the data
data = pd.DataFrame(arff.load('C:/NSL-KDD/KDDTest+.arff'))
with open('C:/NSL-KDD/kddcup.names', 'r') as infile:
    kdd_names = infile.readlines()
kdd_cols = [x.split(':')[0] for x in kdd_names[1:]]
# The dataset include the class attack
kdd_cols += ['class']
# change the columns' names
data.columns = kdd_cols
le = preprocessing.LabelEncoder()
# extracting numerical labels from categorical data
data['protocol_type'] = le.fit_transform(data['protocol_type'])
data['service'] = le.fit_transform(data['service'])
data['flag'] = le.fit_transform(data['flag'])
data['class'] = le.fit_transform(data['class'])
data['land'] = le.fit_transform(data['land'])
data['logged_in'] = le.fit_transform(data['logged_in'])
data['is_host_login'] = le.fit_transform(data['is_host_login'])
data['is_guest_login'] = le.fit_transform(data['is_guest_login'])

# data.drop_duplicates(inplace=True)
# data.reset_index(drop=True, inplace=True)
# print(data.info)
dataset = data.values
#print(dataset

################################

####### functions used in this work ################
# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    row1 = row1[:-1]
    row2 = row2[:-1]
    distance = np.linalg.norm(row1 - row2)
    return distance

def weighted_distance(row1, row2, weights):
    row1 = row1[:-1]
    row2 = row2[:-1]
    distance = np.sum(np.multiply(weights, np.square(row1 - row2)))
    return distance

# Locate the most similar neighbors
def get_neighbors2(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

def get_neighbors(train, test_row, num_neighbors, weights):
    distances = list()
    for train_row in train:
        dist = weighted_distance(test_row, train_row, weights)
        distances.append((train_row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors

# Make a prediction with neighbors
def predict_classification(train, test_row, num_neighbors, weights):
    neighbors = get_neighbors(train, test_row, num_neighbors, weights)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    ###### define here the new weights based on difference
    for row in neighbors:
        row_test = test_row[:-1]
        label_test = test_row[-1]
        row_neighbor = row[:-1]
        label_neighbor = row[-1]
        difference = np.abs(row_test - row_neighbor)
        #print(np.max(difference))
        #print(np.min(difference))
        th = 0.5
        delta = 0.1
        boolArr = (difference < th)

        if label_test == prediction:
            weights[boolArr] = weights[boolArr] * (3)
            weights[~boolArr] = weights[~boolArr] * (4/3)
            #weights[boolArr] = weights[boolArr] + 2*delta
            #weights[~boolArr] = weights[~boolArr] + delta
            #weights = np.multiply(weights, 1/(difference+1e-10))
            #weights = weights
        else:
            #weights = weights
            weights[boolArr] = weights[boolArr]*(1/5)
            weights[~boolArr] = weights[~boolArr]*(4/5)
            #weights[boolArr] = weights[boolArr] - 2*delta
            #weights[~boolArr] = weights[~boolArr] - delta
            #weights = np.multiply(weights, difference)
            #weights = weights
            #difference = difference + 1e-10
            #difference = difference / np.sum(difference)
            #weights = np.multiply(weights, 1/(difference + 1e-10))

        weights = weights / np.sum(weights)
    return prediction, weights

def predict_classification2(train, test_row, num_neighbors):
    neighbors = get_neighbors2(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


# wkNN Algorithm
def w_k_nearest_neighbors(train, test, num_neighbors):
    weights = (1 / (len(test[0]) - 1)) * np.ones(len(test[0]) - 1)
    predictions = list()
    for row in test:
        output, weights = predict_classification(train, row, num_neighbors, weights)
        predictions.append(output)
    return predictions, weights
    
# kNN Algorithm
def k_nearest_neighbors(train, test, num_neighbors):
    predictions = list()
    for row in test:
        output = predict_classification2(train, row, num_neighbors)
        predictions.append(output)
    return predictions

########### Functions related to Dataset ############
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm1, algorithm2, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    print(np.shape(folds))
    scores1 = list()
    scores2 = list()
    whole_label = list()
    whole_predicted1 = list()
    whole_predicted2 = list()
    for i in range(len(folds)):
        test_set = list(folds[i])
        train_set = list(folds)
        del train_set[i]
        train_set = sum(train_set, [])
        print(np.shape(train_set))
        print(np.shape(test_set))
        predicted1, weights1 = algorithm1(train_set, test_set, *args)
        predicted2 = algorithm2(train_set, test_set, *args)
        actual = [row[-1] for row in test_set]
        accuracy1 = accuracy_metric(actual, predicted1)
        accuracy2 = accuracy_metric(actual, predicted2)
        scores1.append(accuracy1)
        scores2.append(accuracy2)
        for j in range(len(actual)):
            whole_label.append(actual[j])
            whole_predicted1.append(predicted1[j])
            whole_predicted2.append(predicted2[j])
    return scores1, weights1, scores2, whole_label, whole_predicted1, whole_predicted2


dataset1 = dataset[0:2000, :]
#dataset1 = dataset
#Normalize the dataset between 0 and 1
dataset1[:, 0:42] = minmax_scale(dataset1[:, 0:42], feature_range=(0, 1))

n_folds = 10
num_neighbors = 5
scores1, weights, scores2, whole_label, whole_predicted1, whole_predicted2 = evaluate_algorithm(dataset1, w_k_nearest_neighbors, k_nearest_neighbors, n_folds, num_neighbors)

print('Feature Weighted KNN')
print('Scores: %s' % scores1)
print('Mean Accuracy1: %.3f%%' % (sum(scores1)/float(len(scores1))))
print(classification_report(whole_label, whole_predicted1))
r1 = roc_auc_score(whole_label,  whole_predicted1)
print('ROC Area: %s' % r1)
print('weights of the features')
print(weights)

print('--------------------------------------------------------------')
print('KNN')
print('Scores: %s' % scores2)
print('Mean Accuracy2: %.3f%%' % (sum(scores2)/float(len(scores2))))
print(classification_report(whole_label, whole_predicted2))
r2 = roc_auc_score(whole_label,  whole_predicted2)
print('ROC Area: %s' % r2)

####################################
fpr1, tpr1, threshold1 = roc_curve(whole_label, whole_predicted1)
fpr2, tpr2, threshold2 = roc_curve(whole_label, whole_predicted2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr1, tpr1, label='Weighted KNN')
plt.plot(fpr2, tpr2, label='KNN')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()