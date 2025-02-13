# Starter code prepared by Borna Ghotbi, Polina Zablotskaia, and Ariel Shann for Computer Vision
# based on a MATLAB code by James Hays and Sam Birch

import numpy as np
from util import load, build_vocabulary, get_bags_of_sifts
from classifiers import nearest_neighbor_classify, svm_classify
import sklearn
import matplotlib.pyplot as plt

# For this assignment, you will need to report performance for sift features on two different classifiers:
# 1) Bag of sift features and nearest neighbor classifier
# 2) Bag of sift features and linear SVM classifier

# For simplicity you can define a "num_train_per_cat" variable, limiting the number of
# examples per category. num_train_per_cat = 100 for instance.

# Sample images from the training/testing dataset.
# You can limit number of samples by using the n_sample parameter.

print('Getting paths and labels for all train and test data\n')
train_image_paths, train_labels = load("sift/train")
test_image_paths, test_labels = load("sift/test")

''' Step 1: Represent each image with the appropriate feature
 Each function to construct features should return an N x d matrix, where
 N is the number of paths passed to the function and d is the 
 dimensionality of each image representation. See the starter code for
 each function for more details. '''

print('Extracting SIFT features\n')
# TODO: You code build_vocabulary function in util.py
kmeans = build_vocabulary(train_image_paths, vocab_size=200)

# TODO: You code get_bags_of_sifts function in util.py
train_image_feats = get_bags_of_sifts(train_image_paths, kmeans)
test_image_feats = get_bags_of_sifts(test_image_paths, kmeans)


# make a histogram of all the average values
numberOfImages = train_image_feats.shape[0]
vocab_size = train_image_feats.shape[1]
zeroArray = np.zeros(vocab_size)
listOfLabels = np.unique(train_labels)
label_dict = {}
k = 0
# create a dictionary with all the labels being the keys
# taken from https://stackoverflow.com/questions/5036700/how-can-you-dynamically-create-variables
while k < len(listOfLabels):
    key = listOfLabels[k]
    value = zeroArray
    label_dict[key] = value
    k += 1

# add each value to the dictionary
for i in range(numberOfImages):
    className = train_labels[i]
    vals = train_image_feats[i, :]
    label_dict[className] = label_dict[className] + vals

# take the average score of these values
for i in range(len(listOfLabels)):
    label_dict[listOfLabels[i]] = label_dict[listOfLabels[i]] / 100

# produce histograms for each label
for i in range(len(listOfLabels)):
    bins = np.arange(vocab_size)
    values = label_dict[listOfLabels[i]]
    plt.bar(bins, values, width=0.4)
    plt.title("Histogram for Label " + str(i))
    plt.savefig("Histogram " + str(i+1))
    plt.clf()


# If you want to avoid recomputing the features while debugging the
# classifiers, you can either 'save' and 'load' the extracted features
# to/from a file.

''' Step 2: Classify each test image by training and using the appropriate classifier
 Each function to classify test features will return an N x l cell array,
 where N is the number of test cases and each entry is a string indicating
 the predicted one-hot vector for each test image. See the starter code for each function
 for more details. '''

print('Using nearest neighbor classifier to predict test set categories\n')
# TODO: YOU CODE nearest_neighbor_classify function from classifers.py
pred_labels_knn = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats)

print('Using support vector machine to predict test set categories\n')
# TODO: YOU CODE svm_classify function from classifers.py
pred_labels_svm = svm_classify(train_image_feats, train_labels, test_image_feats)

print('---Evaluation---\n')
# Step 3: Build a confusion matrix and score the recognition system for 
#         each of the classifiers.
# TODO: In this step you will be doing evaluation. 
# 1) Calculate the total accuracy of your model by counting number
#   of true positives and true negatives over all. 
# 2) Build a Confusion matrix and visualize it. 
#   You will need to convert the one-hot format labels back
#   to their category name format.

# calculate KNN accuracy:
print("Nearest Neighbours Accuracy:")
knnScore = sklearn.metrics.accuracy_score(test_labels, pred_labels_knn)
print(knnScore)

# calculate SVM accuracy:
print("SVM accuracy:")
svmScore = sklearn.metrics.accuracy_score(test_labels, pred_labels_svm)
print(svmScore)

# calculate KNN confusion matrix:
knnConfusionMatrix = sklearn.metrics.confusion_matrix(test_labels, pred_labels_knn)
knnDisp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=knnConfusionMatrix)
knnDisp.plot()
plt.show()

# calculate SVM confusion matrix:
svmConfusionMatrix = sklearn.metrics.confusion_matrix(test_labels, pred_labels_svm)
svmDisp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=svmConfusionMatrix)
svmDisp.plot()
plt.show()

# Interpreting your performance with 100 training examples per category:
#  accuracy  =   0 -> Your code is broken (probably not the classifier's
#                     fault! A classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .10 -> Your performance is chance. Something is broken or
#                     you ran the starter code unchanged.
#  accuracy ~= .40 -> Rough performance with bag of SIFT and nearest
#                     neighbor classifier. 
#  accuracy ~= .50 -> You've gotten things roughly correct with bag of
#                     SIFT and a linear SVM classifier.
#  accuracy >= .60 -> You've added in spatial information somehow or you've
#                     added additional, complementary image features. This
#                     represents state of the art in Lazebnik et al 2006.
#  accuracy >= .85 -> You've done extremely well. This is the state of the
#                     art in the 2010 SUN database paper from fusing many 
#                     features. Don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> You used modern deep features trained on much larger
#                     image databases.
#  accuracy >= .96 -> You can beat a human at this task. This isn't a
#                     realistic number. Some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.
