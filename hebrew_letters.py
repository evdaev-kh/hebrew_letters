""""
@author Khai Evdaev
"""

import numpy as np
import pandas as pd
from PIL import Image
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 8]
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics


def Logistic_Classification(x_train, y_train, x_test, y_test, visualize=False):
    """"
        Brief: Computes the Logistic regression for classification

        Params:
            x_train     - training dataset x vals (i.e. pixels of images)
            y_train     - label for each input value
            x_test      - test dataset for the images
            y_test      - test labels for the images
    """
    logistic_model = LogisticRegression(solver='saga', penalty='elasticnet')
    logistic_model.fit(np.array(x_train), y_train)

    y_pred = logistic_model.predict(x_test)

    print("Logistic Regression Accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

    if visualize:
        # Create the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()

    return logistic_model, y_pred

def SVM_Classification(x_train, y_train, x_test, y_test, visualize=False):
    """"
        Brief: Computes the SVM classification

        Params:
            x_train     - training dataset x vals (i.e. pixels of images)
            y_train     - label for each input value
            x_test      - test dataset for the images
            y_test      - test labels for the images
            visualize   - whether to display the confusion matrix, set to False by default
    """
    svm_model = SVC(kernel='rbf', gamma='scale', C=1)
    svm_model.fit(np.array(x_train), y_train)

    y_pred = svm_model.predict(x_test)

    print("SVM Accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

    if visualize:
        # Create the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()

    return svm_model, y_pred

def RandomForest_Classification(x_train, y_train, x_test, y_test, visualize=False):
    """"
        Brief: Computes the Random Forest classification

        Params:
            x_train     - training dataset x vals (i.e. pixels of images)
            y_train     - label for each input value
            x_test      - test dataset for the images
            y_test      - test labels for the images
            visualize   - whether to display the confusion matrix, set to False by default
    """
    rf_model = RandomForestClassifier(n_estimators=376, max_depth=11)
    rf_model.fit(np.array(x_train), y_train)

    y_pred = rf_model.predict(x_test)

    print("SVM Accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

    if visualize:
        # Create the confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)

        metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
        plt.show()

    return rf_model, y_pred

if __name__ == "__main__":
    #Load training and testing datasets
    train_pixels = pd.read_csv('./training.csv')
    X_train = train_pixels.drop('Label', axis=1)
    Y_train = train_pixels['Label']
    Y_train = np.array(Y_train)

    test_frame = pd.read_csv('testing.csv')
    X_test = test_frame.drop('Label', axis=1)
    Y_test = test_frame['Label']

    RandomForest_Classification(X_train, Y_train, X_test, Y_test, visualize=True)

