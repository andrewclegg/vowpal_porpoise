"""
This example showcases how to uses scikit-learn's cross-validation tools with
VW_Classifier. We load the MNIST digits dataset, convert it to VW_Classifier's
expected input format, then classify the digit as < 5 or >= 5.
"""
from numpy import *
import numpy as np
from numpy import array
import operator
from sklearn.base import TransformerMixin
from sklearn.datasets import load_digits
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from vowpal_porpoise.sklearn import VW_Classifier


class Array2Dict(TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples, n_features = np.shape(X)
        result = []
        for i in range(n_samples):
            result.append({
                          str(j): X[i, j]
                          for j in range(n_features)
                          if X[i, j] != 0
                          })
        return result


def main():
    # load iris data in, make a binary decision problem out of it
    data = load_digits()

    X = Array2Dict().fit_transform(data.data)
    y = data.target  + 1

    i = int(0.8 * len(X))
    X_train, X_test = X[:i], X[i:]
    y_train, y_test = y[:i], y[i:]

    # do the actual learning
    m =  VW_Classifier(loss='logistic', moniker='example_sklearn', passes=10, silent=True, learning_rate=10, raw = True, oaa = 10)
    m.fit(X_train, y_train)

    # print confusion matrix on test data
    y_pred = m.predict(X_test)
    print("F1 score on test is", f1_score(y_pred, y_test))

    # The raw probabilities are as follows
    y_prob = m.predict_proba(X_test)
    print("The raw probabilities for each test example are", y_prob)
    

if __name__ == '__main__':
    main()
