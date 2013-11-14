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


sigmoid = lambda x: 1/(1+exp(-x))
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
    m =  VW_Classifier(loss='logistic', moniker='example_sklearn', passes=10, silent=True, learning_rate=10, raw=True, oaa = 10)
    m.fit(X_train, y_train)
    # print confusion matrix on test data
    y_est = m.predict_proba(X_test)
    lines = y_est
    #print y_est
    probs = []
    for i, line in enumerate(lines):
      line = line.split()
      labels, vs = zip(*[[float(x) for x in l.split(':')] for l in line[:]])
      probs__ = sigmoid(asarray(vs))
      probs_ = probs__/probs__.sum()
      probs.append(probs_)

    probs = np.asarray(probs)
    print probs

if __name__ == '__main__':
    main()
