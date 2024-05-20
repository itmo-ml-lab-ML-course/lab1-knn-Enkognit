from collections import defaultdict

import pandas as pd
from sklearn.metrics import f1_score

from ml_knn.knn import KNNClassifier
from ml_knn.metrics import epanechnikov_kernel, triangular_kernel


class LOWESS:

    def __init__(self, it=3, knn=KNNClassifier(), kernel_func=epanechnikov_kernel):
        self.it = it
        self.knn = knn
        self.kernel_func = kernel_func

    def get_weights(self, df: pd.DataFrame):
        weights = defaultdict(lambda: 1)
        for _ in range(self.it):
            weights = self.get_weights_iter(df, weights)
        return weights

    def get_weights_iter(self, df: pd.DataFrame, weights):
        res = defaultdict(lambda: 1)
        for i, row in df.iterrows():
            self.knn.fit(df.drop(i), weights)
            probs = self.knn.predict(pd.DataFrame.from_records([row[:-1]]), get_probs=True)
            res[i] = self.kernel_func(1 - probs[0][row[df.columns[-1]]])
        return res

def sample(df: pd.DataFrame, weights, p):
    res = []
    for i, row in df.iterrows():
        if weights[i] >= p:
            res.append(row)
    return pd.DataFrame(res)

def lowess_neighbors_result(k, train, test):
    weights = LOWESS(knn=KNNClassifier(k=k)).get_weights(train)
    knn_norm = KNNClassifier(k=k)
    knn_lowess = KNNClassifier(k=k)
    knn_norm.fit(train)
    print(weights.items())
    knn_lowess.fit(sample(train, weights, 0.2))
    expected = test[test.columns[-1]]
    print('Neighbors results:'
          f'    Uniform: {f1_score(expected, knn_norm.predict(test[test.columns[:-1]]))}'
          f'    LOWESS: {f1_score(expected, knn_lowess.predict(test[test.columns[:-1]]))}')
