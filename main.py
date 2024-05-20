import math
import time
from collections import defaultdict

import numpy
import optuna
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KDTree, KNeighborsClassifier, RadiusNeighborsClassifier
from typing import Callable, Optional
import pandas as pd
from scipy.spatial.distance import minkowski, cosine

from ml_knn.knn import KNNClassifier
from ml_knn.lowess import lowess_neighbors_result
from ml_knn.params_optimize import find_my_best_params_fix, find_lib_best_params_fix, find_my_best_params_unfix, \
    find_lib_best_params_unfix, best_my_fix, kernels
from ml_knn.plots import neighbors_dependency, radius_dependency

optuna.logging.set_verbosity(optuna.logging.ERROR)

# Parsen-Rosenblatt

def get_dataset(name: str) -> pd.DataFrame:
    dataset = pd.read_csv(name)
    for col in dataset.columns:
        if dataset[col].dtype == object:
            arr = sorted(list(set(dataset[col].to_numpy())))
            print(arr)
            dataset[col] = dataset[col].map(lambda x: arr.index(x))
    dataset = (dataset - dataset.min()) / (dataset.max() - dataset.min())
    return dataset

def test_classifier(train, test):
    knn = KNNClassifier(k=5)
    knn.fit(train)

    expected = test[test.columns[-1]].to_numpy()
    gained = knn.predict(test[test.columns[:-1]])
    print(expected)
    print(gained)
    print(sum([1 if e != g else 0 for e, g in zip(expected, gained)]) / len(gained) * 100, '% loss', sep='')
    print('F-score:', f1_score(expected, gained))


# def error_k_dep():
#     KNNClassifier()

dt_name = 'datasets/KNNAlgorithmDataset.csv'
def load_dataset():
    df = get_dataset(dt_name)

    train_, test_ = train_test_split(df, test_size=0.20)

    train: pd.DataFrame = train_
    test: pd.DataFrame = test_

    train.to_csv(dt_name[:-4] + '_train.csv')
    test.to_csv(dt_name[:-4] + '_test.csv')

    return df, train, test

def main():
    # df, train, test = load_dataset()

    train = pd.read_csv(dt_name[:-4] + '_train.csv', index_col=0)
    test = pd.read_csv(dt_name[:-4] + '_test.csv', index_col=0)
    print(len(train), len(test))

    test_classifier(train, test)

    find_my_best_params_fix(train)
    find_lib_best_params_fix(train)
    find_my_best_params_unfix(train)
    find_lib_best_params_unfix(train)

    for kernel in kernels:
        neighbors_dependency(train, test, kernel)
        radius_dependency(train, test, kernel)

    lowess_neighbors_result(best_my_fix['k'], train, test)


if __name__ == "__main__":
    main()