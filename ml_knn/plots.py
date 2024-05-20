from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from tqdm import tqdm

from ml_knn.knn import KNNClassifier
from ml_knn.lowess import LOWESS, sample
from ml_knn.params_optimize import best_my_fix, kernels


def neighbors_dependency(train, test, kernel):
    expected = test[test.columns[-1]]
    a = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    for i in tqdm(range(1, 15), 'Dependency neighbors'):
        knn_my_norm = KNNClassifier(k=i, default_class=0, kernel=kernel)
        knn_my_lowess = KNNClassifier(k=i, default_class=0, kernel=kernel)
        knn_lib_norm = KNeighborsClassifier(n_neighbors=i)
        knn_lib_lowess = KNeighborsClassifier(n_neighbors=i)

        weights = LOWESS(knn=KNNClassifier(k=i, default_class=0, kernel=kernels[best_my_fix['kernel']])).get_weights(train)
        # print('\n', weights.items())

        lowess_train = sample(
            train,
            weights,
            0.2)

        knn_my_norm.fit(train)
        knn_my_lowess.fit(train, weights=weights)
        knn_lib_norm.fit(train[train.columns[:-1]], train[train.columns[-1]])
        knn_lib_lowess.fit(lowess_train[lowess_train.columns[:-1]], lowess_train[lowess_train.columns[-1]])
        gained1 = knn_my_norm.predict(test[test.columns[:-1]])
        gained2 = knn_my_lowess.predict(test[test.columns[:-1]])
        gained3 = knn_lib_norm.predict(test[test.columns[:-1]])
        gained4 = knn_lib_lowess.predict(test[test.columns[:-1]])
        a.append(i)
        b1.append(f1_score(expected, gained1))
        b2.append(f1_score(expected, gained2))
        b3.append(f1_score(expected, gained3))
        b4.append(f1_score(expected, gained4))
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(a, b1, label='My normal')
    ax[0].plot(a, b2, label='My LOWESS')
    ax[1].plot(a, b3, label='Lib normal')
    ax[1].plot(a, b4, label='Lib LOWESS')
    ax[0].legend()
    ax[1].legend()
    fig.savefig(f'results/neighbors_{kernel.__name__}.jpg')

def radius_dependency(train, test, kernel):
    expected = test[test.columns[-1]]
    a = []
    b = []
    b1 = []
    b2 = []
    b3 = []
    b4 = []
    for r in tqdm(range(1, 100), 'Dependency range'):
        knn_my_norm = KNNClassifier(radius=r / 10, default_class=0, kernel=kernel)
        knn_my_lowess = KNNClassifier(radius=r / 10, default_class=0, kernel=kernel)
        knn_lib_norm = RadiusNeighborsClassifier(radius=r / 10, outlier_label=0)
        knn_lib_lowess = RadiusNeighborsClassifier(radius=r / 10, outlier_label=0)

        weights = (LOWESS(knn=KNNClassifier(radius=r / 10, default_class=0, kernel=kernels[best_my_fix['kernel']]))
                   .get_weights(train))
        # print('\n', weights.items())

        lowess_train = sample(
            train,
            weights,
            0.2)

        if len(lowess_train) < 10:
            continue

        knn_my_norm.fit(train)
        knn_my_lowess.fit(train, weights=weights)
        knn_lib_norm.fit(train[train.columns[:-1]], train[train.columns[-1]])
        knn_lib_lowess.fit(lowess_train[lowess_train.columns[:-1]], lowess_train[lowess_train.columns[-1]])
        gained1 = knn_my_norm.predict(test[test.columns[:-1]])
        gained2 = knn_my_lowess.predict(test[test.columns[:-1]])
        gained3 = knn_lib_norm.predict(test[test.columns[:-1]])
        gained4 = knn_lib_lowess.predict(test[test.columns[:-1]])
        a.append(r / 10)
        b1.append(f1_score(expected, gained1))
        b2.append(f1_score(expected, gained2))
        b3.append(f1_score(expected, gained3))
        b4.append(f1_score(expected, gained4))
    fig, ax = plt.subplots(nrows=2)
    ax[0].plot(a, b1, label='My normal')
    ax[0].plot(a, b2, label='My LOWESS')
    ax[1].plot(a, b3, label='Lib normal')
    ax[1].plot(a, b4, label='Lib LOWESS')
    ax[0].legend()
    ax[1].legend()
    fig.savefig(f'results/radius_{kernel.__name__}.jpg')