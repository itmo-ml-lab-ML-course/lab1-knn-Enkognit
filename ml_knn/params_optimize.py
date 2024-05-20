import numpy as np
import optuna
from optuna.samplers import GridSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from ml_knn.knn import KNNClassifier
from ml_knn.metrics import gauss_kernel, epanechnikov_kernel, uniform_kernel, triangular_kernel, kernel_factory

best_my_fix = {'k': 13, 'kernel': 3}
best_lib_fix = {'k': 1}
best_my_unfix = {'radius': 1.7242377750544042}
best_lib_unfix = {'radius': 0.061652883765028865}

kernels = [gauss_kernel, epanechnikov_kernel, uniform_kernel, triangular_kernel,
           kernel_factory(2, 2), kernel_factory(3, 3), kernel_factory(4, 4)]

def find_my_best_params_fix(df, valid_part = 0.21):
    train, valid = train_test_split(df, test_size=valid_part)
    x_valid, y_valid = valid[valid.columns[:-1]], valid[valid.columns[-1]]

    def tryf(trial):
        k = trial.suggest_int('k', 1, 50)
        kernel = trial.suggest_int(
            'kernel',
            0, len(kernels) - 1
        )
        knn = KNNClassifier(k=k, default_class=0, kernel=kernels[kernel])
        knn.fit(train)
        gained = knn.predict(x_valid)
        return f1_score(y_valid, gained)

    study = optuna.create_study(direction="maximize", sampler=GridSampler({
        'k': range(1, 50),
        'kernel': range(len(kernels))
    }))
    study.optimize(
        tryf,
        n_trials=50 * len(kernels),
        n_jobs=20,
        show_progress_bar=True
    )

    global best_my_fix
    best_my_fix = study.best_params
    print('Best of my fix:', best_my_fix)
    return study.best_params

def find_lib_best_params_fix(df, valid_part = 0.21):
    train, valid = train_test_split(df, test_size=valid_part)
    x_train, y_train = valid[valid.columns[:-1]].to_numpy(), valid[valid.columns[-1]].to_numpy()
    x_valid, y_valid = valid[valid.columns[:-1]].to_numpy(), valid[valid.columns[-1]].to_numpy()

    def tryf(trial):
        k = trial.suggest_int('k', 1, 50)
        knn = KNeighborsClassifier(n_neighbors=k, leaf_size=1000, algorithm='kd_tree')
        knn.fit(x_train, y_train)
        gained = knn.predict(x_valid)
        return f1_score(y_valid, gained)

    study = optuna.create_study(direction="maximize", sampler=GridSampler({
        'k': range(1, 50)
    }))
    study.optimize(
        tryf,
        n_trials=50,
        n_jobs=20,
        show_progress_bar=True
    )

    global best_lib_fix
    best_lib_fix = study.best_params
    print('Best of lib fix:', best_lib_fix)
    return study.best_params

def find_my_best_params_unfix(df, valid_part = 0.21):
    train, valid = train_test_split(df, test_size=valid_part)
    x_valid, y_valid = valid[valid.columns[:-1]], valid[valid.columns[-1]]

    def tryf(trial):
        radius = trial.suggest_float('radius', 0.05, 14)
        kernel = trial.suggest_int(
            'kernel',
            0, len(kernels) - 1
        )
        knn = KNNClassifier(radius=radius, default_class=0, kernel=kernels[kernel])
        knn.fit(train)
        gained = knn.predict(x_valid)
        return f1_score(y_valid, gained)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        tryf,
        n_trials=1000,
        n_jobs=20,
        show_progress_bar=True
    )

    global best_my_unfix
    best_my_unfix = study.best_params
    print('Best of my unfix:', best_my_unfix)
    return study.best_params


def find_lib_best_params_unfix(df, valid_part = 0.21):
    train, valid = train_test_split(df, test_size=valid_part)
    x_train, y_train = valid[valid.columns[:-1]], valid[valid.columns[-1]]
    x_valid, y_valid = valid[valid.columns[:-1]], valid[valid.columns[-1]]

    def tryf(trial):
        radius = trial.suggest_float('radius', 0.01, 15)
        knn = RadiusNeighborsClassifier(radius=radius, leaf_size=1000, outlier_label=0, algorithm='kd_tree')
        knn.fit(x_train, y_train)
        gained = knn.predict(x_valid)
        return f1_score(y_valid, gained)

    study = optuna.create_study(direction="maximize")
    study.optimize(
        tryf,
        n_trials=1000,
        n_jobs=20,
        show_progress_bar=True
    )

    global best_lib_unfix
    best_lib_unfix = study.best_params
    print('Best of my unfix:', best_lib_unfix)
    return study.best_params