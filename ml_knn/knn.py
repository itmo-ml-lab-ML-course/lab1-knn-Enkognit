from collections import defaultdict
from typing import Optional

import pandas as pd
from sklearn.neighbors import KDTree

from ml_knn import Classifier, one, zero
from ml_knn.metrics import gauss_kernel


class KNNClassifier:

    k: Optional[int] = 5
    weights: dict[int, float] = defaultdict(one)
    radius: Optional[float] = None
    # minkowski, euclidian, cosine
    dist_func: str = 'minkowski'
    leaf_size: int = 2
    # gauss_kernel

    def __init__(self,
                 dist_func='minkowski',
                 k=None,
                 radius=None,
                 leaf_size=1000,
                 default_class=None,
                 kernel=gauss_kernel):
        self.kernel_func = kernel
        self.k = k
        self.radius = radius
        self.leaf_size = leaf_size
        self.default_class = default_class
        self.dist_func = dist_func

    def fit(self, df: pd.DataFrame, weights=None):
        if weights:
            self.weights = weights
        else:
            self.weights = defaultdict(one)
        if self.k and len(df) <= self.k:
            self.k = len(df) - 1
        self.X = df[df.columns[:-1]].to_numpy()
        self.Y = df[df.columns[-1]].to_numpy()
        self.kdtree = KDTree(
            self.X,
            leaf_size=self.leaf_size,
            metric=self.dist_func
        )

    def predict(self, df: pd.DataFrame, get_probs=False):
        res = []
        probs = []
        X = df.to_numpy()
        if self.radius is None:
            dsts, inds = self.kdtree.query(X, k=self.k + 1)
        else:
            inds, dsts = self.kdtree.query_radius(X, r=self.radius, return_distance=True)
        for dst, ind in zip(dsts, inds):
            mp: dict[int, float] = defaultdict(zero)
            rd = self.radius if self.radius is not None else dst[-1]
            sm = 0
            for i in range(len(ind) - (1 if self.radius is None else 0)):
                o = self.weights[ind[i]] * self.kernel_func(dst[i] / rd)
                if o > 0:
                    mp[self.Y[ind[i]]] += o
                    sm += o


            for k, v in mp.items():
                mp[k] /= sm

            probs.append(mp)

            mx = -1e18
            cl = self.default_class
            for k, v in mp.items():
                if v > mx:
                    mx = v
                    cl = k

            res.append(cl)
        if get_probs:
            return probs
        else:
            return res
