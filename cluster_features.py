"""Custom sklearn transformer for adding k-means cluster labels as features."""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans


class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    """Add a k-means cluster label as an extra numeric feature.

    This transformer fits a KMeans model on the input features and then
    appends the corresponding cluster label as an additional column. 
    It is designed to be used inside a scikit-learn Pipeline.

    The typical usage pattern is:

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("cluster", ClusterFeatureAdder()),
            ("model", RandomForestRegressor(...)),
        ])

    Args:
        n_clusters: Number of clusters to fit in KMeans.
        random_state: Seed for the internal KMeans model to ensure
            reproducible clustering.

    Attributes:
        kmeans_: Fitted KMeans instance after calling `fit`.
    """

    def __init__(self, n_clusters: int = 5, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_: Optional[KMeans] = None

    def fit(self, X, y=None):
        """Fit the internal KMeans model to the data.

        Args:
            X: Array-like of shape (n_samples, n_features).
            y: Ignored. Present for API consistency by convention.

        Returns:
            ClusterFeatureAdder: The fitted instance (self).
        """
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.kmeans_.fit(X)
        return self

    def transform(self, X):
        """Append k-means cluster labels as an extra feature column.

        Args:
            X: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray: Array of shape (n_samples, n_features + 1) where the
            last column contains the cluster labels as floats.
        """
        if self.kmeans_ is None:
            raise RuntimeError("ClusterFeatureAdder must be fitted before calling transform().")

        clusters = self.kmeans_.predict(X)
        clusters = clusters.reshape(-1, 1).astype("float32")
        return np.hstack([X, clusters])
