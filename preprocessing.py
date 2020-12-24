# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:41:57 2019

@author: Erynsul
"""
import numpy as np
from sklearn.utils import check_array, inplace_column_scale
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.base import TransformerMixin, BaseEstimator
from scipy import sparse


class PercentileTruncator(TransformerMixin, BaseEstimator):
    """
    Truncates values of the columns of an array to the left and right quantiles.

    Parameters
    ----------
    quantile_range : tuple (q_min, q_max), 0.0 < q_min < q_max < 100.0
        Default: (25.0, 75.0) = (1st quantile, 3rd quantile) = IQR
        Quantile range used to calculate ``scale_``.
    copy : boolean, optional, default is True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    left_limit_ : array of floats
        The left value used to truncate.
    right_limit_ : array of floats
        The right value used to truncate.
    """

    def __init__(self, quantile_range=(1.0, 99.0), copy=True):
        self.quantile_range = quantile_range
        self.copy = copy

    def fit(self, X, y=None):
        """Compute the median and quantiles to be used for scaling.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to compute the median and quantiles
            used for later scaling along the features axis.
        """
        # at fit, convert sparse matrices to csc for optimized computation of
        # the quantiles
        X = check_array(
            X, accept_sparse='csc',
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite='allow-nan'
        )

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" %
                             str(self.quantile_range))

        quantiles = []
        for feature_idx in range(X.shape[1]):
            if sparse.issparse(X):
                column_nnz_data = X.data[
                    X.indptr[feature_idx]:
                    X.indptr[feature_idx + 1]
                ]
                column_data = np.zeros(shape=X.shape[0], dtype=X.dtype)
                column_data[:len(column_nnz_data)] = column_nnz_data
            else:
                column_data = X[:, feature_idx]

            quantiles.append(np.nanpercentile(column_data, self.quantile_range))

        quantiles = np.transpose(quantiles)

        self.left_limit_ = quantiles[0]
        self.right_limit_ =  quantiles[1]

        return self

    def transform(self, X):
        """Center and scale the data.
        Parameters
        ----------
        X : {array-like, sparse matrix}
            The data used to scale along the specified axis.
        """
        check_is_fitted(self, attributes=['left_limit_', 'right_limit_'])
        X = check_array(
            X, 
            accept_sparse=('csr', 'csc'), 
            copy=self.copy,
            estimator=self,
            dtype=FLOAT_DTYPES,
            force_all_finite='allow-nan'
        )

        if sparse.issparse(X):
            return 'not implemented'
            inplace_column_scale(X, 1.0 / self.scale_)  # Add the transformation
        else:
            for feature_idx in range(X.shape[1]):
                lq_ = self.left_limit_[feature_idx]
                rq_ = self.right_limit_[feature_idx]
                l_filt = X[:, feature_idx] < lq_
                r_filt = X[:, feature_idx] > rq_
                X[l_filt, feature_idx] = lq_
                X[r_filt, feature_idx] = rq_
        return X

    def _more_tags(self):
        return {'allow_nan': True}
