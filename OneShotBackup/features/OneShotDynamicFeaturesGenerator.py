from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import (check_is_fitted, check_random_state,
                                FLOAT_DTYPES)
from features.ActionGenerator import ActionGenerator
from features.AratiosGenerator import AratiosGenerator
from features.BehavioralGenerator import BehavioralGenerator
from features.GapsGenerator import GapsGenerator


class OneShotDynamicFeaturesGenerator(BaseEstimator, TransformerMixin):
    """ OneShot Dynamic Features
    """

    def __init__(self, generateActions=False, generateAratios=False, generateBehavioral=False, generateGaps=False):
        self.generators = []

        if generateGaps:
            self.generators.append(GapsGenerator())
        if generateBehavioral:
            self.generators.append(BehavioralGenerator())
        if generateAratios:
            self.generators.append(AratiosGenerator())
        if generateActions:
            self.generators.append(ActionGenerator())


    def _reset(self):
        pass

    def fit(self, X, y=None):
        """.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y : Passthrough for ``Pipeline`` compatibility.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        warn_on_dtype=True, estimator=self, dtype=FLOAT_DTYPES)

        for generator in self.generators:
            generator.fit(X, y)

        return self

    def transform(self, X, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.
        """

        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        #TODO: here I need to add the transformation using the table from the "fit" stage.
        for generator in self.generators:
            X = generator.transform(X)

        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Transformed array.
        """
        check_is_fitted(self, 'scale_')

        #TODO: in order to perform inverse I need to keep the added columns. It might be unessary.
        for generator in self.generators:
            X = generator.inverse_transform(X)

        return X
