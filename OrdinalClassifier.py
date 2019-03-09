import numbers
import numpy as np
import sklearn

from six import with_metaclass
from abc import ABCMeta
from sklearn.base import ClassifierMixin
from sklearn.ensemble.base import BaseEnsemble
from sklearn.utils import check_X_y, check_array, column_or_1d
from sklearn.utils.multiclass import check_classification_targets

from sklearn.externals.joblib import Parallel, delayed #For parallel computing TODO: check if we need to be parallel or not
from sklearn.utils.validation import has_fit_parameter, check_is_fitted

def _transform_data(ordered_class, class_value, y):
    """" private function used to transform the data into len(ordered_classes)-1 derived datasets of binary classification problems
        returns a pair of (class_value, derived_y)
    """
    ordered_class = ordered_class#.tolist()
    y_derived = [int(ordered_class.index(i) > ordered_class.index(class_value)) for i in y]

    return y_derived

def _build_classifier(binary_classifier, X, y_derived):
    """Private function used to build a batch of classifiers within a job."""
    # Build classifier
    return binary_classifier.fit(X, np.array(y_derived))


def _build_classifiers(binary_classifier, ordered_class, X, y):
    """Private function used to build a batch of classifiers within a job."""
    # Build classifiers
    classifiers = []
    for i in range(0,len(ordered_class)-1): #I Assume that this will be in the correct order of the class values! pass on the last one as it not needed in the prediction phase
        y_derived = _transform_data(ordered_class, ordered_class[i], y)
        classifier = _build_classifier(binary_classifier(), X, y_derived)
        classifiers.append(classifier) # TODO: Approve that the order is maintained!

    return classifiers

def _predict_proba(binary_classifiers, n_classes, X):
    """Private function used to compute (proba-)predictions."""
    n_samples = X.shape[0]
    final_proba = np.zeros((n_samples, n_classes))

    for i in range(0, n_classes):
        if i==0:
            current_proba = binary_classifiers[i].predict_proba(X)
            current_index = binary_classifiers[i].classes_.tolist().index(1)
            final_proba[:,i] += 1 - current_proba[:,current_index]
        else:
            previous_proba = binary_classifiers[i - 1].predict_proba(X)
            previous_index = binary_classifiers[i - 1].classes_.tolist().index(1)
            if i==n_classes-1:
                final_proba[:, i] += previous_proba[:, previous_index]
            else:
                current_proba = binary_classifiers[i].predict_proba(X)
                current_index = binary_classifiers[i].classes_.tolist().index(1)
                final_proba[:, i] += previous_proba[:, previous_index] - current_proba[:,current_index]

    return final_proba

def _decision_function(estimators, estimators_features, X):
    """Private function used to compute decisions within a job."""
    # TODO: Check if we need this!
    pass

class OrdinalClassifier(with_metaclass(ABCMeta, BaseEnsemble, ClassifierMixin)):
    """Base class for ordinal meta-classifier.

    """

    def __init__(self,
                 base_classifier=None,
                 ordered_class=None,
                 max_samples=1.0,
                 max_features=1.0,
                 warm_start=False,
                 n_jobs=1,
                 verbose=0):

        self.base_classifier = base_classifier
        self.ordered_class = ordered_class
        self.max_samples = max_samples
        self.max_features = max_features
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.verbose = verbose

        super(OrdinalClassifier, self).__init__(
            base_estimator=base_classifier,
            n_estimators=len(ordered_class))


    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        pass

    def fit(self, X, y, sample_weight=None):
        """Build a ordinal meta classifier of binary classifiers from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        return self._fit(X, y, self.max_samples, sample_weight=sample_weight)

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        """Build a ordinal meta classifier of binary classifiers from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        max_samples : int or float, optional (default=None)
            Argument to use instead of self.max_samples.

        max_depth : int, optional (default=None)
            Override value used when constructing base estimator. Only
            supported if the base estimator has a max_depth parameter.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        # Convert data
        X, y = check_X_y(X, y, ['csr', 'csc'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, (numbers.Integral, np.integer)):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        # Store validated integer feature sampling value
        self._max_features = max_features

        # ---------------------------------------------Our CODE

        # Here we train the sub classifiers. for each classifier we transform the y differently according to the method
        self.classifiers_ = _build_classifiers(self.base_classifier, self.ordered_class, X, y)

        #----------------------------------------------END of Our CODE

        return self

    def _validate_y(self, y):
        y = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_ = np.array(self.ordered_class)
        self.n_classes_ = len(self.classes_)

        return y

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is computed as the class with
        the highest mean predicted probability. If base estimators do not
        implement a ``predict_proba`` method, then it resorts to voting.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the base estimators in the
        ensemble. If base estimators do not implement a ``predict_proba``
        method, then it resorts to voting and the predicted class probabilities
        of an input sample represents the proportion of estimators predicting
        each class.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        #check_is_fitted(self, "classes_")
        # Check data
        X = check_array(X, accept_sparse=['csr', 'csc'])
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))

        # ---------------------------------------------Our CODE
        proba = _predict_proba(self.classifiers_, self.n_classes_, X)
        #----------------------------------------------END of Our CODE


        return proba

    def decision_function(self, X):
        pass
        #TODO: check if we need this!

    def print_tree(self,index,out_file='self.classifiers_[index]'):
        sklearn.tree.export_graphviz(self.classifiers_[index], out_file=out_file)
        # return self.classifiers_[index]