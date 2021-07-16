"""
Constructor of the estimator model and topological feature transformer hyperparameter spaces.
"""

from sklearn.model_selection import _split as sksp, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error
from hyperopt import hp, tpe, atpe, pyll, STATUS_OK, fmin, Trials
from hyperopt import rand as hprand
from hyperopt.exceptions import AllTrialsFailed
import numpy as np
import warnings
import copy
import time
from uuid import uuid4

class TDAModelData():
    """ A structure for storing data samples, persistence diagrams and other sample features for training
    machine learning models.

    Parameters
    ----------
    data : iterable
        Collection of N data samples.
    targets : (N,) numpy array
        Target values or labels associated to each N data samples.
    features : (N,k) numpy array
        k-dimensional feature vectors derived from the N data samples.
    pers_dgms : iterable or dictionary of (*,2) numpy arrays
        A collection of persistence diagrams given either as an iterable collection of (*,2) numpy arrays or a
        dictionary keyed by homological dimension and values equal to iterable collections of (*,2) numpy arrays.
    dim : int or None
        The homological dimension of the persistence diagrams, if given as an iterable collection of (*,2) numpy arrays.
    """

    def __init__(self, pers_dgms=None, targets=None, features=None, data=None, dim=None):
        """ TDAModelData constructor method
        """
        self.data = None
        self.targets = None
        self.features = None
        self.pers_dgms = {}
        if pers_dgms is None:
            pers_dgms = {}

        self.add_data(data, targets)
        self.add_features(features)
        self.add_diagrams(pers_dgms, dim=dim)

    def add_data(self, data=None, targets=None):
        """
        Add an iterable collection of arbitrary data and associated labels/targets.

        Parameters
        ----------
        data : iterable
            Ordered collection of N data samples.
        targets : (N,d) numpy array
            Labels and/or target values associated with the N data samples.
        """
        # Verify input data types are correct
        if data is not None:
            try:
                iter(data)
            except TypeError:
                raise TypeError("The argument data is expected to be an iterable.")

        if targets is not None:
            if (not isinstance(targets, np.ndarray)) or (targets.ndim != 1):
                raise TypeError("The argument targets is expected to be a (N,) numpy array.")

        # Verify the number of data samples matches the number targets
        if data is not None and targets is not None:
            if len(data) != len(targets):
                raise ValueError("All data inputs must be iterables of the same size.")

        # Verify inputs have expected size
        if targets is not None:
            if self.data is not None:
                if len(targets) != len(self.data):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d targets." % len(self.data))

            if self.features is not None:
                if len(targets) != len(self.features):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d targets." % len(self.features))

            if len(self.pers_dgms) > 0:
                dim = list(self.pers_dgms.keys())[0]
                if len(targets) != len(self.pers_dgms[dim]):
                    raise ValueError("All data inputs must be iterables of the same size. Expected %d targets." % len(self.pers_dgms[dim]))

        if data is not None:
            if self.targets is not None:
                if len(data) != len(self.targets):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d data samples." % len(self.targets))

            if self.features is not None:
                if len(data) != len(self.features):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d data samples." % len(self.features))

            if len(self.pers_dgms) > 0:
                dim = list(self.pers_dgms.keys())[0]
                if len(data) != len(self.pers_dgms[dim]):
                    raise ValueError("All data inputs must be iterables of the same size. C Expected %d data samples." % len(self.pers_dgms[dim]))

        self.data = data
        self.targets = targets

    def add_diagrams(self, pers_dgms, dim=None):
        """
        Add a collection of persistence diagrams and verify their format.

        Parameters
        ----------
        pers_dgms : iterable or dict
            Ordered collection of N (*,2) numpy arrays encoding N persistence diagrams, or dictionary with keys equal to
            dimension and values equal to ordered collections of N (*,2) numpy arrays encoding N persistence diagrams.
        dim : int or None
            Non-negative integer specifying the homological dimension of the persistence diagrams.
        """
        # Handle if diagrams are already given in a dictionary
        if isinstance(pers_dgms, dict):
            for dim in pers_dgms.keys():
                if not isinstance(dim, int):
                    raise TypeError("Persistence diagram dictionaries must be keyed by non-negative homological dimensions.")
                if dim < 0:
                    raise ValueError("Persistence diagram dictionaries must be keyed by non-negative homological dimensions.")

                self.add_diagrams(pers_dgms[dim], dim=dim)
        else:
            # Validate diagrams are iterable
            try:
                iter(pers_dgms)
            except TypeError:
                raise TypeError("Persistence diagrams must be stored in an iterable collection of (*,2) numpy arrays.")

            if dim is None:
                raise ValueError("Please specify the homological dimension of the persistence diagrams using the dim argument.")

            for d, dgms in self.pers_dgms.items():
                # Warn if data in the given dimension already exists
                if d == dim:
                    warn_string = "TDAModelData() instance already contains diagrams in dimension %d, overwriting with current input." % dim
                    warnings.warn(warn_string)

                # Verify the correct number of diagrams
                if len(pers_dgms) != len(dgms):
                    raise ValueError("All data inputs must be iterables of the same size. Expected %d diagrams." % len(dgms))

                if self.targets is not None:
                    if len(pers_dgms) != len(self.targets):
                        raise ValueError(
                            "All data inputs must be iterables of the same size.  Expected %d diagrams." % len(self.targets))

                if self.data is not None:
                    if len(pers_dgms) != len(self.data):
                        raise ValueError(
                            "All data inputs must be iterables of the same size.  Expected %d diagrams." % len(self.data))

                if self.features is not None:
                    if len(pers_dgms) != len(self.features):
                        raise ValueError(
                            "All data inputs must be iterables of the same size.  Expected %d diagrams." % len(self.features))

            # Verify type and format of diagrams
            for pers_dgm in pers_dgms:
                if not isinstance(pers_dgm, np.ndarray):
                    raise TypeError("Persistence diagrams are expected to be (*,2)-numpy arrays.")
                if (pers_dgm.ndim != 2) or (pers_dgm.shape[1] != 2):
                    raise ValueError("Persistence diagrams are expected to be (*,2)-numpy arrays.")

                if any(pers_dgm[:, 1] - pers_dgm[:, 0] < 0):
                    raise ValueError("Persistence diagrams are expected to be in birth-death coordinates.")

            self.pers_dgms.update({dim: pers_dgms})

    def add_features(self, features=None):
        """
        Add a collection of sample feature vectors.

        Parameters
        ----------
        features : (N,d) numpy array
            Feature vectors associated with the N data samples.
        """
        # Verify input data type is correct
        if features is not None:
            if (not isinstance(features, np.ndarray)) or (features.ndim != 2):
                raise TypeError("The argument features is expected to be an (N,d) numpy array.")

        # Verify input has expected size
        if features is not None:
            if self.data is not None:
                if len(features) != len(self.data):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d feature vectors." % len(self.data))

            if self.targets is not None:
                if len(features) != len(self.targets):
                    raise ValueError("All data inputs must be iterables of the same size.  Expected %d feature vectors." % len(self.targets))

            if len(self.pers_dgms) > 0:
                dim = list(self.pers_dgms.keys())[0]
                if len(features) != len(self.pers_dgms[dim]):
                    raise ValueError("All data inputs must be iterables of the same size. Expected %d feature vectors." % len(self.pers_dgms[dim]))

        self.features = features

class TDAModelParameterSpace():
    """ A searchable configuration space that specifies the hyperparameters of
        (A) one or more scikit-tda persistence diagram transformer interfaces
        (B) one or more scikit-learn estimator interfaces
       A choice of one of each (A) and (B) comprises a learning model whose inputs include topological feature vectors.
       Variable parameters are specified by hyperopt distribution objects (see [1]).

    Parameters
    ----------
    transformers :
        A dictionary keyed by homological dimension with values equal to one or more scikit-tda persistence diagram transformer interfaces.
    estimators :
        One or a list of scikit-learn estimator or pipeline interfaces.

    Example
    -------

    Notes
    -----
    [1] Bergstra et. al., "Hyperopt: a Python library for model selection and hyperparameter optimization," Computational Science & Discovery, vol. 8, pp. 014008, 2015, https://iopscience.iop.org/article/10.1088/1749-4699/8/1/014008
    [2] Motta et. al., "Hyperparameter Optimization of Topological Features for Machine Learning Applications," 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, pp. 1107-1114, 2019, https://doi.org/10.1109/ICMLA.2019.00185
    """

    def __init__(self, transformers=None, estimators=None, default_params=False):
        """ ModelParameterSpace constructor method.
        """
        # Build the base parameter space for the specified transformer(s) and estimator(s)
        self._build_base(transformers=transformers, estimators=estimators, default_params=default_params)

    def _build_base(self, transformers, estimators, default_params):
        if transformers is None:
            transformers = {}

        if estimators is None:
            estimators = []

        self.space = {'transformer_space': {},
                      'estimator_space': {}}

        # Initialize the transformer subspace(s)
        self.add_transformers(transformers, default_params=default_params)

        # Initialize the estimator subspace(s)
        self.add_estimators(estimators, default_params=default_params)

    @property
    def hp_space(self):
        """
        The valid hyperopt stochastic expression parameter space encoded in self.space. Converts the dictionary
        representation of stochastic expression(s) defining the parameter space into a valid hyperopt dictionary
        of stochastic expression(s).
        """
        hp_space = copy.deepcopy(self.space)

        # Determine depths of choice stochastic expressions
        choice_depths = set(self._choice_depths(hp_space))

        # Convert non-choice stochastic expressions to hyperopt form
        self._se_dict_walk(hp_space, convert_choices_depth=False)

        # Convert choice stochastic expressions to hyperopt form, from deepest to most shallow depths
        for choice_depth in sorted(choice_depths, reverse=True):
            self._se_dict_walk(hp_space, convert_choices_depth=choice_depth)

        # Convert transformer and estimator subspaces into valid hyperopt stochastic expressions
        if len(self.estimators) == 0:
            hp_space['estimator_space'] = {}
        elif len(self.estimators) == 1:
            hp_space['estimator_space'] = list(hp_space['estimator_space'].values())[0]
        else:
            hp_space['estimator_space'] = hp.choice(label='estimator_space',
                                                    options=list(hp_space['estimator_space'].values()))
        if len(self.transformers.keys()) == 0:
            hp_space['transformer_space'] = {}
        for dim in self.transformers.keys():
            if len(self.transformers[dim]) == 1:
                hp_space['transformer_space'][dim] = list(hp_space['transformer_space'][dim].values())[0]
            else:
                hp_space['transformer_space'][dim] = hp.choice(label='transformer_space_%d' % dim,
                                                                options=list(hp_space['transformer_space'][dim].values()))
        return hp_space

    @property
    def estimators(self):
        """
        The list of estimators in the parameter space.
        """
        return list(self.space['estimator_space'].keys())

    @property
    def transformers(self):
        """
        The list of persistence diagram transformers in the parameter space.
        """
        return {dim: list(self.space['transformer_space'][dim].keys()) for dim in self.space['transformer_space'].keys()}

    def _labeler(self, method, space=None, dim=None):
        """
        Labels a persistence diagram transformer or an estimator with a unique label.

        Parameters
        ----------
        method :
            A scikit-TDA transformer or scikit-learn estimator interface.
        space : str
            The top level parameter-space key to iterate over.
        """
        if space == 'estimator_space':
            space_dict = self.space[space]
        elif dim in self.space[space]:
            space_dict = self.space[space][dim]
        else:
            space_dict = {}

        count = 1
        for current_label, model_dict in space_dict.items():
            if isinstance(method(), model_dict['method']):
                count += 1

        if dim is None:
            label = '%s_%d' % (method.__name__, count)
        else:
            label = '%s_%d_%d' % (method.__name__, dim, count)

        return label

    def add_transformers(self, transformers, default_params=False):
        """
        Add one or more persistence diagram transformers to the hyperparameter space.

        Parameters
        ----------
        transformers :
            A dictionary keyed by homological dimension with values equal to one or more scikit-tda persistence diagram
            transformer interfaces.
        default_params : bool
            Add the default parameters of the transformer(s).
        """
        self._validate_transformers(transformers)

        for dim, trans in transformers.items():
            trans = self._ensure_iterable(trans)
            for tran in trans:
                transformer_label = self._labeler(tran, space='transformer_space', dim=dim)
                if dim not in self.space['transformer_space']:
                    self.space['transformer_space'][dim] = {transformer_label: {'method': tran, 'params': {}}}
                else:
                    self.space['transformer_space'][dim].update({transformer_label: {'method': tran, 'params': {}}})

                if default_params:
                    self.space['transformer_space'][dim][transformer_label]['params'].update(tran().get_params())

    def remove_transformers(self, transformers):
        """
        Remove one or more persistence diagram transformers from the hyperparameter space.

        Parameters
        ----------
        transformers :
            One or more scikit-tda persistence diagram transformer interfaces, or transformer label strings, or ints.
        """
        transformers = self._ensure_iterable(transformers)
        dims = self.space['transformer_space'].copy().keys()
        for dim in dims:
            for transformer in transformers:
                if isinstance(transformer, int):
                    try:
                        self.space['transformer_space'].pop(transformer)
                    except KeyError:
                        pass
                elif isinstance(transformer, str):
                    try:
                        self.space['transformer_space'][dim].pop(transformer)
                    except KeyError:
                        pass
                else:
                    for current_label, model_dict in self.space['transformer_space'][dim].copy().items():
                        try:
                            if isinstance(transformer(), model_dict['method']):
                                self.space['transformer_space'][dim].pop(current_label)
                        except TypeError:
                            pass

            try:
                if len(self.space['transformer_space'][dim]) == 0:
                    self.space['transformer_space'].pop(dim)
            except KeyError:
                pass

    def add_estimators(self, estimators, default_params=False):
        """
        Add one or more estimators to the hyperparameter space.

        Parameters
        ----------
        estimators :
            One or more scikit-learn estimator interfaces.
        default_params : bool
            Add the default parameters of the estimator.
        """
        self._validate_estimators(estimators)
        estimators = self._ensure_iterable(estimators)

        for estimator in estimators:
            estimator_label = self._labeler(estimator, space='estimator_space')
            self.space['estimator_space'].update({estimator_label: {'method': estimator, 'params': {}}})
            if default_params:
                self.space['estimator_space'][estimator_label]['params'].update(estimator().get_params())

        self._estimator_types()

    def remove_estimators(self, estimators):
        """
        Remove one or more estimators from the hyperparameter space.

        Parameters
        ----------
        estimators :
            One or more scikit-learn estimator or pipeline interfaces or estimator label strings.
        """
        estimators = self._ensure_iterable(estimators)

        for estimator in estimators:
            if isinstance(estimator, str):
                try:
                    self.space['estimator_space'].pop(estimator)
                except KeyError:
                    pass
            else:
                for current_label, model_dict in self.space['estimator_space'].copy().items():
                    if isinstance(estimator(), model_dict['method']):
                        self.space['estimator_space'].pop(current_label)

    def add_transformer_params(self, transformer, se_dict):
        """
        Add a collection of transformer hyperparameters to one or more persistence diagram transformers in the hyperparameter space.

        Parameters
        ----------
        transformer :
            A scikit-tda persistence diagram transformer interface or transformer label string
        se_dict :
            The dictionary output of a ModelParameterSpace() distribution: A dictionary keyed with a transformer
            parameter label with value equal to a dictionary keyed by the distribution type with value equal to a
            dictionary of arguments needed to specify the distribution
        """
        dims = self.space['transformer_space'].keys()
        for dim in dims:
            if isinstance(transformer, str):
                try:
                    self.space['transformer_space'][dim][transformer]['params'].update(se_dict)
                except KeyError:
                    pass
            else:
                for current_label, model_dict in self.space['transformer_space'][dim].copy().items():
                    if isinstance(transformer(), model_dict['method']):
                        model_dict['params'].update(se_dict)

    def remove_transformer_params(self, transformer, params):
        """
        Remove a collection of transformer hyperparameters from one or more persistence diagram transformers in the hyperparameter space.

        Parameters
        ----------
        transformer :
            A scikit-tda persistence diagram transformer interface or transformer label string
        params : string or iterable of strings
            One or more parameter labels to remove from the transformer parameter space
        """
        params = self._ensure_iterable(params)
        dims = self.space['transformer_space'].keys()
        for param in params:
            for dim in dims:
                if isinstance(transformer, str):
                    try:
                        self.space['transformer_space'][dim][transformer]['params'].pop(param)
                    except KeyError:
                        pass
                else:
                    for current_label, model_dict in self.space['transformer_space'][dim].copy().items():
                        if isinstance(transformer(), model_dict['method']):
                            model_dict['params'].pop(param)

    def add_estimator_params(self, estimator, se_dict):
        """
        Add a collection of estimator hyperparameters to one or more estimators in the hyperparameter space.

        Parameters
        ----------
        estimator :
            A scikit-learn estimator or pipeline interface or estimator label string.
        se_dict :
            The dictionary output of a ModelParameterSpace() distribution: A dictionary keyed with an estimator
            parameter label with value equal to a dictionary keyed by the distribution type with value equal to a
            dictionary of arguments needed to specify the distribution.
        """
        if isinstance(estimator, str):
            self.space['estimator_space'][estimator]['params'].update(se_dict)
        else:
            for current_label, model_dict in self.space['estimator_space'].copy().items():
                if isinstance(estimator(), model_dict['method']):
                    model_dict['params'].update(se_dict)

    def remove_estimator_params(self, estimator, params):
        """
        Remove a collection of estimator hyperparameters from one or more estimators in the hyperparameter space.

        Parameters
        ----------
        estimator :
            A scikit-learn estimator or pipeline interface or estimator label string.
        params : string or iterable of strings
            One or more parameter labels to remove from the estimator parameter space.
        """
        params = self._ensure_iterable(params)

        for param in params:
            if isinstance(estimator, str):
                self.space['estimator_space'][estimator]['params'].pop(param)
            else:
                for current_label, model_dict in self.space['estimator_space'].copy().items():
                    if isinstance(estimator(), model_dict['method']):
                        model_dict['params'].pop(param)

    @staticmethod
    def choice(options):
        """
        Parameters
        ----------
        options : list
            A list of possible valid choices for param
        """
        return {'choice_se': {'options': options}}

    @staticmethod
    def randint(upper):
        """
        Parameters
        ----------
        upper : non-negative int
            The upperbound of allowable integers
        """
        return {'randint_se': {'upper': upper}}

    @staticmethod
    def uniform(low, high):
        """
        Parameters
        ----------
        low : float
            The lowerbound of the uniform distribution range
        high : float
            The upperbound of the uniform distribution range
        """
        return {'uniform_se': {'low': low, 'high': high}}

    @staticmethod
    def quniform(low, high, q):
        """
        Parameters
        ----------
        low : float
            The lowerbound of the uniform distribution range
        high : float
            The upperbound of the uniform distribution range
        q : float
            The discrete quantity interval size
        """
        return {'quniform_se': {'low': low, 'high': high, 'q': q}}

    @staticmethod
    def loguniform(low, high):
        """
        Parameters
        ----------
        low : float
            The lowerbound of the distribution range: exp(low)
        high : float
            The upperbound of the distribution range: exp(high)
        """
        return {'loguniform_se': {'low': low, 'high': high}}

    @staticmethod
    def qloguniform(low, high, q):
        """
        Parameters
        ----------
        low : float
            The lowerbound of the distribution range: exp(low)
        high : float
            The upperbound of the distribution range: exp(high)
        q : float
            The discrete quantity interval size
        """
        return {'qloguniform_se': {'low': low, 'high': high, 'q': q}}

    @staticmethod
    def normal(mu, sigma):
        """
        Parameters
        ----------
        mu : float
            The mean of the normal distribution
        sigma : float
            The standard deviation of the normal distribution
        """
        return {'normal_se': {'mu': mu, 'sigma': sigma}}

    @staticmethod
    def qnormal(mu, sigma, q):
        """
        Parameters
        ----------
        mu : float
            The mean of the normal distribution
        sigma : float
            The standard deviation of the normal distribution
        q : float
            The discrete quantity interval size
        """
        return {'qnormal_se': {'mu': mu, 'sigma': sigma, 'q': q}}

    @staticmethod
    def lognormal(mu, sigma):
        """
        Parameters
        ----------
        mu : float
            The mean of the normal distribution
        sigma : float
            The standard deviation of the normal distribution
        """
        return {'lognormal_se': {'mu': mu, 'sigma': sigma}}

    @staticmethod
    def qlognormal(mu, sigma, q):
        """
        Parameters
        ----------
        mu : float
            The mean of the normal distribution
        sigma : float
            The standard deviation of the normal distribution
        q : float
            The discrete quantity interval size
        """
        return {'qlognormal_se': {'mu': mu, 'sigma': sigma, 'q': q}}

    def sample(self):
        """
        Draw a random sample from the current parameter space.
        """
        return pyll.stochastic.sample(self.hp_space)

    def _choice_depths(self, node, depth=0, choice_depths=None):
        """
        Determine the depths in the nested-dictionary hyperparameter space where choice stochastic expressions occur
        """
        if choice_depths is None:
            choice_depths = []

        if isinstance(node, dict):
            depth += 1
            for key, value in node.items():
                if key == 'choice_se':
                    choice_depths.append(depth)

                if isinstance(value, dict):
                    self._choice_depths(value, depth=depth, choice_depths=choice_depths)

                elif isinstance(value, list) or isinstance(value, tuple):
                    for item in value:
                        self._choice_depths(item, depth=depth, choice_depths=choice_depths)

        return choice_depths

    def _se_dict_walk(self, node, parent_node=None, parent_key=None, choice_label=None, ind=None, depth=0, convert_choices_depth=False):
        """
        Walk a nested-dictionary hyperparameter space and convert values to valid hyperopt stochastic expressions.
        """
        se_labels = ['randint_se', 'uniform_se', 'quniform_se', 'loguniform_se', 'qloguniform_se',
                 'normal_se', 'qnormal_se', 'lognormal_se', 'qlognormal_se']

        if isinstance(node, dict):
            depth += 1
            for key, value in node.items():
                if (key == 'choice_se') and (choice_label is None):
                    choice_label = parent_key

                if ((key == 'choice_se') and (convert_choices_depth == depth)) or (key in se_labels):
                    se_label = key
                    se_args = value
                    if ind is not None:
                        parent_node[parent_key][ind] = self._hpse(se_label)(label=choice_label+str(uuid4()), **se_args)
                    else:
                        parent_node[parent_key] = self._hpse(se_label)(label=parent_key+str(uuid4()), **se_args)

                if isinstance(value, dict):
                    self._se_dict_walk(value, parent_node=node, parent_key=key, choice_label=choice_label, ind=None,
                                       depth=depth, convert_choices_depth=convert_choices_depth)

                elif isinstance(value, list) or isinstance(value, tuple):
                    ind = 0
                    for item in value:
                        self._se_dict_walk(item, parent_node=node,  parent_key=key, choice_label=choice_label, ind=ind,
                                           depth=depth, convert_choices_depth=convert_choices_depth)
                        ind += 1

    def _hpse(self, se_label):
        """
        Returns the hyperopt stochastic expression function specified by an allowable string.

        Parameters
        ----------
        se_label : str
            A string encoding a valid hyperopt stochastic expression. Valid choices: 'choice_se', 'randint_se',
            'uniform_se', 'quniform_se', 'loguniform_se', 'qloguniform_se', 'normal_se', 'qnormal_se', 'lognormal_se',
            'qlognormal_se'

        Returns
        -------
        Valid hyperopt stochastic expression
        """
        str_to_se_dict = {'choice_se': hp.choice, 'randint_se': hp.randint, 'uniform_se': hp.uniform,
                          'quniform_se': hp.quniform, 'loguniform_se': hp.loguniform, 'qloguniform_se': hp.qloguniform,
                          'normal_se': hp.normal, 'qnormal_se': hp.qnormal, 'lognormal_se': hp.lognormal,
                          'qlognormal_se': hp.qlognormal}
        try:
            return str_to_se_dict[se_label]
        except KeyError:
            raise ValueError("Invalid stochastic expression type. Valid choices are 'choice', 'randint', 'uniform', 'quniform', 'loguniform', 'qloguniform', 'normal', 'qnormal', 'lognormal', and 'qlognormal'.")

    def _validate_transformers(self, transformers):
        """
        Validate that all transformers are valid.

        Parameters
        ----------
        transformers :
            One or a list of scikit-tda persistence diagram to feature vector transformer interfaces
        """
        if not isinstance(transformers, dict):
            raise TypeError("Transformers must be a dictionary keyed with non-negative integer homological dimensions.")
        else:
            for dim, trans in transformers.items():
                if not isinstance(dim, int):
                    raise TypeError("Persistence diagram dictionaries must be keyed by non-negative homological dimensions.")
                if dim < 0:
                    raise ValueError("Persistence diagram dictionaries must be keyed by non-negative homological dimensions.")

                trans = self._ensure_iterable(trans)

                for tran in trans:
                    if isinstance(tran, object):
                        transform = getattr(tran, "transform", None)
                        if not callable(transform):
                            raise TypeError('A transformer must be valid persistence diagram transformation function with a .transform() method.')
                    else:
                        raise TypeError('A transformer must be valid persistence diagram transformation function with a .transform() method.')

            return True

    def _validate_estimators(self, estimators):
        """
        Validate that all estimators are valid.

        Parameters
        ----------
        estimators :
            One or a list of scikit-learn estimator or pipeline interfaces
        """
        try:
            iter(estimators)
        except TypeError:
            estimators = [estimators]

        for estimator in estimators:
            if isinstance(estimator, object):
                fit = getattr(estimator, "fit", None)
                predict = getattr(estimator, "predict", None)
                if (not callable(fit)) or (not callable(predict)):
                    raise TypeError('An estimator must be valid scikit-learn model with .fit() and .predict() methods.')
            else:
                raise TypeError('An estimator must be valid scikit-learn model with .fit() and .predict() methods.')

        return True

    def _ensure_iterable(self, var):
        """
        Converts any object to a list containing the object, if the object isn't already an iterable non-string.
        """
        if isinstance(var, str):
            var = [var]

        try:
            iter(var)
        except TypeError:
            var = [var]

        return var

    def _estimator_types(self):
        estimator_types = [getattr(self.space['estimator_space'][estimator_label]['method'], "_estimator_type", None)
                           for estimator_label in self.estimators]
        if len(set(estimator_types)) > 1:
            warnings.warn("Estimators are of different types. This may lead to unexpected results.")

        return estimator_types

class TDAModelParameterSearchCV():
    """ Class enabling hyperparamter optimization of a supervised learning model by maximizing cross-validation of a
       model evaluation score by searching over a possibly-conditional hyperparameter space determined by a choice of
        (A) one or more scikit-tda persistence diagram transformer interfaces and
        (B) one or more scikit-learn estimator interfaces.
       A choice of one of each (A) and (B) determines a learning model whose inputs include topological feature vectors.
       A point in the parameter space is expected to be a dictionary with two top level keys equal to
       'transformer_space' and 'estimator_space' with values equal to a dictionary containing two keys:
       'method' and 'params'. The value of the 'method' entries is expected to be a scikit-learn estimator interface or
       a scikit-TDA diagram transformer interface, while the value of 'params' is a dictionary of arguments needed to
       specify the method. Use TDAModelParameterSpace() to construct a valid hyperparameter space.

    Parameters
    ----------
    param_space : TDAModelParameterSpace instance
        The hyperparameter space over which to perform optimzation.
    scoring : string or callable
        A single string (see [3]) or a callable (see [4]) to evaluate the predictions on the test set.
    cv : int, cross-validation generator or an iterable, default=5
        Determines the cross-validation splitting strategy. Possible inputs for cv are:
            - integer, to specify the number of folds in a (Stratified)KFold,
            - CV splitter,
            - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if the estimator is a classifier and the target is either binary or multiclass,
        StratifiedKFold is used. In all other cases, KFold is used.
        Refer to [5] for the various cross-validation strategies that can be used here.
    method : string or hyperopt suggest method
        Determines which algorithm to use to search the hyperparameter space. Valid inputs for method are:
            - 'random' (hyperopt.rand.suggest) - performs an naive random search without updating parameter distributions
            - 'tpe' (hyperopt.tpe.suggest) - use Tree of Parzen Estimators (see [6])
            - 'atpe' (hyperopt.atpe.suggest) - use adaptive Tree of Parzen Estimtors (see [7]).
    n_candidates : int, default=100
        The number of candidate parameters to sample and evaluate.
    refit : bool, default=True
        If True, refit an estimator using the best found parameters on the whole dataset. The refitted estimator is made
        available at the best_estimator attribute.
    n_jobs : int or None
        Number of jobs to run in parallel. Training the estimator and computing the score are parallelized over the cross-validation splits. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

    Example
    -------

    Notes
    -----
    [1] Bergstra et. al., "Hyperopt: a Python library for model selection and hyperparameter optimization," Computational Science & Discovery, vol. 8, pp. 014008, 2015, https://iopscience.iop.org/article/10.1088/1749-4699/8/1/014008
    [2] Motta et. al., "Hyperparameter Optimization of Topological Features for Machine Learning Applications," 18th IEEE International Conference On Machine Learning And Applications (ICMLA), Boca Raton, FL, USA, pp. 1107-1114, 2019, https://doi.org/10.1109/ICMLA.2019.00185
    [3] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    [4] https://scikit-learn.org/stable/modules/model_evaluation.html#scoring
    [5] https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation
    [6] Bergstra et. al., "Algorithms for Hyper-Parameter Optimization," Advances in Neural Information Processing Systems, vol. 24, 2011, https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf
    [7] https://github.com/electricbrainio/hypermax
    """

    def __init__(self, param_space, scoring=None, cv=5, method='random', n_candidates=100, refit=True, n_jobs=None):
        """ TDAParameterSearchCV constructor method
        """
        # Validate parameters
        self._validate_arguments(param_space=param_space, n_candidates=n_candidates, refit=refit, n_jobs=n_jobs)

        self.param_space = param_space
        self.scoring = self._ensure_valid_scoring(scoring)
        self.cv = self._ensure_valid_cv(cv)
        self.method = self._ensure_valid_method(method)
        self.n_candidates = n_candidates
        self.refit = refit
        self.n_jobs = n_jobs
        self.trials = Trials()
        self._searched = False

    @property
    def best_transformer(self):
        """
        An instance of the best persistence diagram transformer with the best choice of hyperparameters found by .search().
        """
        try:
            best_transformer = {}
            for dim, transformer in self.trials.best_trial['result']['params']['transformer_space'].items():
                best_transformer.update({dim: transformer['method'](**transformer['params'])})
            return best_transformer
        except AllTrialsFailed:
            return None

    @property
    def best_estimator(self):
        """
        An instance of the best estimator with the best choice of hyperparameters found by .search().
        """
        try:
            best_estimator_method = self.trials.best_trial['result']['params']['estimator_space']['method']
            best_estimator_params = self.trials.best_trial['result']['params']['estimator_space']['params']
            return best_estimator_method(**best_estimator_params)
        except AllTrialsFailed:
            return None

    def _transform_dgms(self, param_sample, pers_dgms):
        """
        Transforms a collection of persistence diagrams into feature vectors using the transformer interface and
        associated parameters specified by the output of TDAModelParameterSpace().sample().

        Parameters
        ----------
        param_sample : dict
            A sample from a TDA model hyperparameter space, i.e. output of TDAModelParameterSpace().sample().
        pers_dgms :
            Dictionary of ordered collections of N persistence diagrams, with keys encoding homological
            dimension.

        Returns
        -------
        dgm_features : numpy.ndarray
            NxK numpy array encoding the K-dimensional feature vectors associated to the N persistence diagrams.
        """
        dims = param_sample['transformer_space'].keys()

        dgm_features = None
        for dim in dims:
            transformer_method = param_sample['transformer_space'][dim]['method']
            transformer_params = param_sample['transformer_space'][dim].get('params', {})

            transformer = transformer_method(**transformer_params)

            X = transformer.transform(pers_dgms[dim])
            X = np.array([features.flatten() for features in X])
            if dgm_features is None:
                dgm_features = X
            else:
                dgm_features = np.concatenate((dgm_features, X), axis=1)
                                                                         # TODO: could speed up by pre-allocating array of correct size:
                                                                         #  would require transformers have method to return num features
        return dgm_features

    def fit(self, model_data):
        """
        Fit the best estimator using model_data and the best known choice of persistence diagram transformer and estimator hyperparameters.

        Parameters
        ----------
        model_data : TDAModelData instance
            A collection of persistence diagrams, optional additional feature vectors, and target values.
        """
        # If search has been performed, fit using optimal parameters
        if self._searched:
            dgm_features = self._transform_dgms(self.trials.best_trial['result']['params'], model_data.pers_dgms)
            if model_data.features is not None:
                X = np.concatenate((dgm_features, model_data.features), axis=1)
            else:
                X = dgm_features

            # Extract model data target values
            y = model_data.targets

            self.best_estimator.fit(X, y)

        # Else first perform the search and fit using optimal parameters
        else:
            self.search(model_data)
            self.fit(model_data)

    def search(self, model_data):
        """
        Search for the choice of model hyperparameters which maximizes the cross-validation score on the supplied data.

        Parameters
        ----------
        model_data : TDAModelData instance
            A collection of persistence diagrams, optional additional feature vectors, and target values.
        """
        # Define the objective function over the hyperparameter space
        objective = lambda params: self._cv_loss(params, model_data=model_data)

        # Search the hyperparameter space to minimize the objective function (maximize the cross-validation score)
        fmin(fn=objective,
             space=self.param_space.hp_space,
             algo=self.method,
             max_evals=self.n_candidates,
             trials=self.trials)

        self._searched = True

        if self.refit:
            self.fit(model_data)

    def _cv_loss(self, param_sample, model_data):
        """
        The cross-validation loss function to be minimized.

        Parameters
        ----------
        param_sample : dict
            A sample from a TDA model hyperparameter space, i.e. output of TDAModelParameterSpace().sample().
        model_data : TDAModelData instance
            Structured collection of data, persistence diagrams, data feature vectors, and target values.
        """
        starttime = time.time()

        # Construct estimator
        estimator_method = param_sample['estimator_space']['method']
        estimator_params= param_sample['estimator_space'].get('params', {})
        estimator = estimator_method(**estimator_params)

        # Transform diagrams into feature vectors
        dgm_features = self._transform_dgms(param_sample, model_data.pers_dgms)

        # Concatenate diagram features with other data features, if they exist
        if model_data.features is not None:
            X = np.concatenate((dgm_features, model_data.features), axis=1)
        else:
            X = dgm_features

        # Extract model data target values
        y = model_data.targets

        # Compute cross validation scores and average loss
        scores = cross_val_score(estimator, X, y, cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs)
        loss = -scores.mean()

        endtime = time.time()

        return {'loss': loss, 'scores': scores, 'status': STATUS_OK, 'params': param_sample, 'scoring': self.scoring, 'eval_time': endtime - starttime}

    def _ensure_valid_scoring(self, scoring):
        """
        Ensures the scoring argument is of the expected type.
        """
        # Set default based on estimator type in the hyperparameter space
        if scoring is None:
            if self._estimators_are_classifiers():
                scoring = make_scorer(accuracy_score, greater_is_better=True)
            else:
                scoring = make_scorer(mean_squared_error, greater_is_better=False)

        # Validate the scoring function
        if scoring.__class__.__name__ in ['_PredictScorer', '_ProbaScorer', '_ThresholdScorer']:
            return scoring
        else:
            raise TypeError("scoring must be a scorer object made with sklearn.metrics.make_scorer().")

    def _ensure_valid_cv(self, cv):
        """
        Ensures the cv argument is valid and of the expected type.
        """
        return sksp.check_cv(cv, y=None, classifier=self._estimators_are_classifiers())

    def _ensure_valid_method(self, method):
        """
        Ensures the method argument is valid and of the expected type.
        """
        valid_str_to_method = {'random': hprand.suggest, 'tpe': tpe.suggest, 'atpe': atpe.suggest}

        if method in valid_str_to_method.keys():
            return valid_str_to_method[method]
        elif method in valid_str_to_method.values():
            return method
        else:
            raise ValueError("method must be a str in %s, or hyperopt function in %s" % (list(valid_str_to_method.keys()),
                                                                                         [val.__module__ + '.' + val.__name__
                                                                                          for val in valid_str_to_method.values()]))

    def _estimators_are_classifiers(self):
        """
        Check if all estimators in the parameter space are classifiers.
        """
        if self.param_space is not None:
            estimator_types = set(self.param_space._estimator_types())
            if len(estimator_types) != 1:
                return False
            elif list(estimator_types)[0] == 'classifier':
                return True
        else:
            return False

    @staticmethod
    def _validate_arguments(param_space=None, n_candidates=None, refit=None, n_jobs=None):
        """
        Validates other class argument types and values.
        """
        # Validate n_candidates
        if not isinstance(n_candidates, int):
            raise TypeError("n_candidates must be a positive integer.")
        elif n_candidates <= 0:
            raise ValueError("n_candidates must be a positive integer.")

        # Validate refit
        if not isinstance(refit, bool):
            raise TypeError("refit must be True or False.")

        # Validate param_space
        if not isinstance(param_space, TDAModelParameterSpace):
            raise TypeError("param_space must be a TDAModelParameterSpace instance.")

        # Validate n_jobs
        if n_jobs is not None:
            if isinstance(n_jobs, int):
                if n_jobs != -1 and n_jobs <= 0:
                    raise ValueError("n_jobs must be a positive integer, -1, or None.")
            else:
                raise TypeError("n_jobs must be a positive integer, -1, or None.")