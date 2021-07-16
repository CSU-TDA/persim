from __future__ import division

import collections

import copy
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator

__all__ = ["PersistencePairOrderer"]

class PersistencePairOrderer(BaseEstimator, TransformerMixin):
    """Transformer which converts persistence diagrams into a feature vector by returning an ordered range of persistence pair coordinates.

    Parameters
    ----------
    start_pair : int >= 0
        The starting index of persistence pair coordinates to save. (default: 0)
    num_pairs : int >= 1 or None
        The number of persistence pair coordinates to save. None retains all pairs (default: None).
    birth_only : bool
        Return only the birth coordinates of persistence pairs. (default: False)
    death_only : bool
        Return only the death coordinates of persistence pairs. (default: False)
    order_by : str
        String specifying how to order persistence pairs. Valid choices 'persistence' (default), 'birth', 'death'
    descending : bool
        Order pairs in decreasing order (default: True)
    n_jobs : int or None
        Number of cores to use to transform diagrams into vectors. -1 uses maximum number available. (default: None, uses a single core).
    """

    def __init__(self, start_pair=0, num_pairs=None, birth_only=False, death_only=False, order_by='persistence', descending=True, n_jobs=None):
        """ PersistencePairOrderer constructor method
        """
        self._validate_parameters(start_pair=start_pair, num_pairs=num_pairs, birth_only=birth_only, death_only=death_only, order_by=order_by, descending=descending, n_jobs=n_jobs)

        self.birth_only = birth_only
        self.death_only = death_only
        self.order_by = order_by
        self.descending = descending
        self.n_jobs = n_jobs

    def fit(self, pers_dgms):
        """ Choose the number of pairs equal to the maximum number of pairs across all persistence pairs across one or more persistence diagrams.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarrays
            Collection of one or more persistence diagrams.
        """
        pers_dgms, singular = self._ensure_iterable(pers_dgms)
        self.num_pairs = max([len(pers_dgm) for pers_dgm in pers_dgms])

    def transform(self, pers_dgms, n_jobs=None):
        """ Transform a persistence diagram or an iterable containing a collection of persistence diagrams into
        a vector of ordered persistence coordinates.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarrays
            Collection of one or more persistence diagrams.
        n_jobs : int or None
            Number of cores to use to transform diagrams into vectors. -1 uses maximum number available. (default: None, uses a single core).

        Returns
        -------
        list
            Collection of numpy.ndarrays encoding fixed-length, ordered persistence pairs in the same order as pers_dgms.
        """
        if n_jobs is not None:
            parallelize = True
        else:
            parallelize = False

        # if diagram is empty, return vector or all zeros
        if len(pers_dgms) == 0:
            return np.zeros(self.num_pairs)

        # convert to a list of diagrams if necessary
        pers_dgms, singular = self._ensure_iterable(pers_dgms)

        # Set default to maximum range
        if self.num_pairs is None:
            self.fit(pers_dgms)

        if parallelize:
            pers_vecs = Parallel(n_jobs=n_jobs)(
                delayed(_transform)(pers_dgm, self.start_pair, self.num_pairs, self.birth_only, self.death_only,
                                    self.order_by, self.descending) for pers_dgm in pers_dgms)
        else:
            pers_vecs = [_transform(pers_dgm, self.start_pair, self.num_pairs, self.birth_only, self.death_only,
                                    self.order_by, self.descending) for
                         pers_dgm in pers_dgms]

        if singular:
            pers_vecs = pers_vecs[0]

        return pers_vecs

    def fit_transform(self, pers_dgms):
        """ Chooses the number of pairs equal to the maximum number of pairs across all persistence pairs across one or
        more persistence diagrams, and then transforms the persistence diagram(s) into vectors of ordered persistence
        coordinates.

        Parameters
        ----------
        pers_dgms : one or an iterable of (-,2) numpy.ndarray
            Collection of one or more persistence diagrams.

        Returns
        -------
        list
            Collection of numpy.ndarrays encoding fixed-length, ordered persistence pairs in the same order as pers_dgms.
        """
        pers_dgms = copy.deepcopy(pers_dgms)

        # fit imager parameters
        self.fit(pers_dgms)

        # transform diagrams to images
        pers_vecs = self.transform(pers_dgms)

        return pers_vecs

    def __repr__(self):
        params = tuple([self.start_pair, self.num_pairs, self.birth_only, self.death_only, self.order_by, self.descending, self.n_jobs])
        repr_str = 'PersistencePairOrderer(start_pair=%s, num_pairs=%s, birth_only=%s, death_only=%s, order_by=%s, descending=%s, n_jobs=%s)' % params
        return repr_str

    def _validate_parameters(self, start_pair=None, num_pairs=None, birth_only=None, death_only=None, order_by=None, descending=None, n_jobs=None):
        valid_order_by = ['persistence', 'birth', 'death']

        # validate start_pair
        if (not isinstance(start_pair, (int, float))) or (start_pair < 0):
            raise TypeError("start_pair must be a non-negative integer.")
        else:
            self.start_pair = int(start_pair)

        # validate num_pairs
        if num_pairs is not None:
            if (not isinstance(num_pairs, (int, float))) or (num_pairs <= 0):
                raise TypeError("num_pairs must be a positive integer.")
            else:
                self.num_pairs = int(num_pairs)
        else:
            self.num_pairs = num_pairs

        # validate birth_only
        if not isinstance(birth_only, bool):
            raise TypeError("birth_only  must be True or False.")

        # validate death_only
        if not isinstance(death_only, bool):
            raise TypeError("death_only  must be True or False.")

        # ensure at most one is selected
        if birth_only and death_only:
            raise ValueError("birth_only and death_only cannot both be True.")

        # validate order_by
        if order_by not in valid_order_by:
            raise ValueError("order_by must be either 'persistence', 'birth', or 'death'.")

        # validate descending
        if not isinstance(descending, bool):
            raise TypeError("descending  must be True or False.")

        # validate n_jobs
        if n_jobs is not None:
            if not isinstance(n_jobs, int) or n_jobs < -1 or n_jobs == 0:
                ValueError("n_jobs must be either a positive integer, -1, or None.")

    def _ensure_iterable(self, pers_dgms):
        # if first entry of first entry is not iterable, then diagrams is singular and we need to make it a list of diagrams
        try:
            singular = not isinstance(pers_dgms[0][0], collections.Iterable)
        except IndexError:
            singular = False

        if singular:
            pers_dgms = [pers_dgms]

        return pers_dgms, singular


def _transform(pers_dgm, start_pair, num_pairs, birth_only, death_only, order_by, descending):
        """ Transform a persistence diagram into an ordered persistence pair vector.
        
        Parameters
        ----------
        start_pair : int >= 0
            The starting index of persistence pair coordinates to save. (default: 0)
        num_pairs : int >= 1 or None
            The number of persistence pair coordinates to save. None retains all pairs (default: None).
        birth_only : bool
            Return only the birth coordinates of persistence pairs. (default: False)
        death_only : bool
            Return only the death coordinates of persistence pairs. (default: False)
        order_by : str
            String specifying how to order persistence pairs. Valid choices 'persistence' (default), 'birth', 'death'
        descending : bool
            Order pairs in decreasing order (default: True)

        Returns
        -------
        numpy.ndarray
            (N,) numpy.ndarray encoding the ordered persistence pair vector corresponding to pers_dgm, where N=num_pairs
            if birth_only or death_only, otherwise N=2*num_pairs
        """
        pers_dgm = np.copy(pers_dgm)

        if order_by == 'persistence':
            persistence = pers_dgm[:,1] - pers_dgm[:,0]
            index_order = persistence.argsort()
        elif order_by == 'birth':
            index_order = pers_dgm[:,0].argsort()
        elif order_by == 'death':
            index_order = pers_dgm[:,1].argsort()

        if descending:
            index_order = np.flip(index_order)

        pers_dgm = pers_dgm[index_order,:]

        if birth_only:
            pair_vec = pers_dgm[start_pair:start_pair+num_pairs, 0]
            num_pads = max(0, num_pairs - len(pers_dgm))
        elif death_only:
            pair_vec = pers_dgm[start_pair:start_pair + num_pairs, 1]
            num_pads = max(0, num_pairs - len(pers_dgm))
        else:
            pair_vec = pers_dgm[start_pair:start_pair + num_pairs, :].flatten()
            num_pads = 2*max(0, num_pairs - len(pers_dgm))

        # pad with zeros if needed
        pair_vec = np.concatenate((pair_vec, np.zeros(num_pads)))


        return pair_vec