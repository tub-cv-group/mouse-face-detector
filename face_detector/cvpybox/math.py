"""
    Part of Mouse Face Detector

    Author: Manuel Wöllhaf
    Contact: n.andresen@tu-berlin.de, andresen.niek@gmail.com
    Affiliation: TU Berlin, Computer Vision & Remote Sensing
    License: MIT

    Part of the publication:
    "Automatic pain face analysis in mice: Applied to a varied dataset with non-standardized conditions"
    DOI: not yet published

    Created: 2018-09-01
    Last Modified: 2026-01-30

    This file contains various utility functions.
"""

import numpy as np


def zero_divide(a, b):
    """ Divide arguments element-wise using numpy.true_divide. Returns zero in any cases that
        result in nan or infinite values (e.g. division by zero).
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        if len(c.shape) > 0:
            c[~np.isfinite(c)] = 0
            c = np.nan_to_num(c)
        else:
            c = c if np.isfinite(c) else 0
            c = np.nan_to_num(c).item()

    return c


def max_joint(X, nclasses, dist=None):
    """ Get the element that maximizes the joint probability.

    Args:
    X: numpy array (n, d)
      Array containing n elements of size d.

    Returns:
      elem: numpy array (d,)
    """
    dist = None if dist is None else np.array(dist)

    if X.shape[0] == 1:  # only on element
        return np.squeeze(X)
    elif len(X.shape) < 2:  # only on dimension
        p = np.bincount(X, minlength=nclasses)
        p = p if dist is None else zero_divide(p, dist)
        return np.argmax(p)
    else:
        # compute one bincount per column (dimension)
        p = np.vstack([np.bincount(X[:, i], minlength=nclasses) for i in range(X.shape[1])])
        # compensate for prior distribution
        p = p if dist is None else zero_divide(p, dist)
        # normalize
        p = zero_divide(p, np.expand_dims(p.sum(axis=1), axis=1))
        # evaluate
        idx = np.argmax(np.sum(np.log(p[range(X.shape[1]), X]), axis=1))
        return X[idx]
