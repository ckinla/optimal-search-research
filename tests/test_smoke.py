"""Smoke test verifying the Python research environment is set up."""

import sys

import numpy as np
import scipy

import optimal_search


def test_environment_imports():
    assert sys.version_info[:2] == (3, 12)
    assert optimal_search.__doc__
    assert np.isclose(np.sum(np.arange(4)), 6)
    assert scipy.__version__
