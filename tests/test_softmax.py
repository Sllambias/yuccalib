import pytest

import numpy as np
from yuccalib.utils.softmax import softmax

def test_softmax():
    x = np.array([1, 1])
    assert np.all(softmax(x, axis=0) == np.array([1/2, 1/2]))
