"""Stage 2: Build the initial simplex tableau from standard form matrices."""

import numpy as np
from typing import Tuple

class TableauBuilder:
    """Construct the tableau used by the simplex algorithm.

    The tableau has the form:
        [ -c_std | 0 ]
        [  A_std | b ]
    where the first row encodes the objective function.
    """

    def __init__(self, c_std: np.ndarray, A_std: np.ndarray, b: np.ndarray):
        self.c_std = c_std
        self.A_std = A_std
        self.b = b
        self.m, self.n = A_std.shape

    def build(self) -> np.ndarray:
        """Return the initial tableau as a 2‑D NumPy array."""
        top = np.hstack([-self.c_std, np.array([0.0])])
        bottom = np.hstack([self.A_std, self.b.reshape(-1, 1)])
        tableau = np.vstack([top, bottom])
        return tableau
