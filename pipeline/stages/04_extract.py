"""Stage 4: Extract solution vector and objective value from final tableau."""

import numpy as np
from typing import Tuple

class SolutionExtractor:
    """Read basic variables from tableau and compute objective value."""

    def __init__(self, tableau: np.ndarray, original_n: int):
        self.tableau = tableau
        self.original_n = original_n
        self.m, self.n_plus_one = tableau.shape
        self.n = self.n_plus_one - 1

    def extract(self) -> Tuple[np.ndarray, float]:
        # Identify basic columns (single 1 with zeros elsewhere) among first original_n vars
        basic_vals = np.zeros(self.original_n)
        for var_idx in range(self.original_n):
            col = self.tableau[1:, var_idx]
            if np.count_nonzero(np.isclose(col, 1.0)) == 1 and np.count_nonzero(np.isclose(col, 0.0)) == self.m - 2:
                row = np.where(np.isclose(col, 1.0))[0][0] + 1
                basic_vals[var_idx] = self.tableau[row, -1]
        objective = self.tableau[0, -1]
        return basic_vals, objective
