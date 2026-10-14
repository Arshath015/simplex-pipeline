"""Stage 3: Perform pivot operations until optimality or infeasibility is reached."""

import numpy as np
from typing import Tuple

class PivotEngine:
    """Encapsulate the pivoting logic of the Simplex method.

    Uses Bland's rule for entering and leaving variable selection to avoid cycles.
    """

    def __init__(self, tableau: np.ndarray):
        self.tableau = tableau.copy()
        self.m, self.n_plus_one = self.tableau.shape  # includes RHS column
        self.n = self.n_plus_one - 1  # number of decision+slack vars

    def _choose_entering(self) -> int:
        # Most negative coefficient in objective row (excluding RHS)
        costs = self.tableau[0, :-1]
        candidates = np.where(costs < -1e-12)[0]
        if candidates.size == 0:
            return -1  # optimal
        # Bland's rule: smallest index
        return int(candidates.min())

    def _choose_leaving(self, col: int) -> int:
        column = self.tableau[1:, col]
        rhs = self.tableau[1:, -1]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = np.where(column > 1e-12, rhs / column, np.inf)
        min_ratio = ratios.min()
        if np.isinf(min_ratio):
            return -1  # unbounded
        # Bland's rule for ties
        candidates = np.where(np.isclose(ratios, min_ratio))[0]
        return int(candidates.min() + 1)  # +1 to account for objective row offset

    def _pivot(self, pivot_row: int, pivot_col: int) -> None:
        piv_elem = self.tableau[pivot_row, pivot_col]
        self.tableau[pivot_row, :] /= piv_elem
        for r in range(self.m):
            if r != pivot_row:
                factor = self.tableau[r, pivot_col]
                self.tableau[r, :] -= factor * self.tableau[pivot_row, :]

    def run(self) -> Tuple[np.ndarray, bool]:
        """Execute pivot loop.

        Returns
        -------
        tableau : np.ndarray
            Final tableau after termination.
        unbounded : bool
            True if the LP is unbounded.
        """
        while True:
            entering = self._choose_entering()
            if entering == -1:
                break  # optimal
            leaving = self._choose_leaving(entering)
            if leaving == -1:
                return self.tableau, True  # unbounded
            self._pivot(leaving, entering)
        return self.tableau, False
