"""Stage 1: Parse raw LP data into standard form matrices.

Converts user‑provided objective vector `c`, constraint matrix `A` and RHS `b`
into NumPy arrays and validates dimensions. Also adds slack variables to transform
inequalities into equalities.
"""

from typing import Tuple, List
import numpy as np

class Parser:
    """Parse inputs and produce canonical matrices for the simplex tableau.

    Parameters
    ----------
    c : List[float]
        Coefficients of the objective function (to be maximized).
    A : List[List[float]]
        Constraint matrix where each row represents a <= inequality.
    b : List[float]
        Right‑hand side constants for each constraint.
    """

    def __init__(self, c: List[float], A: List[List[float]], b: List[float]):
        self.c = np.asarray(c, dtype=float)
        self.A = np.asarray(A, dtype=float)
        self.b = np.asarray(b, dtype=float)
        self._validate()

    def _validate(self) -> None:
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError('Number of constraints and length of b must match')
        if self.A.shape[1] != self.c.shape[0]:
            raise ValueError('Number of variables in A must match length of c')
        if np.any(self.b < 0):
            raise ValueError('All b entries must be non‑negative for standard form')

    def to_standard_form(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (c_std, A_std, b_std) with slack variables added.

        Returns
        -------
        c_std : np.ndarray
            Objective coefficients extended with zeros for slack variables.
        A_std : np.ndarray
            Constraint matrix augmented with identity for slack variables.
        b_std : np.ndarray
            Right‑hand side vector (unchanged).
        """
        m, n = self.A.shape
        # Slack variables: one per constraint
        slack = np.eye(m)
        A_std = np.hstack([self.A, slack])
        c_std = np.concatenate([self.c, np.zeros(m)])
        return c_std, A_std, self.b.copy()
