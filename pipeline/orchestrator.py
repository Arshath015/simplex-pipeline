"""Orchestrator that wires the simplex pipeline stages together."""

from typing import List, Tuple
import numpy as np

from .stages.01_parse import Parser
from .stages.02_tableau import TableauBuilder
from .stages.03_pivot import PivotEngine
from .stages.04_extract import SolutionExtractor

class SimplexSolver:
    """Public façade exposing a single `solve` method.

    Parameters
    ----------
    c : List[float]
        Objective coefficients.
    A : List[List[float]]
        Constraint matrix (<= inequalities).
    b : List[float]
        RHS vector (must be non‑negative).
    """

    def __init__(self, c: List[float], A: List[List[float]], b: List[float]):
        self.c = c
        self.A = A
        self.b = b

    def solve(self) -> Tuple[np.ndarray, float]:
        # Stage 1: parse
        parser = Parser(self.c, self.A, self.b)
        c_std, A_std, b_std = parser.to_standard_form()
        # Stage 2: tableau
        tableau = TableauBuilder(c_std, A_std, b_std).build()
        # Stage 3: pivot
        final_tableau, unbounded = PivotEngine(tableau).run()
        if unbounded:
            raise RuntimeError('Linear program is unbounded')
        # Stage 4: extract solution
        extractor = SolutionExtractor(final_tableau, original_n=len(self.c))
        solution, objective = extractor.extract()
        return solution, objective
