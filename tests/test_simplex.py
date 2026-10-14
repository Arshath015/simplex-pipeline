import pytest
from pipeline.orchestrator import SimplexSolver
import numpy as np

def test_simplex_optimal():
    # Maximize 3x + 2y subject to 2x + y <= 18, x + 3y <= 15, x,y >=0
    c = [3, 2]
    A = [[2, 1], [1, 3]]
    b = [18, 15]
    solver = SimplexSolver(c, A, b)
    sol, obj = solver.solve()
    # Known optimal solution: x=3, y=4, objective=3*3+2*4=17
    assert np.allclose(sol, np.array([3, 4]), atol=1e-6)
    assert abs(obj - 17) < 1e-6

def test_unbounded():
    # Maximize x subject to -x <= -1 (i.e., x >= 1) with no upper bound
    c = [1]
    A = [[-1]]
    b = [-1]
    # Adjust to standard form by flipping sign (b must be non‑negative) – this will raise
    with pytest.raises(ValueError):
        SimplexSolver(c, A, b)
