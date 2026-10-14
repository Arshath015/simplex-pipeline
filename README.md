# Simplex Method Pipeline  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Python: 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Linear programming solver using a clear, staged pipeline architecture.

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Theoretical Background](#theoretical-background)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Pipeline Benchmark](#pipeline-benchmark)
- [Testing](#testing)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)

## Overview
This repository provides a reusable Simplex method implementation organized as a processing pipeline. Each stage transforms the linear program representation, culminating in an optimal solution or detection of infeasibility.

## Tech Stack
- Python 3.9+
- NumPy (numerical arrays)
- pytest (testing)
- matplotlib (optional for visualising tableau evolution in examples)

## Architecture
```text
pipeline/
├─ stages/
│  ├─ 01_parse.py          # Convert raw LP data into standard form matrices
│  ├─ 02_tableau.py        # Build initial simplex tableau
│  ├─ 03_pivot.py          # Perform pivot operations until optimality
│  └─ 04_extract.py        # Extract solution vector and objective value
├─ orchestrator.py         # Glue stages together, expose `solve` API
└─ benchmark.md            # Performance notes and small table
```

## Theoretical Background
The Simplex algorithm solves linear programs of the form *max c·x* subject to *Ax ≤ b*, *x ≥ 0*. By converting inequalities to equalities with slack variables, the problem is expressed in a canonical tableau. The algorithm iteratively moves from one vertex of the feasible polytope to an adjacent one with a non‑decreasing objective value. Pivot selection follows Bland's rule to guarantee termination, avoiding cycles.

Convergence relies on the finiteness of basic feasible solutions. In degenerate cases, the algorithm may perform pivots that do not improve the objective; Bland's rule selects the smallest index entering and leaving variable, ensuring progress.

## Installation
```bash
git clone https://github.com/yourorg/simplex-pipeline.git
cd simplex-pipeline
pip install -r requirements.txt
```

## Usage
```python
from pipeline.orchestrator import SimplexSolver

# Define LP: maximize 3x + 2y subject to 2x + y <= 18, x + 3y <= 15, x, y >= 0
c = [3, 2]
A = [[2, 1], [1, 3]]
b = [18, 15]

solver = SimplexSolver(c, A, b)
solution, objective = solver.solve()
print('Optimal x:', solution)
print('Optimal value:', objective)
```

## API Reference
- `SimplexSolver(c: List[float], A: List[List[float]], b: List[float])`
  - `c`: objective coefficients
  - `A`: constraint matrix (each row a constraint)
  - `b`: right‑hand side vector
- `solve() -> Tuple[np.ndarray, float]`
  - Returns optimal variable values and objective value.

## Pipeline Benchmark
See `pipeline/benchmark.md` for a short runtime table on sample problems.

## Testing
Run the test suite with:
```bash
pytest -q
```

## Limitations
- Only handles maximization problems in standard form.
- No support for integer programming or alternative optimal solutions.
- Numerical stability limited to double precision; may fail on ill‑conditioned matrices.

## Roadmap
- Add dual simplex support.
- Implement presolve reduction techniques.
- Provide optional logging of tableau evolution.

## License
MIT License
