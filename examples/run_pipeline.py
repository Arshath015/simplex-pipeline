import os
import numpy as np
from pipeline.orchestrator import SimplexSolver
import matplotlib.pyplot as plt

# Example LP: maximize 5x + 4y subject to
# 6x + 4y <= 24
# x + 2y <= 6
# -x + y <= 1
c = [5, 4]
A = [[6, 4], [1, 2], [-1, 1]]
b = [24, 6, 1]

solver = SimplexSolver(c, A, b)
solution, objective = solver.solve()

# Prepare results directory
os.makedirs('results', exist_ok=True)
# Write textual summary
with open('results/output.txt', 'w') as f:
    f.write(f'Optimal variables: {solution}\n')
    f.write(f'Optimal objective value: {objective}\n')

# Optional: plot objective value over iterations (simple mock plot)
iterations = list(range(1, 6))
obj_vals = [0, 10, 15, 18, objective]
plt.figure()
plt.plot(iterations, obj_vals, marker='o')
plt.title('Objective value progression (mock)')
plt.xlabel('Iteration')
plt.ylabel('Objective')
plt.grid(True)
plt.savefig('results/progression.png')
print('Example run completed, results stored in results/')
