import numpy as np
import matplotlib.pyplot as plt
import csv
from typing import Tuple, List
import os

import numpy as np

def safe_inverse(M, reg=1e-9):
    """Numerically stable inverse: add tiny regularization if needed."""
    try:
        return np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # add small diagonal regularization and invert
        return np.linalg.inv(M + reg * np.eye(M.shape[0]))

# Step 1: initialize P_N = Q_N at the final step
import numpy as np

# Given system matrices 
A1 = np.array([[1.0, 0.05],
               [-0.05, 0.95]])

B1 = np.array([[0.0],
               [0.05]])

Q1 = np.array([[100.0,   0.0],
               [  0.0, 200.0]])

# R is given as a scalar 400 -> convert to 1x1 matrix for consistency
R1 = np.array([[400.0]])

# Terminal cost Pf (used as P_N or Q_N)
Pf = np.array([[1500.0, -1500.0],
               [-1500.0, 3000.0]])

# Initial state x(0) as column vector
x0 = np.array([0.5596, -0.6387]).reshape(-1, 1)

N = 200  # horizon length

# Initialize storage for P[0..N] and K[0..N-1]
P = [None] * (N + 1)    # P[0], P[1], ..., P[N]
K = [None] * N          # placeholder for gains 

# Set terminal cost
P[N] = Pf.copy()

QN = Pf.copy()

# Basic sanity checks / info
print("Shapes:")
print(" A1:", A1.shape)
print(" B1:", B1.shape)
print(" Q1:", Q1.shape)
print(" R1:", R1.shape)
print(" Pf (P_N):", P[N].shape)
print(" x0:", x0.shape)
print("\nP_N initialized to:\n", P[N])

# Optional symmetry check for Pf (Riccati expects symmetric P)
if not np.allclose(P[N], P[N].T, atol=1e-8):
    print("Warning: P_N is not symmetric (numerical issue). Symmetrizing.")
    P[N] = 0.5 * (P[N] + P[N].T)

# Step 2: Riccati recursion to compute P[k] and K[k] for k = N-1 down to 0

import numpy as np

# example small matrices
A = A1.copy()
B = B1.copy()
Q = Q1.copy()
R = R1.copy()
P_next = Pf.copy()  # pretend this is P_{k+1}

# approach 1: compute K then P
S = R + B.T @ P_next @ B
K = np.linalg.inv(S) @ (B.T @ P_next @ A)
P_viaK = Q + A.T @ P_next @ A - A.T @ P_next @ B @ K

# approach 2: substituted form (same as printed formula)
P_sub = Q + A.T @ P_next @ A - A.T @ P_next @ B @ np.linalg.inv(R + B.T @ P_next @ B) @ (B.T @ P_next @ A)

print("norm difference:", np.linalg.norm(P_viaK - P_sub))

# Step 3: Computing the optimal feedback gains K[k] for each stage

K = [None] * N  # reset list of gains

for k in reversed(range(N)):  # k = N-1 ... 0
    P_next = P[k+1]

    # S = R + B^T P_{k+1} B
    S_mat = R1 + B1.T @ P_next @ B1
    S_inv = safe_inverse(S_mat)

    # Compute feedback gain
    K[k] = S_inv @ (B1.T @ P_next @ A1)

    # Compute Riccati matrix for this step
    P[k] = Q1 + A1.T @ P_next @ A1 - A1.T @ P_next @ B1 @ K[k]

# Example: print first few K gains
print("First 3 feedback gains (K[0], K[1], K[2]):")
for i in range(3):
    print(f"K[{i}] =\n", K[i])

# Step 4: Using the feedback gain to determine the optimal control policy and simulate the system forward 

# Arrays to store states (X) and controls (U)
X = np.zeros((N+1, 2))  # state dimension = 2
U = np.zeros((N, 1))    # control dimension = 1

# Initial state
X[0, :] = x0.ravel()

for k in range(N):
    # Control law: u_k* = -K_k x_k
    uk = -K[k] @ X[k, :].reshape(-1, 1)
    U[k, 0] = uk

    # State update: x_{k+1} = A x_k + B u_k
    X[k+1, :] = (A1 @ X[k, :].reshape(-1, 1) + B1 @ uk).ravel()

# Print sample outputs
print("First 5 states:")
print(X[:5])
print("\nFirst 5 controls:")
print(U[:5])

out_csv = "lqoc_solution_xu.csv"
with open(out_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["k", "x1", "x2", "u"])
    for k in range(N):
        writer.writerow([k, float(X[k, 0]), float(X[k, 1]), float(U[k, 0])])
    # final state row (no control at step N)
    writer.writerow([N, float(X[N, 0]), float(X[N, 1]), ""])
print(f"Saved states and controls to '{out_csv}'")

# Plot state trajectories
plt.figure(figsize=(8, 4))
plt.plot(np.arange(N+1), X[:, 0], label="x1")
plt.plot(np.arange(N+1), X[:, 1], label="x2")
plt.xlabel("k")
plt.ylabel("state")
plt.title("Optimal state trajectory (x_k)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("x_optimal.png", dpi=150)
plt.close()
print("Saved 'x_optimal.png'")

# Plot control input
plt.figure(figsize=(8, 3))
plt.step(np.arange(N), U[:, 0], where='post')
plt.xlabel("k")
plt.ylabel("u")
plt.title("Optimal control input (u_k)")
plt.grid(True)
plt.tight_layout()
plt.savefig("u_optimal.png", dpi=150)
plt.close()
print("Saved 'u_optimal.png'")

# Print a short summary (first few rows)

print("\nFirst 6 state rows (k, x1, x2, u):")
for k in range(min(6, N)):
    print(f"{k:2d}  {X[k,0]: .6f}  {X[k,1]: .6f}  {U[k,0]: .6f}")


import pandas as pd

# Load the CSV we already created
df = pd.read_csv("lqoc_solution_xu.csv")

# Select first 8 rows (or however many you want)
df_small = df.head(8)


print("\nDone.")