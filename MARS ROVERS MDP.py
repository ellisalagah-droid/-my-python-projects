# ------------------------------------------------------------
# Mars Rover MDP — Value Iteration & Policy Iteration (save plots)
# ------------------------------------------------------------
# Requirements: numpy, matplotlib
#   pip install numpy matplotlib
# ------------------------------------------------------------

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend so we can save figures reliably
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# MDP Definition (finalized)
# =========================
gamma = 0.95

STATES = ["Start", "SiteA", "SiteB", "Base", "Destroyed", "Immobile"]
ACTIONS = {
    "Start": ["left", "right"],
    "SiteA": ["left", "right"],
    "SiteB": ["left", "right"],
    "Base": [],
    "Destroyed": [],
    "Immobile": [],
}
TERMINALS = {"Base", "Destroyed", "Immobile"}

# Transition model: P[(s, a)] = list of (prob, next_state, reward, done)
P = {}

# From Start
P[("Start", "left")]  = [(0.9, "SiteA", -1, False),
                         (0.1, "Immobile", -3, True)]
P[("Start", "right")] = [(0.5, "Base", +10, True),
                         (0.5, "Destroyed", -10, True)]
# From SiteA
P[("SiteA", "left")]  = [(1.0, "Start", -1, False)]
P[("SiteA", "right")] = [(0.8, "SiteB", -1, False),
                         (0.2, "Immobile", -3, True)]
# From SiteB
P[("SiteB", "left")]  = [(1.0, "SiteA", -1, False)]
P[("SiteB", "right")] = [(1.0, "Base", +10, True)]

def is_terminal(s: str) -> bool:
    return s in TERMINALS

# =================
# Value Iteration
# =================
def value_iteration(theta=1e-8, max_iter=10000, verbose=False):
    """Returns (V, pi, deltas) where deltas[i] = max |V_{i+1} - V_i|."""
    V = {s: 0.0 for s in STATES}
    deltas = []

    for it in range(max_iter):
        delta = 0.0
        for s in STATES:
            if is_terminal(s):
                continue
            v_old = V[s]
            best_q = -np.inf
            for a in ACTIONS[s]:
                q = 0.0
                for (p, s_next, r, done) in P[(s, a)]:
                    q += p * (r + (0.0 if done else gamma * V[s_next]))
                best_q = max(best_q, q)
            V[s] = best_q
            delta = max(delta, abs(v_old - V[s]))
        deltas.append(delta)
        if verbose:
            print(f"[VI] iter={it:4d}  delta={delta:.3e}")
        if delta < theta:
            break

    # Greedy policy extraction from converged V
    pi = {}
    for s in STATES:
        if is_terminal(s):
            pi[s] = None
            continue
        best_a, best_q = None, -np.inf
        for a in ACTIONS[s]:
            q = sum(p * (r + (0.0 if done else gamma * V[s_next]))
                    for (p, s_next, r, done) in P[(s, a)])
            if q > best_q:
                best_q, best_a = q, a
        pi[s] = best_a

    return V, pi, deltas

# =================
# Policy Iteration
# =================
def policy_iteration(theta=1e-10, max_eval_iter=10000, verbose=False):
    """
    Returns (V, pi, changes_per_iter) where changes_per_iter[i] is the number
    of states whose chosen action changed at improvement step i.
    """
    pi = {s: (ACTIONS[s][0] if ACTIONS[s] else None) for s in STATES}
    V = {s: 0.0 for s in STATES}
    changes_per_iter = []

    while True:
        # Policy Evaluation (iterative)
        for _ in range(max_eval_iter):
            delta = 0.0
            for s in STATES:
                if is_terminal(s) or pi[s] is None:
                    continue
                a = pi[s]
                v_new = sum(p * (r + (0.0 if done else gamma * V[s_next]))
                            for (p, s_next, r, done) in P[(s, a)])
                delta = max(delta, abs(v_new - V[s]))
                V[s] = v_new
            if delta < theta:
                break

        # Policy Improvement
        changes = 0
        for s in STATES:
            if is_terminal(s):
                continue
            old_a = pi[s]
            best_a, best_q = None, -np.inf
            for a in ACTIONS[s]:
                q = sum(p * (r + (0.0 if done else gamma * V[s_next]))
                        for (p, s_next, r, done) in P[(s, a)])
                if q > best_q:
                    best_q, best_a = q, a
            pi[s] = best_a
            if old_a != best_a:
                changes += 1
        changes_per_iter.append(changes)

        if verbose:
            print(f"[PI] policy changes this iter: {changes}")
        if changes == 0:
            break

    return V, pi, changes_per_iter

# ===========
# Run & Save
# ===========
if __name__ == "__main__":
    # Run algorithms
    V_vi, pi_vi, deltas_vi = value_iteration(verbose=False)
    V_pi, pi_pi, policy_changes = policy_iteration(verbose=False)

    # Focus states for reporting
    focus = ["Start", "SiteA", "SiteB"]

    print("=== Value Iteration (optimal) ===")
    for s in focus:
        print(f"{s:7s}: V* = {V_vi[s]:8.4f}   pi* = {pi_vi[s]}")

    print("\n=== Policy Iteration (optimal) ===")
    for s in focus:
        print(f"{s:7s}: V* = {V_pi[s]:8.4f}   pi* = {pi_pi[s]}")

    # Determine output directory (script's folder)
    script_dir = Path(__file__).resolve().parent

    # 1) Value Iteration Convergence (max |ΔV|)
    plt.figure(figsize=(6,4))
    plt.plot(deltas_vi, linewidth=2)
    plt.title("Value Iteration Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Max |ΔV|")
    plt.yscale("log")  # log scale clarifies convergence curve
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    vi_path = script_dir / "vi_convergence.png"
    plt.savefig(str(vi_path), dpi=200)
    plt.close()

    # 2) Policy Iteration Convergence (policy changes per iteration)
    plt.figure(figsize=(6,4))
    plt.plot(policy_changes, marker="o", linewidth=2)
    plt.title("Policy Iteration Convergence")
    plt.xlabel("Improvement Step")
    plt.ylabel("# of Policy Changes")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    pi_path = script_dir / "pi_convergence.png"
    plt.savefig(str(pi_path), dpi=200)
    plt.close()

    # 3) Final Optimal Values Comparison (bar chart)
    states_order = ["Start", "SiteA", "SiteB"]
    V_vi_vals = [V_vi[s] for s in states_order]
    V_pi_vals = [V_pi[s] for s in states_order]
    x = np.arange(len(states_order))
    w = 0.38

    plt.figure(figsize=(6.4,4.2))
    plt.bar(x - w/2, V_vi_vals, width=w, label="Value Iteration")
    plt.bar(x + w/2, V_pi_vals, width=w, label="Policy Iteration")
    plt.xticks(x, states_order)
    plt.ylabel("Optimal Value  V*(s)")
    plt.title("Optimal Values: VI vs PI")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    values_path = script_dir / "vi_vs_pi_values.png"
    plt.savefig(str(values_path), dpi=200)
    plt.close()

    print("\nSaved figures:")
    print(f" - {vi_path}")
    print(f" - {pi_path}")
    print(f" - {values_path}")
