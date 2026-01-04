# ------------------------------------------------------------
# Mars Rover MDP — SARSA (On-Policy) with Learning Curve
# ------------------------------------------------------------
# Requirements: numpy, matplotlib
#   pip install numpy matplotlib
# ------------------------------------------------------------

import numpy as np
import random
import math
import csv
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# MDP Definition (finalized)
# =========================
GAMMA = 0.95

STATES = ["Start", "SiteA", "SiteB", "Base", "Destroyed", "Immobile"]
ACTIONS = {
    "Start": ["left", "right"],
    "SiteA": ["left", "right"],
    "SiteB": ["left", "right"],
}
TERMINALS = {"Base", "Destroyed", "Immobile"}

# Transition model for sampling: P[(s,a)] = [(p, s_next, reward, done), ...]
P = {
    # From Start
    ("Start", "left"):  [(0.9, "SiteA", -1, False),
                         (0.1, "Immobile", -3, True)],
    ("Start", "right"): [(0.5, "Base", +10, True),
                         (0.5, "Destroyed", -10, True)],
    # From SiteA
    ("SiteA", "left"):  [(1.0, "Start", -1, False)],
    ("SiteA", "right"): [(0.8, "SiteB", -1, False),
                         (0.2, "Immobile", -3, True)],
    # From SiteB
    ("SiteB", "left"):  [(1.0, "SiteA", -1, False)],
    ("SiteB", "right"): [(1.0, "Base", +10, True)],
}

def is_terminal(s: str) -> bool:
    return s in TERMINALS

def step_env(s: str, a: str, rng: random.Random):
    """Sample (s', r, done) according to P(s'|s,a)."""
    trans = P[(s, a)]
    r = rng.random()
    cum = 0.0
    for p, s_next, rew, done in trans:
        cum += p
        if r <= cum:
            return s_next, rew, done
    # numeric guard (shouldn't happen if probs sum to 1)
    return trans[-1][1], trans[-1][2], trans[-1][3]

def epsilon_greedy(Q, s: str, eps: float, rng: random.Random):
    if is_terminal(s):
        return None
    if rng.random() < eps:
        return rng.choice(ACTIONS[s])
    # greedy with stable tie-break
    vals = [(Q[(s, a)], a) for a in ACTIONS[s]]
    return max(vals, key=lambda t: (t[0], t[1]))[1]

def sarsa_train(
    num_episodes=20000,
    alpha=0.1,
    gamma=GAMMA,
    eps_start=0.1,
    eps_min=0.01,
    eps_decay=0.9995,
    seed=0,
    max_steps_per_episode=1000,
    log_returns=True,
):
    rng = random.Random(seed)
    np.random.seed(seed)

    # Initialize Q(s,a) = 0
    Q = {(s, a): 0.0 for s in ACTIONS for a in ACTIONS[s]}

    episode_returns = []

    eps = eps_start
    for ep in range(num_episodes):
        s = "Start"
        a = epsilon_greedy(Q, s, eps, rng)
        G = 0.0  # total return for this episode

        for t in range(max_steps_per_episode):
            s_next, r, done = step_env(s, a, rng)
            G += r

            if done:
                td_target = r
                Q[(s, a)] += alpha * (td_target - Q[(s, a)])
                break

            a_next = epsilon_greedy(Q, s_next, eps, rng)
            td_target = r + gamma * Q[(s_next, a_next)]
            Q[(s, a)] += alpha * (td_target - Q[(s, a)])

            s, a = s_next, a_next

        if log_returns:
            episode_returns.append(G)

        # epsilon decay
        eps = max(eps_min, eps * eps_decay)

    # Derive greedy policy and V(s) from Q
    V = {}
    pi = {}
    for s in STATES:
        if is_terminal(s):
            V[s] = 0.0
            pi[s] = None
        elif s in ACTIONS:
            best_q, best_a = max(((Q[(s, a)], a) for a in ACTIONS[s]), key=lambda t: (t[0], t[1]))
            V[s] = best_q
            pi[s] = best_a
        else:
            V[s] = 0.0
            pi[s] = None

    return Q, V, pi, episode_returns

def moving_average(x, w=100):
    if len(x) == 0:
        return np.array([])
    w = min(w, len(x))
    return np.convolve(x, np.ones(w)/w, mode="valid")

if __name__ == "__main__":
    # ---- Train SARSA ----
    Q, V, pi, ep_returns = sarsa_train(
        num_episodes=20000,
        alpha=0.1,
        gamma=GAMMA,
        eps_start=0.1,
        eps_min=0.01,
        eps_decay=0.9995,
        seed=42,
        max_steps_per_episode=1000,
        log_returns=True,
    )

    # ---- Print learned values and policy (focus states) ----
    focus = ["Start", "SiteA", "SiteB"]
    print("=== SARSA (learned greedy policy and V from Q) ===")
    for s in focus:
        print(f"{s:7s}: V≈{V[s]:7.3f}   pi={pi[s]}")

    # ---- Save learning curve ----
    plt.figure(figsize=(7,4))
    plt.plot(ep_returns, alpha=0.35, label="Episode return")
    ma = moving_average(ep_returns, w=200)
    if len(ma) > 0:
        x_ma = np.arange(len(ma)) + 200 - 1
        plt.plot(x_ma, ma, linewidth=2, label="Moving average (w=200)")
    plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    plt.title("SARSA Learning Curve (Mars Rover)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # Save into script's project folder
    script_dir = Path(__file__).resolve().parent
    lc_path = script_dir / "sarsa_learning_curve.png"
    plt.savefig(str(lc_path), dpi=200)
    plt.close()

    # ---- Optional: save episode returns to CSV for report ----
    csv_path = script_dir / "sarsa_episode_returns.csv"
    with open(str(csv_path), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return"])
        for i, G in enumerate(ep_returns, start=1):
            writer.writerow([i, G])

    print("\nSaved files:")
    print(f" - {lc_path}")
    print(f" - {csv_path}")
