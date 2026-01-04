# ------------------------------------------------------------
# Cliff Walking (4×12) — SARSA vs Q-Learning 
#  • Falling into the cliff ENDS the episode (done=True).
#  • Defaults: ε=0.1, α=0.5, γ=1.0, EPISODES=500, RUNS=10.
# Outputs (PNG): sarsa_cliff_returns.png, qlearning_cliff_returns.png,
#                cliff_returns_compare.png
# ------------------------------------------------------------
# pip install numpy matplotlib
# ------------------------------------------------------------

import numpy as np
from math import ceil
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ============== Environment (Cliff 4x12) ==============
N_ROWS, N_COLS = 4, 12
START = (3, 0)
GOAL  = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}  # cells (3,1)...(3,10)

ACTIONS = ['U', 'D', 'L', 'R']
A2DELTA = {'U': (-1, 0), 'D': ( 1, 0), 'L': (0, -1), 'R': (0,  1)}

def in_bounds(r, c):
    return 0 <= r < N_ROWS and 0 <= c < N_COLS

def step(state, action):
    """
    Deterministic dynamics.
    Rewards: -1 per move; -100 if stepping into the cliff (episode TERMINATES);
             reaching GOAL gives -1 and ends the episode.
    """
    if state == GOAL:
        return state, 0, True

    dr, dc = A2DELTA[action]
    nr, nc = state[0] + dr, state[1] + dc
    if not in_bounds(nr, nc):
        nr, nc = state  # bump into wall -> stay

    ns = (nr, nc)
    if ns in CLIFF:            # fall off the cliff -> terminal
        return ns, -100, True
    if ns == GOAL:
        return ns, -1, True
    return ns, -1, False

# ============== Helpers ==============
def s_index(s):  # (r,c) -> integer index
    return s[0] * N_COLS + s[1]

def epsilon_greedy(Q, s_idx, eps, rng):
    """
    ε-greedy with random tie-break among argmax actions
    (closer to Sutton & Barto implementations).
    """
    if rng.random() < eps:
        return rng.randrange(Q.shape[1])
    best = np.max(Q[s_idx])
    candidates = np.flatnonzero(np.isclose(Q[s_idx], best))
    return int(rng.choice(list(candidates)))

def moving_average(x, w=100):
    """Return moving average with 'same' length (centered)."""
    x = np.asarray(x)
    if x.size == 0:
        return x
    w = int(min(w, x.size))
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode='same')

# Try to use Savitzky-Golay from scipy if available; otherwise fall back to moving average
try:
    from scipy.signal import savgol_filter
    _HAS_SAVGOL = True
except Exception:
    _HAS_SAVGOL = False

def plot_mean_std(mean, std, title, outfile, color, ylim=(-100, 10)):
    x = np.arange(len(mean))
    plt.figure(figsize=(7,4))
    plt.plot(x, mean, color=color, label="mean")
    plt.fill_between(x, mean-std, mean+std, color=color, alpha=0.2, label="±1 std")
    plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    if ylim is not None:
        plt.ylim(*ylim)  # focuses view like textbook figure
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards (per episode)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Ensure outfile is saved into the script's directory
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / Path(outfile)
    plt.savefig(str(out_path), dpi=200)
    plt.close()

# ============== Algorithms ==============
def run_sarsa_once(episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = random.Random(seed)
    np.random.seed(seed)
    nS, nA = N_ROWS * N_COLS, len(ACTIONS)
    Q = np.zeros((nS, nA), dtype=float)
    returns = []

    for _ in range(episodes):
        s = START
        s_i = s_index(s)
        a = epsilon_greedy(Q, s_i, eps, rng)
        G = 0.0
        while True:
            ns, r, done = step(s, ACTIONS[a])
            G += r
            ns_i = s_index(ns)
            if done:
                # terminal: no future term
                Q[s_i, a] += alpha * (r - Q[s_i, a])
                break
            a_next = epsilon_greedy(Q, ns_i, eps, rng)
            td_target = r + gamma * Q[ns_i, a_next]
            Q[s_i, a] += alpha * (td_target - Q[s_i, a])
            s, s_i, a = ns, ns_i, a_next
        returns.append(G)

    return np.array(returns)

def run_qlearning_once(episodes=500, alpha=0.5, gamma=1.0, eps=0.1, seed=1):
    rng = random.Random(seed)
    np.random.seed(seed)
    nS, nA = N_ROWS * N_COLS, len(ACTIONS)
    Q = np.zeros((nS, nA), dtype=float)
    returns = []

    for _ in range(episodes):
        s = START
        s_i = s_index(s)
        G = 0.0
        while True:
            a = epsilon_greedy(Q, s_i, eps, rng)
            ns, r, done = step(s, ACTIONS[a])
            G += r
            ns_i = s_index(ns)
            if done:
                Q[s_i, a] += alpha * (r - Q[s_i, a])
                break
            max_next = np.max(Q[ns_i])
            td_target = r + gamma * max_next
            Q[s_i, a] += alpha * (td_target - Q[s_i, a])
            s, s_i = ns, ns_i
        returns.append(G)

    return np.array(returns)

def many_runs(alg_fn, runs=10, episodes=500, **kwargs):
    """Run algorithm multiple times; return mean and std across runs."""
    all_R = []
    for r in range(runs):
        arr = alg_fn(episodes=episodes,
                     seed=kwargs.get("seed_base", 0) + r,
                     alpha=kwargs["alpha"], gamma=kwargs["gamma"], eps=kwargs["eps"])
        all_R.append(arr)
    R = np.stack(all_R, axis=0)  # (runs, episodes)
    return R.mean(axis=0), R.std(axis=0)

# ============== Run & Save Figures ==============
if __name__ == "__main__":
    EPISODES = 500
    RUNS     = 10
    ALPHA    = 0.5
    GAMMA    = 1.0
    EPS      = 0.1

    # SARSA
    sarsa_mean, sarsa_std = many_runs(
        run_sarsa_once, runs=RUNS, episodes=EPISODES,
        alpha=ALPHA, gamma=GAMMA, eps=EPS, seed_base=1000
    )
    # smooth mean and std using Savitzky-Golay when available, else moving average
    ma_w = 31  # smoothing window (odd for savgol), for RUNS=10 we keep a moderate window
    def smooth_series(y):
        y = np.asarray(y)
        if y.size == 0:
            return y
        # Ensure odd window length and <= len(y)
        w = int(min(ma_w, y.size if y.size % 2 == 1 else y.size - 1))
        if w < 3:
            return y
        if _HAS_SAVGOL:
            # polyorder must be less than window length
            poly = min(3, w - 1)
            try:
                return savgol_filter(y, window_length=w, polyorder=poly, mode='interp')
            except Exception:
                return moving_average(y, w=w)
        else:
            return moving_average(y, w=w)

    sarsa_mean_s = smooth_series(sarsa_mean)
    sarsa_std_s  = smooth_series(sarsa_std)
    plot_mean_std(sarsa_mean_s, sarsa_std_s,
                  f"SARSA (ε={EPS}, α={ALPHA}, γ={GAMMA}) — mean ± std over {RUNS} runs",
                  "sarsa_cliff_returns.png", color="#1f77b4", ylim=(-100, 10))

    # Q-Learning
    ql_mean, ql_std = many_runs(
        run_qlearning_once, runs=RUNS, episodes=EPISODES,
        alpha=ALPHA, gamma=GAMMA, eps=EPS, seed_base=2000
    )
    ql_mean_s = smooth_series(ql_mean)
    ql_std_s  = smooth_series(ql_std)
    plot_mean_std(ql_mean_s, ql_std_s,
                  f"Q-Learning (ε={EPS}, α={ALPHA}, γ={GAMMA}) — mean ± std over {RUNS} runs",
                  "qlearning_cliff_returns.png", color="#d62728", ylim=(-100, 10))

    # Combined (means only), 
    x = np.arange(EPISODES)
    plt.figure(figsize=(7,4))
    plt.plot(x, sarsa_mean_s, label="SARSA", color="#1f77b4")
    plt.plot(x, ql_mean_s, label="Q-Learning", color="#d62728")
    plt.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    plt.ylim(-100, 10)
    plt.title(f"Cliff Walking: SARSA vs Q-Learning (mean over {RUNS} runs)")
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards (per episode)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # Save combined comparison into script folder
    script_dir = Path(__file__).resolve().parent
    compare_path = script_dir / "cliff_returns_compare.png"
    plt.savefig(str(compare_path), dpi=200)
    plt.close()

    # Print saved file locations so the user can see them
    print("Saved:")
    print(f" - {script_dir / 'sarsa_cliff_returns.png'}")
    print(f" - {script_dir / 'qlearning_cliff_returns.png'}")
    print(f" - {compare_path}")
