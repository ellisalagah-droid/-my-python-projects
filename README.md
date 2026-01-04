# Project 1 — Chaotic Time-Series Forecasting

Short one-line summary:
Comparative evaluation of supervised models (Linear, Ridge, MLP, RBF‑KRR, GPR) for short‑ and long‑term forecasting of chaotic time series (Lorenz & Ikeda) under varying noise levels.

Overview
This project experiments with multiple regression models to forecast chaotic time-series data produced by the Lorenz and Ikeda systems. Experiments include short-term (L = 1) and longer-term (L = 10) forecasts, and three noise settings (clean, 10 dB, 3 dB). Performance is measured with MSE, MAE, and R². The notebook / scripts run the data generation, model training, evaluation, and produce result tables/plots.

Contents
- data/ (optional) — any input data or saved simulated series
- notebooks/ or scripts/ — Jupyter notebooks or Python scripts that run experiments
- results/ — saved model outputs, metrics, or plots (optional)
- requirements.txt — Python dependencies (recommended)
- README.md — this file

Key dependencies 
- Python 3.8+
- numpy
- scipy
- scikit-learn
- matplotlib
- pandas 
- joblib (optional, for model save/load)

requirements.txt 
numpy
scipy
scikit-learn
matplotlib
pandas
joblib

Quick setup 
1. Create and activate a virtual environment
   - macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate
   - Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

2. Install dependencies
   pip install -r requirements.txt

How to run (examples)
- If there is a Jupyter notebook:
  1. start Jupyter:
     jupyter notebook
  2. open the notebook (e.g., notebooks/chaotic_forecasting.ipynb) and run the cells.

- If there are scripts:
  - Generate data and run experiments:
    python scripts/run_experiments.py --dataset lorenz --horizon 1 --noise clean
  - Example to run all experiments (adjust names to match scripts):
    python scripts/run_experiments.py --all

Notes on experiments & outputs
- Models used:
  - Linear Regression (sklearn.linear_model.LinearRegression)
  - Ridge Regression (sklearn.linear_model.Ridge)
  - MLP (sklearn.neural_network.MLPRegressor)
  - Kernel Ridge with RBF kernel (sklearn.kernel_ridge.KernelRidge with kernel='rbf')
  - Gaussian Process Regression (sklearn.gaussian_process.GaussianProcessRegressor)

- Metrics saved in results/ are MSE, MAE, and R², aggregated across noise settings and horizons.

- Expected runtime: GPR can be slow on large datasets; consider downsampling for quick experiments.

What I built / My role
Solo — implemented data generation for Lorenz and Ikeda systems, implemented multiple forecasting pipelines, evaluated results, and produced plots and short writeups comparing methods.

Key takeaway
Simple linear methods perform well on clean short-horizon data, while nonlinear models (MLP, RBF-KRR) are more robust to noisy signals and longer horizon forecasting. GPR performs well on clean data but is noticeably slower and more sensitive to noise.

contact
- Contact: github.com/ellisalagah-droid

# Project 2 — Finite-horizon LQOC with Riccati Recursion

Short one-line summary:
Finite-horizon linear‑quadratic optimal control using Riccati recursion to compute optimal trajectories and feedback gains.

Overview
This project implements a finite-horizon Linear Quadratic Optimal Control (LQOC) solver using dynamic programming and the Riccati recursion. The code computes the cost-to-go matrices P_k, the optimal feedback gains K_k, the optimal control inputs u_k, and the resulting optimal state trajectories x_k for a given discrete-time linear system and finite horizon.

Contents
- scripts/ or notebooks/ — scripts or notebooks that run the Riccati recursion and simulate trajectories
- examples/ — example system matrices (A, B, Q, R), initial conditions and run-configs
- results/ — saved trajectories, K matrices, and plots
- requirements.txt — Python dependencies 
- README.md — this file

Key dependencies 
- Python 3.8+
- numpy
- scipy
- matplotlib
- (optional) control or custom helper functions for LTI systems

requirements.txt
numpy
scipy
matplotlib

Quick setup
1. Create and activate a virtual environment
   - macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate
   - Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

2. Install dependencies
   pip install -r requirements.txt

How to run 
- If you have a notebook:
  1. Start Jupyter:
     jupyter notebook
  2. Open the notebook and run the cells.
 

What this project computes
- Backwards Riccati recursion to compute P_k for k = N, N-1, ..., 0
- Feedback gains K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A
- Closed-loop trajectories using u_k = -K_k x_k
- Plots of state trajectories and control inputs over the horizon

Notes & expected outputs
- The method supports smooth control actions and produces optimal state trajectories for given quadratic costs.
- The scripts save K_k matrices and trajectories to `results/` (if enabled).
- Typical plots: states vs time, control inputs vs time, and optionally cost-to-go evolution.

What I built / My role
Solo — implemented dynamic programming solver (Riccati recursion), simulation of closed-loop system, and visualization scripts.

- Contact: github.com/ellisalagah-droid

- # Project 3 — Mars Rover Navigation: DP & RL

Short one-line summary:
Mars Rover navigation using Dynamic Programming (Value / Policy Iteration) and Reinforcement Learning (SARSA, Q‑learning); all converge to the same optimal policy with small numerical differences.

Overview
This project compares Dynamic Programming (Value Iteration, Policy Iteration) and Reinforcement Learning (SARSA, Q‑learning) algorithms on a grid-based Mars Rover navigation task. The rover must reach its base while minimizing penalties and avoiding hazardous states. The code computes optimal policies and value functions (DP) and learns policies via interaction (RL). Results show that all four algorithms converge to the same optimal policy, with SARSA and Q‑learning showing slight numerical differences due to exploration and estimation bias.

Contents
- env/ or src/ — environment code (state representation, transition dynamics, reward function)
- algorithms/ — implementations of value iteration, policy iteration, SARSA, and Q-learning
- notebooks/ or scripts/ — example runs, training loops, and evaluation scripts
- results/ — saved policies, value maps, and plots (optional)
- requirements.txt — Python dependencies (recommended)
- README.md — this file

 dependencies
- Python 3.8+
- numpy
- matplotlib
- (optional) tqdm for progress bars

requirements.txt
numpy
matplotlib
tqdm

Quick setup
1. Create and activate a virtual environment:
   - macOS / Linux:
     python3 -m venv venv
     source venv/bin/activate
   - Windows (PowerShell):
     python -m venv venv
     .\venv\Scripts\Activate.ps1

2. Install dependencies:
   pip install -r requirements.txt

How to run 
- Run Value Iteration:
  python scripts/run_dp.py --algo value_iteration --env 

- Run Policy Iteration:
  python scripts/run_dp.py --algo policy_iteration --env 

- Train RL agent (SARSA or Q-learning):
  python scripts/run_rl.py --algo sarsa --episodes 5000 --env 
  python scripts/run_rl.py --algo q_learning --episodes 5000 --env 

- Evaluate or visualize a saved policy:
  python scripts/evaluate_policy.py --policy results/policy_valueiter.pkl --env 

What this project computes
- DP: value functions and deterministic optimal policies via Value Iteration & Policy Iteration.
- RL: learned action-value estimates (Q(s,a)) and learned policies via SARSA and Q‑learning.
- Visualizations: state-value heatmaps, action arrows, and sample trajectories.

Notes & expected outputs
- All four algorithms converge to the same structural optimal policy in these examples, but SARSA and Q-learning produce slightly different state-value estimates because of stochastic exploration and sample-based learning.
- DP methods require a complete model of the environment. RL methods learn via interaction.
- Typical output files: `results/value_map.png`, `results/policy_arrow_map.png`, and `results/learning_curve.png`.

What I built / My role
Solo — implemented the environment simulator, DP solvers (value & policy iteration), RL algorithms (SARSA & Q-learning), evaluation scripts, and plotting utilities.

contact

- Contact: github.com/ellisalagah-droid
