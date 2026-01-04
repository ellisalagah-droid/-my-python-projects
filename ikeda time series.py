import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Ikeda time series generator

def ikeda_series(u=0.918, N=3000, discard=100):
    x, y = 0.1, 0.1
    series = []
    for n in range(N+discard):
        t = 0.4 - 6/(1 + x**2 + y**2)
        x_new = 1 + u*(x*np.cos(t) - y*np.sin(t))
        y_new = u*(x*np.sin(t) + y*np.cos(t))
        x, y = x_new, y_new
        if n >= discard:  # discard transient
            series.append(x)
    return np.array(series)


# Noise utility
def add_noise_snr(signal, snr_db):
    if np.isinf(snr_db):
        return signal.copy()
    power_signal = np.mean(signal**2)
    snr_linear = 10**(snr_db/10.0)
    power_noise = power_signal / snr_linear
    noise = np.random.normal(scale=np.sqrt(power_noise), size=signal.shape)
    return signal + noise


# Dataset builder

def make_dataset(series, D=10, L=1):
    X, y = [], []
    N = len(series)
    for n in range(N - (D + L - 1)):
        X.append(series[n:n+D])
        y.append(series[n + D - 1 + L])
    return np.array(X), np.array(y)


# Run experiments
def run_experiments(series, snr_levels, horizons, D=10):
    results = []
    for snr in snr_levels:
        noisy_series = add_noise_snr(series, snr)
        for L in horizons:
            X, y = make_dataset(noisy_series, D=D, L=L)
            split = int(0.7*len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            scaler = StandardScaler().fit(X_train)
            X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

            models = {
                "Linear_MLE": LinearRegression(),
                "Ridge_MAP": Ridge(alpha=1.0),
                "MLP": MLPRegressor(hidden_layer_sizes=(20,), max_iter=200,
                                    early_stopping=True, random_state=0),
                "RBF_KRR": KernelRidge(kernel="rbf", gamma=0.1),
                "GPR_sub": GaussianProcessRegressor(
                    kernel=RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-3),
                    alpha=1e-3, random_state=0)
            }

            for name, model in models.items():
                start = time.time()
                model.fit(X_train_s, y_train)
                y_pred = model.predict(X_test_s)
                elapsed = time.time() - start

                results.append({
                    "SNR": "clean" if np.isinf(snr) else f"{snr} dB",
                    "L": L,
                    "Model": name,
                    "MSE": mean_squared_error(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R2": r2_score(y_test, y_pred),
                    "Time (s)": elapsed
                })
    return pd.DataFrame(results)

# Main script

if __name__ == "__main__":
    # Generate Ikeda series
    series_ikeda = ikeda_series(N=3000)

    # Define experiment settings
    snr_levels = [np.inf, 10, 3]  # clean, moderate, heavy noise
    horizons = [1, 10]            # short vs long term

    # Run experiments
    df_results = run_experiments(series_ikeda, snr_levels, horizons, D=10)

    # Save results
    df_results.to_csv("ikeda_results.csv", index=False)
    print("Saved results to ikeda_results.csv")
    print(df_results.head())




    import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Helper to add noise

def add_noise_snr(signal, snr_db):
    if np.isinf(snr_db):
        return signal.copy()
    power_signal = np.mean(signal**2)
    snr_linear = 10**(snr_db/10.0)
    power_noise = power_signal / snr_linear
    noise = np.random.normal(scale=np.sqrt(power_noise), size=signal.shape)
    return signal + noise


# Make dataset for supervised learning

def make_dataset(series, D=10, L=1):
    X, y = [], []
    for n in range(len(series) - (D + L - 1)):
        X.append(series[n:n+D])
        y.append(series[n+D-1+L])
    return np.array(X), np.array(y)


# Plotting helper

def plot_predictions(series, snr_db, L, model, model_name, filename):
    # Add noise
    noisy_series = add_noise_snr(series, snr_db)
    
    # Build dataset
    D = 10
    X, y = make_dataset(noisy_series, D, L)
    split = int(0.7*len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Scale inputs
    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    # Fit and predict
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(y_test[:200], label="True", linewidth=2)
    plt.plot(y_pred[:200], label="Predicted", linestyle="--")
    title = f"Ikeda - {model_name} - {'Clean' if np.isinf(snr_db) else f'{snr_db} dB'} - L={L}"
    plt.title(title)
    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved {filename}")

# ----------------------------
# Example usage
# ----------------------------
from sklearn.linear_model import LinearRegression

# Assuming you already have Ikeda series
series_ikeda = np.loadtxt("ikeda_series.csv")  # or regenerate with your function

# Use a simple model for visualization (e.g., Linear Regression)
model = LinearRegression()

# Generate 4 plots
plot_predictions(series_ikeda, np.inf, 1, model, "Linear", "ikeda_clean_L1.png")
plot_predictions(series_ikeda, 3, 1, model, "Linear", "ikeda_noisy_L1.png")
plot_predictions(series_ikeda, np.inf, 10, model, "Linear", "ikeda_clean_L10.png")
plot_predictions(series_ikeda, 3, 10, model, "Linear", "ikeda_noisy_L10.png")