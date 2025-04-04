# strategy_montecarlo_benchmark.py

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- Configurable Settings ----
stocks = ["AAPL", "MSFT", "GOOGL"]
strategies = ["SMA", "EMA", "RSI"]
num_simulations = 200
trading_days = 252
num_processes = os.cpu_count() or 4
batch_size = 10  # number of simulations per batch

# ---- Data Fetching ----
raw_data = yf.download(stocks, period="2y", group_by="ticker", auto_adjust=True)
data = pd.concat({ticker: raw_data[ticker]["Close"] for ticker in stocks}, axis=1)
data.columns = stocks
data = data.dropna()

# ---- Strategy Functions ----
def apply_strategy(prices, strategy):
    if strategy == "SMA":
        return SMAIndicator(prices, window=100).sma_indicator()
    elif strategy == "EMA":
        return EMAIndicator(prices, window=100).ema_indicator()
    elif strategy == "RSI":
        return RSIIndicator(prices).rsi()
    else:
        raise ValueError("Unknown strategy")

# ---- Simulation Function ----
def simulate_strategy(args):
    stock, strategy = args
    prices = data[stock].dropna()
    returns = prices.pct_change().dropna()
    mean = returns.mean()
    std = returns.std()

    np.random.seed()
    sim_returns = np.random.normal(mean, std, trading_days)
    cum_returns = np.cumprod(1 + sim_returns)

    signal = apply_strategy(prices, strategy).dropna()
    signal = signal[-trading_days:].bfill()
    signal = signal[:trading_days]  # Ensure same length as simulation
    mask = (signal > signal.mean()).values
    masked_cum_returns = np.where(mask, cum_returns, np.nan)

    return masked_cum_returns

# ---- Batch Simulation ----
def simulate_batch(stock, strategy, batch_size):
    return [simulate_strategy((stock, strategy)) for _ in range(batch_size)]

# ---- Cilk-style helpers ----
def cilk_spawn(pool, func, *args):
    return pool.submit(func, *args)

def cilk_sync(futures):
    results = []
    for f in as_completed(futures):
        results.extend(f.result())
    return results

# ---- Main ----
if __name__ == "__main__":
    all_results = {}
    time_records = {"parallel": [], "sequential": []}

    # Sequential version for benchmark
    start_seq = time.perf_counter()
    for strategy in strategies:
        for stock in stocks:
            for _ in range(num_simulations):
                simulate_strategy((stock, strategy))
    end_seq = time.perf_counter()
    time_records["sequential"].append(end_seq - start_seq)

    # Parallel version using multiprocessing
    start_parallel = time.perf_counter()
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for strategy in strategies:
            args_list = [(stock, strategy, batch_size) for stock in stocks for _ in range(num_simulations // batch_size)]
            futures = [cilk_spawn(executor, simulate_batch, *args) for args in args_list]
            results = cilk_sync(futures)
            results = [r for r in results if r is not None]
            all_results[strategy] = results
    end_parallel = time.perf_counter()
    time_records["parallel"].append(end_parallel - start_parallel)

    # ---- Plot Results ----
    plt.figure(figsize=(10, 6))
    colors = {'SMA': 'green', 'EMA': 'blue', 'RSI': 'red'}

    for strategy, runs in all_results.items():
        runs = [r for r in runs if not np.isnan(r).all()]
        avg_path = np.nanmean(runs, axis=0)
        plt.plot(avg_path, label=strategy, color=colors.get(strategy, None))

    plt.title("Monte Carlo Simulation of Strategies (avg path)")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("strategy_montecarlo_benchmark.png")
    plt.show()

    # ---- Plot Timing Histogram ----
    plt.figure(figsize=(6, 4))
    plt.bar(["Sequential", "Parallel"], [time_records["sequential"][0], time_records["parallel"][0]], color=['orange', 'skyblue'])
    plt.title("Execution Time Comparison")
    plt.ylabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig("benchmark_histogram.png")
    plt.show()
