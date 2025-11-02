# entropy_stock_analysis.py
import yfinance as yf
import pandas as pd
import numpy as np
from math import log2
import matplotlib.pyplot as plt

# ----- PARAMETERS -----
TICKER = '^NSEI'                # default: NIFTY 50 on Yahoo Finance; change to '^GSPC' for S&P500, or 'RELIANCE.NS' etc.
START  = '2008-01-01'
END    = '2025-09-28'           # use inclusive end date
WINDOW = 90                     # rolling window in trading days (30, 90, 252 are standard choices)
BINS   = 3                      # default: 3 bins (DOWN/NEUTRAL/UP) - can use 5 for finer granularity
BIN_METHOD = 'quantile'         # 'quantile' or 'fixed' histogram bins
LOW_ENTROPY_QUANTILE = 0.20     # threshold to consider 'low entropy' regime

# ----- DATA DOWNLOAD -----
df = yf.download(TICKER, start=START, end=END, progress=False, auto_adjust=True)

# Handle multi-level columns if present
if isinstance(df.columns, pd.MultiIndex):
    # Flatten the column names
    df.columns = [col[0] for col in df.columns]

# Select and rename columns - use whatever is available
if 'Adj Close' in df.columns:
    df = df[['Adj Close','Volume']].rename(columns={'Adj Close':'adj_close'})
elif 'Close' in df.columns:
    df = df[['Close','Volume']].rename(columns={'Close':'adj_close'})
else:
    # Fall back to first price column available
    price_cols = [col for col in df.columns if col in ['Close', 'Adj Close', 'close', 'adj_close']]
    if price_cols:
        df = df[[price_cols[0],'Volume']].rename(columns={price_cols[0]:'adj_close'})
    else:
        raise ValueError(f"No suitable price column found. Available columns: {df.columns.tolist()}")

df.dropna(inplace=True)

# ----- PREPROCESS: returns -----
df['log_ret'] = np.log(df['adj_close'] / df['adj_close'].shift(1))
df['ret'] = df['adj_close'].pct_change()
df.dropna(inplace=True)

# ----- HELPER: Shannon entropy from counts/probs -----
def shannon_entropy_from_probs(p):
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def entropy_from_array(arr, bins=BINS, method='quantile'):
    arr = np.asarray(arr)
    if len(arr) == 0:
        return np.nan
    if method == 'quantile':
        # equal-frequency bins
        edges = np.quantile(arr, np.linspace(0,1,bins+1))
        edges = np.unique(edges)  # handle flat windows
        if len(edges) <= 1:
            return 0.0
        counts, _ = np.histogram(arr, bins=edges)
    else:
        counts, _ = np.histogram(arr, bins=bins)
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    return shannon_entropy_from_probs(probs)

# ----- ROLLING ENTROPY -----
# use rolling().apply with a small wrapper
def rolling_entropy(series, window=WINDOW, bins=BINS, method=BIN_METHOD):
    return series.rolling(window=window).apply(lambda x: entropy_from_array(x, bins=bins, method=method), raw=False)

df['entropy'] = rolling_entropy(df['log_ret'], window=WINDOW, bins=BINS, method=BIN_METHOD)
# normalized to [0,1] by dividing by max entropy log2(BINS)
max_entropy = log2(BINS)
df['entropy_norm'] = df['entropy'] / max_entropy

# ----- ROLLING VOLATILITY (annualized) -----
df['vol_rolling'] = df['log_ret'].rolling(WINDOW).std() * np.sqrt(252)

# ----- QUICK CORRELATION & VISUAL -----
corr = df[['entropy_norm','vol_rolling']].dropna().corr().iloc[0,1]
print(f"Correlation between {WINDOW}-day normalized entropy and annualized vol: {corr:.3f}")

# Optional visualization
fig, axes = plt.subplots(2, 1, figsize=(12,6))
data_to_plot = df[['entropy_norm','vol_rolling']].dropna()

data_to_plot['entropy_norm'].plot(ax=axes[0], title=f'{TICKER} Normalized Entropy', color='blue')
axes[0].set_ylabel('Normalized Entropy')

data_to_plot['vol_rolling'].plot(ax=axes[1], title=f'{TICKER} Rolling Volatility', color='red')
axes[1].set_ylabel('Annualized Volatility')

plt.tight_layout()
plt.savefig('entropy_volatility_analysis.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to prevent display issues
print("Saved visualization to entropy_volatility_analysis.png")

# ----- SIMPLE LOW-ENTROPY STRATEGY (toy example) -----
# Define low-entropy threshold from full sample distribution
threshold = df['entropy_norm'].quantile(LOW_ENTROPY_QUANTILE)
df['low_entropy'] = (df['entropy_norm'] < threshold).astype(int)

# Build a simple signal: when in low-entropy, predict the sign of mean past returns
past_mean = df['log_ret'].rolling(WINDOW).mean().shift(1)          # use only past info
df['signal'] = 0
mask = df['low_entropy'] == 1
df.loc[mask, 'signal'] = np.sign(past_mean[mask])
df['signal'] = df['signal'].fillna(0)

# compute strategy returns (use simple returns)
df['strategy_ret'] = df['signal'].shift(1) * df['ret']            # shift to avoid look-ahead
df['bench_ret'] = df['ret']

# apply small transaction cost per trade if desired (example 0.01% per turnover)
tc = 0.0001
turns = (df['signal'].shift(1) != df['signal'].shift(2)).astype(int)  # trade when signal changes
df['strategy_ret_net'] = df['strategy_ret'] - turns * tc

# performance metrics
def cum_return(series):
    cumulative = (1 + series.dropna()).cumprod()
    return cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0

def annualized_return(series, periods_per_year=252):
    total_ret = cum_return(series)
    days = series.dropna().shape[0]
    return (1 + total_ret) ** (periods_per_year / days) - 1 if days>0 else np.nan

def sharpe(series, rf=0.0, periods_per_year=252):
    ann_ret = annualized_return(series, periods_per_year)
    ann_vol = series.std() * np.sqrt(periods_per_year)
    return (ann_ret - rf) / ann_vol if ann_vol != 0 else np.nan

# Performance summary
print(f"\n{'='*60}")
print(f"SHANNON ENTROPY STOCK ANALYSIS RESULTS ({TICKER})")
print(f"Period: {START} to {END}")
print(f"Window: {WINDOW} days, Bins: {BINS}, Method: {BIN_METHOD}")
print(f"{'='*60}")

strat_return = cum_return(df['strategy_ret_net'])
bench_return = cum_return(df['bench_ret'])
strat_sharpe = sharpe(df['strategy_ret_net'])

print(f"Strategy cumulative return (net): {strat_return:.4f} ({strat_return*100:.2f}%)")
print(f"Benchmark cumulative return:      {bench_return:.4f} ({bench_return*100:.2f}%)")
print(f"Strategy annualized Sharpe (net): {strat_sharpe:.4f}")

# Additional statistics
valid_entropy = df['entropy_norm'].dropna()
print(f"\nEntropy Statistics:")
print(f"Mean normalized entropy: {valid_entropy.mean():.4f}")
print(f"Std normalized entropy:  {valid_entropy.std():.4f}")
print(f"Low entropy threshold:   {threshold:.4f} ({LOW_ENTROPY_QUANTILE*100:.0f}th percentile)")

low_entropy_days = (df['low_entropy'] == 1).sum()
total_days = len(df)
print(f"Low entropy periods:     {low_entropy_days}/{total_days} days ({low_entropy_days/total_days*100:.1f}%)")

# ----- SAVE OUTPUT -----
df.to_csv('entropy_analysis_output.csv')
print("Saved results to entropy_analysis_output.csv")
