#!/usr/bin/env python3
"""
Debug script to test yfinance data download
"""

import yfinance as yf
import pandas as pd
import numpy as np

# Test downloading NIFTY data
print("Testing NIFTY data download...")
ticker = '^NSEI'
data = yf.download(ticker, start='2019-01-01', end='2025-09-29', progress=False)
print("Downloaded data columns:", data.columns.tolist())
print("Data shape:", data.shape)
print("First few rows:")
print(data.head())

if not data.empty:
    print("\nAdj Close data available:", 'Adj Close' in data.columns)
    if 'Adj Close' in data.columns:
        print("Adj Close first 5 values:")
        print(data['Adj Close'].head())
    else:
        print("Available columns:", data.columns.tolist())
        if 'Close' in data.columns:
            print("Using Close instead of Adj Close")
            print("Close first 5 values:")
            print(data['Close'].head())