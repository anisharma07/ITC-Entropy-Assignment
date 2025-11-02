#!/usr/bin/env python3
"""
Shannon Entropy Analysis for NIFTY vs Top-10 Largecaps
Information Theory and Communication - NIT Hamirpur

Authors: Anirudh Sharma, Anshul Choudhary, Gundra Rohan Reddy, 
         Pushpdeep Singh Chandel, Rishabh Raj
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class StockEntropyAnalyzer:
    def __init__(self):
        """Initialize the analyzer with top 10 Indian largecap stocks."""
        # NIFTY index and top 10 largecap stocks (by market cap as of 2024)
        self.tickers = {
            'NIFTY': '^NSEI',
            'Reliance': 'RELIANCE.NS',
            'TCS': 'TCS.NS', 
            'HDFC Bank': 'HDFCBANK.NS',
            'ICICI Bank': 'ICICIBANK.NS',
            'Bharti Airtel': 'BHARTIARTL.NS',
            'SBI': 'SBIN.NS',
            'LTIMindtree': 'LTIM.NS',
            'Infosys': 'INFY.NS',
            'HUL': 'HINDUNILVR.NS',
            'ITC': 'ITC.NS'
        }
        
        self.data = {}
        self.log_returns = {}
        self.entropy_data = {}
        
    def download_data(self, start_date='2008-01-01', end_date=None):
        """Download stock data using yfinance."""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        print(f"Downloading data from {start_date} to {end_date}...")
        
        for name, ticker in self.tickers.items():
            try:
                print(f"Downloading {name} ({ticker})...")
                stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not stock_data.empty:
                    # Handle multi-level columns from yfinance
                    if isinstance(stock_data.columns, pd.MultiIndex):
                        # Check if Adj Close exists
                        if ('Adj Close', ticker) in stock_data.columns:
                            self.data[name] = stock_data[('Adj Close', ticker)].dropna()
                        elif ('Close', ticker) in stock_data.columns:
                            self.data[name] = stock_data[('Close', ticker)].dropna()
                        else:
                            print(f"✗ {name}: No Close/Adj Close data found")
                            continue
                    else:
                        # Single-level columns
                        if 'Adj Close' in stock_data.columns:
                            self.data[name] = stock_data['Adj Close'].dropna()
                        elif 'Close' in stock_data.columns:
                            self.data[name] = stock_data['Close'].dropna()
                        else:
                            print(f"✗ {name}: No Close/Adj Close data found")
                            continue
                    
                    print(f"✓ {name}: {len(self.data[name])} data points")
                else:
                    print(f"✗ {name}: No data available")
            except Exception as e:
                print(f"✗ Error downloading {name}: {str(e)}")
        
        print(f"\nSuccessfully downloaded data for {len(self.data)} assets")
        
    def compute_log_returns(self):
        """Compute daily log returns for all assets."""
        print("\nComputing log returns...")
        
        for name, prices in self.data.items():
            self.log_returns[name] = np.log(prices / prices.shift(1)).dropna()
            print(f"✓ {name}: {len(self.log_returns[name])} log returns")
            
    def shannon_entropy(self, data, bins=3):
        """
        Calculate Shannon entropy using quantile-based binning.
        
        Parameters:
        - data: pandas Series of log returns
        - bins: number of bins (default: 3 for terciles)
        
        Returns:
        - Shannon entropy value
        """
        if len(data) == 0 or data.isna().all():
            return np.nan
            
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return np.nan
            
        # Create quantile-based bins
        try:
            # Use quantiles to create equal-probability bins
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = clean_data.quantile(quantiles).values
            
            # Handle edge case where all values are the same
            if len(np.unique(bin_edges)) == 1:
                return 0.0
                
            # Discretize data
            digitized = np.digitize(clean_data, bin_edges[1:-1])
            
            # Calculate probabilities
            unique, counts = np.unique(digitized, return_counts=True)
            probabilities = counts / len(clean_data)
            
            # Calculate Shannon entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
            
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return np.nan
    
    def compute_rolling_entropy(self, window=90, bins=3):
        """Compute rolling Shannon entropy for all assets."""
        print(f"\nComputing {window}-day rolling Shannon entropy...")
        
        for name, returns in self.log_returns.items():
            print(f"Processing {name}...")
            
            # Calculate rolling entropy
            rolling_entropy = returns.rolling(window=window, min_periods=window//2).apply(
                lambda x: self.shannon_entropy(x, bins=bins), raw=False
            )
            
            self.entropy_data[name] = rolling_entropy.dropna()
            print(f"✓ {name}: {len(self.entropy_data[name])} entropy values")
    
    def plot_entropy_comparison(self, save_path='entropy_comparison.png'):
        """Create comparison plot of rolling entropy."""
        plt.figure(figsize=(15, 10))
        
        # Plot NIFTY first with thicker line
        if 'NIFTY' in self.entropy_data:
            plt.plot(self.entropy_data['NIFTY'].index, 
                    self.entropy_data['NIFTY'].values, 
                    linewidth=3, label='NIFTY Index', color='red', alpha=0.8)
        
        # Plot individual stocks
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.entropy_data)-1))
        color_idx = 0
        
        for name, entropy in self.entropy_data.items():
            if name != 'NIFTY':
                plt.plot(entropy.index, entropy.values, 
                        linewidth=1.5, label=name, alpha=0.7, color=colors[color_idx])
                color_idx += 1
        
        plt.title('90-Day Rolling Shannon Entropy: NIFTY vs Top-10 Largecaps', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Shannon Entropy (bits)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add annotations for key periods
        plt.axvline(pd.to_datetime('2008-09-15'), color='gray', linestyle='--', alpha=0.5)
        plt.text(pd.to_datetime('2008-09-15'), plt.ylim()[1]*0.9, 'Lehman Crisis', 
                rotation=90, fontsize=10, alpha=0.7)
        
        plt.axvline(pd.to_datetime('2020-03-15'), color='gray', linestyle='--', alpha=0.5)
        plt.text(pd.to_datetime('2020-03-15'), plt.ylim()[1]*0.9, 'COVID-19', 
                rotation=90, fontsize=10, alpha=0.7)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Entropy comparison plot saved as {save_path}")
    
    def generate_statistics_table(self):
        """Generate comprehensive statistics table."""
        stats_data = []
        
        for name, entropy in self.entropy_data.items():
            if len(entropy) > 0:
                stats = {
                    'Asset': name,
                    'Mean Entropy': entropy.mean(),
                    'Std Entropy': entropy.std(),
                    'Min Entropy': entropy.min(),
                    'Max Entropy': entropy.max(),
                    'Median Entropy': entropy.median(),
                    'CV (%)': (entropy.std() / entropy.mean()) * 100,
                    'Data Points': len(entropy)
                }
                stats_data.append(stats)
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values('Mean Entropy', ascending=False)
        
        print("\n" + "="*80)
        print("SHANNON ENTROPY STATISTICS SUMMARY")
        print("="*80)
        print(stats_df.round(4).to_string(index=False))
        print("="*80)
        
        # Save to CSV
        stats_df.to_csv('entropy_statistics.csv', index=False)
        print("✓ Statistics saved to entropy_statistics.csv")
        
        return stats_df
    
    def identify_low_entropy_periods(self, threshold_percentile=10):
        """Identify periods of unusually low entropy."""
        print(f"\nIdentifying low entropy periods (bottom {threshold_percentile}%)...")
        
        low_entropy_periods = {}
        
        for name, entropy in self.entropy_data.items():
            threshold = entropy.quantile(threshold_percentile / 100)
            low_periods = entropy[entropy <= threshold]
            
            if len(low_periods) > 0:
                low_entropy_periods[name] = {
                    'threshold': threshold,
                    'periods': low_periods,
                    'count': len(low_periods),
                    'percentage': (len(low_periods) / len(entropy)) * 100
                }
                
                print(f"{name}: {len(low_periods)} periods below {threshold:.3f}")
        
        return low_entropy_periods
    
    def plot_entropy_distribution(self, save_path='entropy_distribution.png'):
        """Plot distribution of entropy values."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for name, entropy in self.entropy_data.items():
            if plot_idx < len(axes):
                axes[plot_idx].hist(entropy.dropna(), bins=30, alpha=0.7, density=True)
                axes[plot_idx].axvline(entropy.mean(), color='red', linestyle='--', 
                                     label=f'Mean: {entropy.mean():.3f}')
                axes[plot_idx].set_title(f'{name}')
                axes[plot_idx].set_xlabel('Shannon Entropy')
                axes[plot_idx].set_ylabel('Density')
                axes[plot_idx].legend()
                axes[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Distribution of 90-Day Rolling Shannon Entropy', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Entropy distribution plot saved as {save_path}")
    
    def correlation_analysis(self):
        """Analyze correlation between entropy values."""
        # Create DataFrame with all entropy series
        entropy_df = pd.DataFrame(self.entropy_data)
        
        # Calculate correlation matrix
        correlation_matrix = entropy_df.corr()
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, fmt='.3f')
        plt.title('Shannon Entropy Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('entropy_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Entropy correlation heatmap saved as entropy_correlation.png")
        
        return correlation_matrix

def main():
    """Main execution function."""
    print("Shannon Entropy Analysis for Indian Stock Market")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = StockEntropyAnalyzer()
    
    # Download data (last 5 years for faster processing, change to 2008 for full analysis)
    start_date = '2019-01-01'  # Change to '2008-01-01' for full historical analysis
    analyzer.download_data(start_date=start_date)
    
    if len(analyzer.data) == 0:
        print("No data downloaded. Exiting...")
        return
    
    # Compute log returns
    analyzer.compute_log_returns()
    
    # Compute rolling entropy
    analyzer.compute_rolling_entropy(window=90, bins=3)
    
    # Generate visualizations
    analyzer.plot_entropy_comparison()
    analyzer.plot_entropy_distribution()
    
    # Generate statistics
    stats_df = analyzer.generate_statistics_table()
    
    # Identify low entropy periods
    low_entropy = analyzer.identify_low_entropy_periods()
    
    # Correlation analysis
    correlation_matrix = analyzer.correlation_analysis()
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    nifty_mean = stats_df[stats_df['Asset'] == 'NIFTY']['Mean Entropy'].iloc[0]
    stocks_below_nifty = stats_df[(stats_df['Asset'] != 'NIFTY') & 
                                 (stats_df['Mean Entropy'] < nifty_mean)]
    
    print(f"NIFTY average entropy: {nifty_mean:.4f}")
    print(f"Stocks with lower entropy than NIFTY: {len(stocks_below_nifty)}/10")
    print(f"This suggests NIFTY exhibits higher entropy (more randomness) than {len(stocks_below_nifty)} individual stocks.")
    
    if len(stocks_below_nifty) >= 7:
        print("\n✓ HYPOTHESIS CONFIRMED: Index-level returns are closer to random walk")
    else:
        print("\n✗ HYPOTHESIS NEEDS REVISION: Mixed results observed")
    
    print("\nAnalysis complete! Check generated files:")
    print("- entropy_comparison.png")
    print("- entropy_distribution.png") 
    print("- entropy_correlation.png")
    print("- entropy_statistics.csv")

if __name__ == "__main__":
    main()