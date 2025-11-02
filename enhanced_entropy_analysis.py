#!/usr/bin/env python3
"""
Enhanced Shannon Entropy Analysis for NIFTY vs Top-10 Largecaps
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

class EnhancedStockEntropyAnalyzer:
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
        self.volatility_data = {}
        
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
            
    def enhanced_shannon_entropy(self, data, method='adaptive_bins', bins=10):
        """
        Calculate Shannon entropy using multiple binning methods.
        
        Parameters:
        - data: pandas Series of log returns
        - method: binning method ('adaptive_bins', 'fixed_bins', 'std_bins')
        - bins: number of bins or method-specific parameter
        
        Returns:
        - Shannon entropy value
        """
        if len(data) == 0 or data.isna().all():
            return np.nan
            
        # Remove NaN values
        clean_data = data.dropna()
        if len(clean_data) == 0:
            return np.nan
            
        try:
            if method == 'adaptive_bins':
                # Use adaptive binning based on data characteristics
                data_range = clean_data.max() - clean_data.min()
                if data_range == 0:
                    return 0.0
                
                # Create bins based on standard deviations
                mean_val = clean_data.mean()
                std_val = clean_data.std()
                
                if std_val == 0:
                    return 0.0
                
                # Create bins from -3σ to +3σ
                bin_edges = np.linspace(mean_val - 3*std_val, mean_val + 3*std_val, bins + 1)
                
                # Extend edges to include all data
                bin_edges[0] = min(bin_edges[0], clean_data.min() - 1e-10)
                bin_edges[-1] = max(bin_edges[-1], clean_data.max() + 1e-10)
                
            elif method == 'fixed_bins':
                # Fixed width bins across data range
                bin_edges = np.linspace(clean_data.min(), clean_data.max(), bins + 1)
                bin_edges[0] -= 1e-10  # Ensure all data is included
                bin_edges[-1] += 1e-10
                
            elif method == 'quantile_bins':
                # Quantile-based bins (equal frequency)
                quantiles = np.linspace(0, 1, bins + 1)
                bin_edges = clean_data.quantile(quantiles).values
                
                # Handle edge case where quantiles are identical
                if len(np.unique(bin_edges)) <= 2:
                    return 0.0
                    
            # Discretize data
            digitized = np.digitize(clean_data, bin_edges) - 1
            digitized = np.clip(digitized, 0, bins - 1)  # Ensure valid bin indices
            
            # Calculate probabilities
            unique, counts = np.unique(digitized, return_counts=True)
            probabilities = counts / len(clean_data)
            
            # Calculate Shannon entropy (avoid log(0))
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
            
        except Exception as e:
            print(f"Error calculating entropy: {e}")
            return np.nan
    
    def compute_rolling_entropy_multiple_methods(self, window=90):
        """Compute rolling Shannon entropy using multiple methods."""
        print(f"\nComputing {window}-day rolling Shannon entropy with multiple methods...")
        
        methods = ['adaptive_bins', 'fixed_bins', 'quantile_bins']
        
        for method in methods:
            print(f"\n--- Method: {method} ---")
            entropy_method = {}
            
            for name, returns in self.log_returns.items():
                print(f"Processing {name} with {method}...")
                
                # Calculate rolling entropy
                rolling_entropy = returns.rolling(window=window, min_periods=window//2).apply(
                    lambda x: self.enhanced_shannon_entropy(x, method=method, bins=10), raw=False
                )
                
                entropy_method[name] = rolling_entropy.dropna()
                print(f"✓ {name}: {len(entropy_method[name])} entropy values, mean: {entropy_method[name].mean():.4f}")
            
            self.entropy_data[method] = entropy_method
            
            # Quick statistics for this method
            self.print_method_statistics(method, entropy_method)
    
    def compute_volatility(self, window=90):
        """Compute rolling volatility for comparison."""
        print(f"\nComputing {window}-day rolling volatility...")
        
        for name, returns in self.log_returns.items():
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            self.volatility_data[name] = rolling_vol.dropna()
            print(f"✓ {name}: mean volatility = {self.volatility_data[name].mean():.4f}")
    
    def print_method_statistics(self, method, entropy_data):
        """Print quick statistics for a method."""
        stats = []
        for name, entropy in entropy_data.items():
            if len(entropy) > 0:
                stats.append({
                    'Asset': name,
                    'Mean': entropy.mean(),
                    'Std': entropy.std(),
                    'Min': entropy.min(),
                    'Max': entropy.max()
                })
        
        stats_df = pd.DataFrame(stats).sort_values('Mean', ascending=False)
        print(f"\nTop 5 assets by mean entropy ({method}):")
        print(stats_df.head().round(4).to_string(index=False))
    
    def plot_comparative_analysis(self, method='adaptive_bins', save_path=None):
        """Create comprehensive comparative plots."""
        if method not in self.entropy_data:
            print(f"Method {method} not found. Available methods: {list(self.entropy_data.keys())}")
            return
        
        entropy_data = self.entropy_data[method]
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Time series comparison
        ax1 = axes[0, 0]
        
        # Plot NIFTY first with thicker line
        if 'NIFTY' in entropy_data:
            ax1.plot(entropy_data['NIFTY'].index, entropy_data['NIFTY'].values, 
                    linewidth=3, label='NIFTY Index', color='red', alpha=0.8)
        
        # Plot individual stocks
        colors = plt.cm.tab10(np.linspace(0, 1, len(entropy_data)-1))
        color_idx = 0
        
        for name, entropy in entropy_data.items():
            if name != 'NIFTY':
                ax1.plot(entropy.index, entropy.values, 
                        linewidth=1.5, label=name, alpha=0.7, color=colors[color_idx])
                color_idx += 1
        
        ax1.set_title(f'90-Day Rolling Shannon Entropy: NIFTY vs Top-10 Largecaps\n({method})', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Shannon Entropy (bits)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Add crisis markers
        crisis_dates = [
            ('2020-03-15', 'COVID-19'),
            ('2021-04-15', 'COVID Wave 2'),
            ('2022-02-24', 'Russia-Ukraine')
        ]
        
        for date, label in crisis_dates:
            try:
                crisis_date = pd.to_datetime(date)
                if ax1.get_xlim()[0] <= crisis_date.timestamp() <= ax1.get_xlim()[1]:
                    ax1.axvline(crisis_date, color='gray', linestyle='--', alpha=0.5)
                    ax1.text(crisis_date, ax1.get_ylim()[1]*0.95, label, rotation=90, 
                            fontsize=8, alpha=0.7)
            except:
                pass
        
        # 2. Box plot comparison
        ax2 = axes[0, 1]
        entropy_values = []
        labels = []
        
        for name, entropy in entropy_data.items():
            entropy_values.append(entropy.values)
            labels.append(name)
        
        bp = ax2.boxplot(entropy_values, labels=labels, patch_artist=True)
        ax2.set_title('Entropy Distribution Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Shannon Entropy (bits)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color NIFTY box differently
        for i, label in enumerate(labels):
            if label == 'NIFTY':
                bp['boxes'][i].set_facecolor('red')
                bp['boxes'][i].set_alpha(0.7)
        
        # 3. Entropy vs Volatility scatter
        ax3 = axes[1, 0]
        
        for name in entropy_data.keys():
            if name in self.volatility_data:
                # Align data by dates
                entropy_series = entropy_data[name]
                vol_series = self.volatility_data[name]
                
                # Find common dates
                common_dates = entropy_series.index.intersection(vol_series.index)
                if len(common_dates) > 0:
                    entropy_aligned = entropy_series[common_dates]
                    vol_aligned = vol_series[common_dates]
                    
                    color = 'red' if name == 'NIFTY' else 'blue'
                    alpha = 0.8 if name == 'NIFTY' else 0.6
                    size = 60 if name == 'NIFTY' else 40
                    
                    ax3.scatter(vol_aligned, entropy_aligned, 
                              alpha=alpha, label=name, s=size, color=color)
        
        ax3.set_xlabel('Volatility (Annualized)')
        ax3.set_ylabel('Shannon Entropy (bits)')
        ax3.set_title('Entropy vs Volatility Relationship', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics table
        stats_data = []
        for name, entropy in entropy_data.items():
            if len(entropy) > 0:
                stats = [
                    name,
                    f"{entropy.mean():.4f}",
                    f"{entropy.std():.4f}",
                    f"{entropy.min():.4f}",
                    f"{entropy.max():.4f}",
                    f"{(entropy.std() / entropy.mean() * 100):.2f}%"
                ]
                stats_data.append(stats)
        
        # Sort by mean entropy
        stats_data.sort(key=lambda x: float(x[1]), reverse=True)
        
        headers = ['Asset', 'Mean', 'Std', 'Min', 'Max', 'CV(%)']
        
        table = ax4.table(cellText=stats_data, colLabels=headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Color NIFTY row
        for i, row in enumerate(stats_data):
            if row[0] == 'NIFTY':
                for j in range(len(headers)):
                    table[(i+1, j)].set_facecolor('#ffcccc')
        
        ax4.set_title('Entropy Statistics Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'enhanced_entropy_analysis_{method}.png'
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ Enhanced analysis plot saved as {save_path}")
    
    def generate_research_insights(self, method='adaptive_bins'):
        """Generate key research insights and findings."""
        if method not in self.entropy_data:
            return
        
        entropy_data = self.entropy_data[method]
        
        print(f"\n{'='*80}")
        print(f"RESEARCH INSIGHTS - {method.upper()}")
        print(f"{'='*80}")
        
        # Calculate statistics
        stats = {}
        for name, entropy in entropy_data.items():
            stats[name] = {
                'mean': entropy.mean(),
                'std': entropy.std(),
                'cv': entropy.std() / entropy.mean(),
                'min': entropy.min(),
                'max': entropy.max()
            }
        
        # Sort by mean entropy
        sorted_assets = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print("RANKING BY MEAN ENTROPY:")
        for i, (name, stat) in enumerate(sorted_assets, 1):
            print(f"{i:2d}. {name:15s} - {stat['mean']:.4f} ± {stat['std']:.4f}")
        
        # Key findings
        if 'NIFTY' in stats:
            nifty_rank = next(i for i, (name, _) in enumerate(sorted_assets, 1) if name == 'NIFTY')
            nifty_mean = stats['NIFTY']['mean']
            
            stocks_below_nifty = [name for name, stat in stats.items() 
                                if name != 'NIFTY' and stat['mean'] < nifty_mean]
            
            print(f"\nKEY FINDINGS:")
            print(f"• NIFTY ranks #{nifty_rank} out of {len(stats)} assets in mean entropy")
            print(f"• NIFTY mean entropy: {nifty_mean:.4f}")
            print(f"• {len(stocks_below_nifty)}/10 individual stocks have lower entropy than NIFTY")
            
            if len(stocks_below_nifty) >= 7:
                print(f"• HYPOTHESIS SUPPORTED: Index shows higher entropy (more random-walk-like)")
            elif len(stocks_below_nifty) >= 4:
                print(f"• HYPOTHESIS PARTIALLY SUPPORTED: Mixed evidence")
            else:
                print(f"• HYPOTHESIS NOT SUPPORTED: Individual stocks show similar/higher entropy")
            
            # Volatility comparison if available
            if 'NIFTY' in self.volatility_data:
                nifty_vol = self.volatility_data['NIFTY'].mean()
                print(f"• NIFTY mean volatility: {nifty_vol:.4f}")
        
        # Market efficiency insights
        high_entropy_assets = [name for name, stat in sorted_assets[:3]]
        low_entropy_assets = [name for name, stat in sorted_assets[-3:]]
        
        print(f"\nMARKET EFFICIENCY INSIGHTS:")
        print(f"• Most 'efficient' (highest entropy): {', '.join(high_entropy_assets)}")
        print(f"• Least 'efficient' (lowest entropy): {', '.join(low_entropy_assets)}")
        
        # Generate one-liner for paper
        result_pct = len(stocks_below_nifty) * 10
        print(f"\nONE-LINE RESULT FOR PAPER:")
        print(f'"{result_pct}% of large-cap stocks exhibit lower Shannon entropy than NIFTY, '
              f'suggesting the index demonstrates {"higher" if len(stocks_below_nifty) >= 5 else "comparable"} '
              f'randomness consistent with market efficiency theory."')

def main():
    """Main execution function."""
    print("Enhanced Shannon Entropy Analysis for Indian Stock Market")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = EnhancedStockEntropyAnalyzer()
    
    # Download data (full period from 2019 for comprehensive analysis)
    start_date = '2019-01-01'
    analyzer.download_data(start_date=start_date)
    
    if len(analyzer.data) == 0:
        print("No data downloaded. Exiting...")
        return
    
    # Compute log returns
    analyzer.compute_log_returns()
    
    # Compute volatility for comparison
    analyzer.compute_volatility(window=90)
    
    # Compute rolling entropy with multiple methods
    analyzer.compute_rolling_entropy_multiple_methods(window=90)
    
    # Generate comparative analysis for best method
    best_method = 'adaptive_bins'  # Most discriminative method
    analyzer.plot_comparative_analysis(method=best_method)
    
    # Generate research insights
    analyzer.generate_research_insights(method=best_method)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print("Generated files:")
    print(f"- enhanced_entropy_analysis_adaptive_bins.png")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()