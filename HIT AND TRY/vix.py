import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import re

# Raw data provided by the user
data = """
DATE EXPIRY DATE OPTION TYPE STRIKE PRICE OPEN PRICE HIGH PRICE LOW PRICE CLOSE PRICE LAST PRICE SETTLE PRICE Volume VALUE (₹ Lakhs) PREMIUM VALUE (₹ Lakhs) OPEN INTEREST CHANGE IN OI
26-Sep-2025 25-Nov-2025 XX- 25,160.10 25,172.40 24,914.00 24,946.30 24,950.00 24,946.30 4,21,950 1,05,632.67 1,05,632.67 8,39,100 1,40,550
26-Sep-2025 30-Sep-2025 XX- 24,929.90 24,929.90 24,663.10 24,689.90 24,700.00 24,689.90 72,68,550 18,01,003.14 18,01,003.14 96,91,950 -28,13,100
26-Sep-2025 28-Oct-2025 XX- 25,100.00 25,100.00 24,785.50 24,812.00 24,819.70 24,812.00 55,54,350 13,83,149.09 13,83,149.09 97,68,150 37,63,650
25-Sep-2025 30-Sep-2025 XX- 25,100.00 25,133.00 24,954.10 24,967.70 24,962.00 24,967.70 57,92,775 14,49,810.92 14,49,810.92 1,25,05,050 -17,05,650
25-Sep-2025 28-Oct-2025 XX- 25,237.00 25,256.90 25,081.00 25,093.70 25,087.40 25,093.70 34,37,325 8,64,490.15 8,64,490.15 60,04,500 23,16,975
25-Sep-2025 25-Nov-2025 XX- 25,351.70 25,393.70 25,218.70 25,229.60 25,221.90 25,229.60 2,63,475 66,609.72 66,609.72 6,98,550 88,500
24-Sep-2025 28-Oct-2025 XX- 25,338.40 25,338.50 25,209.00 25,235.80 25,237.00 25,235.80 16,67,550 4,21,325.74 4,21,325.74 36,87,525 8,10,750
24-Sep-2025 25-Nov-2025 XX- 25,500.00 25,500.00 25,344.00 25,373.60 25,373.00 25,373.60 2,01,000 51,053.39 51,053.39 6,10,050 53,400
24-Sep-2025 30-Sep-2025 XX- 25,206.50 25,206.50 25,078.00 25,111.90 25,113.00 25,111.90 48,35,325 12,15,446.82 12,15,446.82 1,42,10,700 -1,48,200
23-Sep-2025 28-Oct-2025 XX- 25,412.70 25,458.90 25,282.90 25,383.30 25,384.10 25,383.30 10,22,700 2,59,474.87 2,59,474.87 28,76,775 2,43,225
23-Sep-2025 25-Nov-2025 XX- 25,500.10 25,585.00 25,420.00 25,517.20 25,518.50 25,517.20 1,57,950 40,269.41 40,269.41 5,56,650 22,125
23-Sep-2025 30-Sep-2025 XX- 25,277.70 25,333.20 25,158.00 25,255.80 25,252.00 25,255.80 47,23,725 11,92,385.03 11,92,385.03 1,43,58,900 -3,98,925
22-Sep-2025 25-Nov-2025 XX- 25,577.70 25,636.00 25,502.20 25,533.10 25,535.00 25,533.10 1,71,600 43,892.72 43,892.72 5,34,525 63,900
22-Sep-2025 30-Sep-2025 XX- 25,350.00 25,394.50 25,248.60 25,277.70 25,276.00 25,277.70 38,43,900 9,73,525.34 9,73,525.34 1,47,57,825 -2,49,450
22-Sep-2025 28-Oct-2025 XX- 25,469.70 25,515.00 25,375.00 25,403.40 25,408.80 25,403.40 7,49,400 1,90,704.06 1,90,704.06 26,33,550 3,18,975
19-Sep-2025 28-Oct-2025 XX- 25,629.10 25,629.10 25,475.00 25,534.50 25,541.00 25,534.50 8,74,950 2,23,303.02 2,23,303.02 23,14,575 3,70,875
19-Sep-2025 25-Nov-2025 XX- 25,712.00 25,729.80 25,605.00 25,650.30 25,668.00 25,650.30 1,46,625 37,605.83 37,605.83 4,70,625 5,175
19-Sep-2025 30-Sep-2025 XX- 25,474.00 25,489.80 25,352.00 25,411.20 25,426.70 25,411.20 47,24,025 11,99,982.41 11,99,982.41 1,50,07,275 -5,12,625
18-Sep-2025 25-Nov-2025 XX- 25,725.00 25,766.90 25,681.00 25,752.10 25,755.00 25,752.10 1,53,975 39,624.32 39,624.32 4,65,450 31,350
18-Sep-2025 28-Oct-2025 XX- 25,601.00 25,640.00 25,555.20 25,629.10 25,630.00 25,629.10 7,62,300 1,95,225.20 1,95,225.20 19,43,700 1,71,000
18-Sep-2025 30-Sep-2025 XX- 25,480.20 25,525.00 25,435.10 25,510.90 25,510.00 25,510.90 40,50,300 10,32,331.59 10,32,331.59 1,55,19,900 -1,75,950
17-Sep-2025 28-Oct-2025 XX- 25,500.00 25,599.90 25,490.00 25,540.00 25,545.00 25,540.00 5,46,375 1,39,480.93 1,39,480.93 17,72,700 1,84,950
17-Sep-2025 25-Nov-2025 XX- 25,643.20 25,678.50 25,600.80 25,664.10 25,668.00 25,664.10 1,25,775 32,258.52 32,258.52 4,34,100 22,275
17-Sep-2025 30-Sep-2025 XX- 25,365.00 25,446.70 25,362.80 25,423.40 25,422.00 25,423.40 33,44,850 8,49,979.44 8,49,979.44 1,56,95,850 -3,86,250
16-Sep-2025 25-Nov-2025 XX- 25,355.00 25,590.00 25,355.00 25,568.90 25,588.00 25,568.90 1,75,125 44,685.09 44,685.09 4,11,825 50,400
16-Sep-2025 28-Oct-2025 XX- 25,250.10 25,470.00 25,250.10 25,449.60 25,468.90 25,449.60 5,21,700 1,32,460.85 1,32,460.85 15,87,750 92,025
16-Sep-2025 30-Sep-2025 XX- 25,178.70 25,354.00 25,152.00 25,331.40 25,351.10 25,331.40 45,08,175 11,39,515.21 11,39,515.21 1,60,82,100 -5,72,625
15-Sep-2025 28-Oct-2025 XX- 25,319.90 25,321.10 25,271.20 25,284.00 25,281.00 25,284.00 2,13,375 53,966.35 53,966.35 14,95,725 56,700
15-Sep-2025 30-Sep-2025 XX- 25,200.60 25,204.00 25,147.30 25,164.70 25,169.00 25,164.70 25,72,575 6,47,546.35 6,47,546.35 1,66,54,725 16,950
15-Sep-2025 25-Nov-2025 XX- 25,439.00 25,439.00 25,391.30 25,403.70 25,400.00 25,403.70 55,125 14,006.59 14,006.59 3,61,425 10,050
12-Sep-2025 25-Nov-2025 XX- 25,372.90 25,458.30 25,370.00 25,438.80 25,446.40 25,438.80 98,400 25,014.72 25,014.72 3,51,375 23,775
12-Sep-2025 30-Sep-2025 XX- 25,126.30 25,222.80 25,126.30 25,205.00 25,210.10 25,205.00 31,18,725 7,85,551.73 7,85,551.73 1,66,37,775 -4,24,950
12-Sep-2025 28-Oct-2025 XX- 25,257.60 25,338.00 25,245.60 25,321.80 25,327.80 25,321.80 3,27,900 82,972.38 82,972.38 14,39,025 88,575
11-Sep-2025 28-Oct-2025 XX- 25,178.30 25,233.90 25,167.90 25,222.10 25,222.10 25,222.10 1,77,975 44,861.18 44,861.18 13,50,450 46,950
11-Sep-2025 30-Sep-2025 XX- 25,049.00 25,119.50 25,048.10 25,104.50 25,110.00 25,104.50 21,42,075 5,37,476.31 5,37,476.31 1,70,62,725 -1,68,300
11-Sep-2025 25-Nov-2025 XX- 25,448.30 25,448.30 25,286.40 25,341.60 25,345.00 25,341.60 58,200 14,739.24 14,739.24 3,27,600 15,825
10-Sep-2025 25-Nov-2025 XX- 25,260.00 25,355.00 25,240.40 25,307.50 25,315.00 25,307.50 1,23,975 31,368.96 31,368.96 3,11,775 37,875
10-Sep-2025 30-Sep-2025 XX- 25,021.60 25,125.00 25,001.10 25,072.30 25,079.00 25,072.30 49,59,600 12,43,468.65 12,43,468.65 1,72,31,025 2,99,475
10-Sep-2025 28-Oct-2025 XX- 25,150.00 25,239.90 25,122.60 25,188.60 25,193.00 25,188.60 3,79,125 95,483.12 95,483.12 13,03,500 29,850
09-Sep-2025 28-Oct-2025 XX- 25,007.40 25,080.00 25,007.40 25,065.30 25,068.00 25,065.30 1,72,650 43,264.42 43,264.42 12,73,650 20,550
09-Sep-2025 25-Nov-2025 XX- 25,172.80 25,197.00 25,149.20 25,181.80 25,190.00 25,181.80 69,675 17,542.56 17,542.56 2,73,900 28,125
09-Sep-2025 30-Sep-2025 XX- 24,931.20 24,969.00 24,910.00 24,950.30 24,961.00 24,950.30 27,80,175 6,93,536.03 6,93,536.03 1,69,31,550 -41,700
08-Sep-2025 25-Nov-2025 XX- 25,119.90 25,211.00 25,092.80 25,121.70 25,142.00 25,121.70 1,44,375 36,323.39 36,323.39 2,45,775 62,175
08-Sep-2025 30-Sep-2025 XX- 24,888.50 24,979.00 24,862.20 24,892.70 24,900.80 24,892.70 44,74,800 11,15,060.64 11,15,060.64 1,69,73,250 -1,24,500
08-Sep-2025 28-Oct-2025 XX- 25,020.10 25,093.90 24,975.00 25,007.40 25,025.80 25,007.40 3,04,125 76,138.48 76,138.48 12,53,100 34,725
05-Sep-2025 30-Sep-2025 XX- 24,855.30 24,927.00 24,720.40 24,847.70 24,849.60 24,847.70 55,35,150 13,73,816.59 13,73,816.59 1,70,97,750 1,39,875
05-Sep-2025 25-Nov-2025 XX- 25,130.00 25,152.70 24,963.00 25,085.00 25,077.40 25,085.00 1,06,650 26,723.30 26,723.30 1,83,600 6,150
05-Sep-2025 28-Oct-2025 XX- 24,989.40 25,040.50 24,840.10 24,964.50 24,957.40 24,964.50 3,22,725 80,465.52 80,465.52 12,18,375 84,825
04-Sep-2025 25-Nov-2025 XX- 25,200.00 25,264.80 25,051.10 25,069.80 25,076.20 25,069.80 1,05,000 26,420.74 26,420.74 1,77,450 13,875
04-Sep-2025 30-Sep-2025 XX- 24,926.30 25,048.50 24,809.70 24,827.50 24,836.00 24,827.50 62,21,325 15,50,518.69 15,50,518.69 1,69,57,875 4,56,000
04-Sep-2025 28-Oct-2025 XX- 25,030.00 25,150.00 24,930.00 24,947.80 24,953.00 24,947.80 3,79,275 95,003.01 95,003.01 11,33,550 56,700
03-Sep-2025 25-Nov-2025 XX- 24,914.50 25,070.00 24,880.00 25,052.40 25,049.00 25,052.40 72,750 18,169.23 18,169.23 1,63,575 15,225
03-Sep-2025 30-Sep-2025 XX- 24,680.00 24,837.00 24,638.50 24,813.10 24,820.00 24,813.10 42,35,175 10,47,832.93 10,47,832.93 1,65,01,875 5,550
03-Sep-2025 28-Oct-2025 XX- 24,808.80 24,950.00 24,756.40 24,933.00 24,931.40 24,933.00 2,33,175 57,953.41 57,953.41 10,76,850 46,500
02-Sep-2025 30-Sep-2025 XX- 24,750.00 24,874.70 24,651.90 24,691.80 24,682.20 24,691.80 56,18,325 13,92,010.42 13,92,010.42 1,64,96,325 -1,24,725
02-Sep-2025 28-Oct-2025 XX- 24,881.60 24,990.00 24,770.00 24,808.90 24,800.00 24,808.90 4,26,900 1,06,295.72 1,06,295.72 10,30,350 42,975
02-Sep-2025 25-Nov-2025 XX- 25,000.00 25,105.30 24,895.40 24,926.50 24,914.50 24,926.50 1,65,300 41,337.43 41,337.43 1,48,350 62,775
01-Sep-2025 30-Sep-2025 XX- 24,581.10 24,760.00 24,581.00 24,748.00 24,759.00 24,748.00 39,10,500 9,65,806.74 9,65,806.74 1,66,21,050 10,950
01-Sep-2025 28-Oct-2025 XX- 24,699.00 24,882.10 24,693.50 24,874.00 24,881.70 24,874.00 2,52,225 62,592.38 62,592.38 9,87,375 14,850
01-Sep-2025 25-Nov-2025 XX- 24,847.10 24,998.00 24,839.70 24,990.60 24,995.00 24,990.60 72,000 17,953.16 17,953.16 85,575 38,250
29-Aug-2025 30-Sep-2025 XX- 24,650.00 24,710.00 24,546.00 24,568.50 24,577.30 24,568.50 51,54,825 12,70,133.09 12,70,133.09 1,66,10,100 5,83,050
29-Aug-2025 28-Oct-2025 XX- 24,753.90 24,829.90 24,665.00 24,684.30 24,699.00 24,684.30 4,05,300 1,00,280.67 1,00,280.67 9,72,525 83,625
29-Aug-2025 25-Nov-2025 XX- 24,880.00 24,952.00 24,791.60 24,810.90 24,807.00 24,810.90 92,925 23,121.24 23,121.24 47,325 47,325
"""

# Define a function to calculate entropy
def calculate_entropy(prices_series):
    """Calculates Shannon Entropy for a given price series."""
    df = pd.DataFrame(prices_series)
    df.columns = ['price']
    
    # Step 1: Calculate Log Returns
    df['log_return'] = np.log(df['price'] / df['price'].shift(1))
    df = df.dropna()
    
    if len(df) < 4: # Need enough data for 4 quartiles
        return np.nan

    # Step 2: Discretize with Quantile-Based Method (4 Bins)
    try:
        df['state'] = pd.qcut(df['log_return'], q=4, labels=False, duplicates='drop')
    except ValueError:
        return np.nan # Not enough unique values to form 4 quantiles
        
    # Step 3: Calculate Probabilities
    probabilities = df['state'].value_counts(normalize=True)
    
    # Step 4: Compute Shannon Entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# --- Data Preparation ---
# Use StringIO to read the string data into a pandas DataFrame
data_io = io.StringIO(data)
# The first line is a header, but the column names are merged. We'll define them manually.
columns = [
    'DATE', 'EXPIRY_DATE', 'OPTION_TYPE', 'STRIKE_PRICE', 'OPEN_PRICE', 
    'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'LAST_PRICE', 'SETTLE_PRICE', 
    'Volume', 'VALUE', 'PREMIUM_VALUE', 'OPEN_INTEREST', 'CHANGE_IN_OI'
]
# We need to parse the fixed-width-like format of the data
# This regex is designed to capture the first three columns, then the numerical data
# It specifically looks for the XX- which denotes a futures contract
pattern = re.compile(r"(\d{2}-[A-Za-z]{3}-\d{4})\s+(\d{2}-[A-Za-z]{3}-\d{4})\s+XX-\s+([\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.]+\s+[\d,.-]+)\s*")

parsed_data = []
for line in data.strip().split('\n')[1:]: # Skip header
    match = pattern.search(line)
    if match:
        date, expiry, rest_of_line = match.groups()
        # Split the remaining numerical values
        values = re.split(r'\s+', rest_of_line.strip())
        # Clean commas from numbers
        values = [v.replace(',', '') for v in values]
        row = [date, expiry] + values
        parsed_data.append(row)

# Create DataFrame with appropriate columns for futures
futures_columns = ['DATE', 'EXPIRY_DATE', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 
                   'CLOSE_PRICE', 'LAST_PRICE', 'SETTLE_PRICE', 'Volume', 'VALUE', 
                   'OPEN_INTEREST', 'CHANGE_IN_OI']
df_full = pd.DataFrame(parsed_data, columns=futures_columns)

# Convert data types
df_full['DATE'] = pd.to_datetime(df_full['DATE'], format='%d-%b-%Y')
df_full['EXPIRY_DATE'] = pd.to_datetime(df_full['EXPIRY_DATE'], format='%d-%b-%Y')
numeric_cols = ['SETTLE_PRICE']
for col in numeric_cols:
    df_full[col] = pd.to_numeric(df_full[col])

# Sort by date to ensure correct time series order
df_full = df_full.sort_values(by='DATE').reset_index(drop=True)

# --- Analysis ---
# Get the unique expiry dates to analyze
expiry_dates = df_full['EXPIRY_DATE'].unique()
entropy_results = {}

print("--- Analyzing Shannon Entropy for Each Futures Contract ---")
for expiry in expiry_dates:
    expiry_str = expiry.strftime('%B %Y')
    
    # Isolate the price series for the current contract
    price_series = df_full[df_full['EXPIRY_DATE'] == expiry]['SETTLE_PRICE']
    
    # Calculate entropy
    entropy = calculate_entropy(price_series)
    
    if not np.isnan(entropy):
        entropy_results[expiry_str] = entropy
        print(f"Entropy for {expiry_str} contract: {entropy:.4f} bits")
    else:
        print(f"Could not calculate entropy for {expiry_str} contract (insufficient data).")

print("\n" + "="*40 + "\n")

# --- Key Outcome ---
print("--- Key Outcome ---")
# Find the contract with the highest and lowest entropy
if entropy_results:
    most_predictable = min(entropy_results, key=entropy_results.get)
    least_predictable = max(entropy_results, key=entropy_results.get)

    print("The analysis compares the Shannon Entropy across different NIFTY 50 futures contracts.")
    print("A higher entropy value suggests more randomness and less predictability, while a lower value suggests more predictable patterns.")
    print(f"\n- Most Predictable Contract: {most_predictable} (Entropy: {entropy_results[most_predictable]:.4f} bits)")
    print(f"- Least Predictable (Most Random) Contract: {least_predictable} (Entropy: {entropy_results[least_predictable]:.4f} bits)")
else:
    print("No valid entropy results could be calculated.")

# --- Visualization ---
if entropy_results:
    labels = list(entropy_results.keys())
    values = list(entropy_results.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=['#4a7b9d', '#fdc500', '#c1121f'])
    
    # Add a line for maximum entropy (for a 4-state system)
    plt.axhline(y=2.0, color='gray', linestyle='--', label='Maximum Entropy (2.0 bits)')
    
    plt.ylabel('Shannon Entropy (bits)')
    plt.title('NIFTY Futures Contract Predictability Comparison (Sept 2025)')
    plt.ylim(0, 2.2)
    plt.legend()

    # Add text labels on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig('nifty_entropy_comparison.png')
    print("\nGraph has been generated and saved as 'nifty_entropy_comparison.png'")

