import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import io

# Raw data for Trent Ltd. provided by the user
trent_data = """
Instrument Type Expiry Date Option Strike Open High Low Close Prev. Close Last chng %Chng Volume (Contracts) Value (₹ Lakhs)
Stock Options 30-Sep-2025 Call 4,800.00 40.05 45.00 9.90 11.85 40.00 13.40 -26.60 -66.50 23,860 661.64
Stock Futures 28-Oct-2025 -- 4,755.50 4,805.00 4,683.00 4,707.50 4,780.00 4,723.00 -57.00 -1.19 23,691 1,12,491.50
Stock Futures 30-Sep-2025 -- 4,755.50 4,780.50 4,659.00 4,683.00 4,749.00 4,699.00 -50.00 -1.05 21,689 1,02,506.55
Stock Options 30-Sep-2025 Call 5,000.00 9.00 9.00 2.20 2.75 8.75 2.80 -5.95 -68.00 18,189 95.49
Stock Options 30-Sep-2025 Call 4,900.00 17.60 18.80 4.05 5.15 17.05 5.50 -11.55 -67.74 17,368 188.44
Stock Options 30-Sep-2025 Put 4,700.00 49.95 70.90 23.35 54.10 35.70 44.95 9.25 25.91 13,522 562.38
Stock Options 30-Sep-2025 Put 4,600.00 19.75 26.10 9.80 18.70 13.95 15.60 1.65 11.83 11,847 206.26
Stock Options 30-Sep-2025 Call 5,200.00 3.35 3.70 1.65 1.95 3.55 1.85 -1.70 -47.89 10,368 23.54
Stock Options 30-Sep-2025 Call 5,100.00 5.50 5.80 1.90 2.20 5.15 2.15 -3.00 -58.25 8,933 27.42
Stock Options 30-Sep-2025 Call 4,700.00 82.95 104.90 31.00 37.85 82.95 42.50 -40.45 -48.76 7,737 444.10
Stock Options 28-Oct-2025 Call 5,000.00 105.00 113.20 76.55 83.55 105.05 85.00 -20.05 -19.09 4,202 396.88
Stock Options 25-Nov-2025 Put 4,800.00 244.95 272.10 244.95 272.10 247.95 272.10 24.15 9.74 6 1.53
"""

def analyze_derivatives_activity(data):
    # --- 1. Parse and Clean Data ---
    # Split data into lines and process manually
    lines = data.strip().split('\n')
    
    # Use a more robust parsing approach
    header_line = lines[0]
    data_lines = lines[1:]
    
    # Parse each line by splitting with multiple approaches
    parsed_data = []
    
    for line in data_lines:
        if line.strip():  # Skip empty lines
            # More aggressive splitting - split by any whitespace and then recombine
            parts = line.split()
            
            # Manually reconstruct the columns based on expected structure
            if len(parts) >= 12:  # Minimum required columns
                row = []
                # Instrument Type (Stock Options/Futures)
                row.append(parts[0] + ' ' + parts[1])
                # Expiry Date
                row.append(parts[2])
                # Option (Call/Put/-- for futures)
                row.append(parts[3])
                # Strike
                row.append(parts[4])
                # Open, High, Low, Close, Prev Close, Last
                row.extend(parts[5:11])
                # Change, Change %
                row.extend(parts[11:13])
                # Volume and Value (last two)
                row.extend(parts[13:15])
                
                parsed_data.append(row)
    
    # Create DataFrame with proper column names
    columns = ['Instrument Type', 'Expiry Date', 'Option', 'Strike', 'Open', 'High', 
               'Low', 'Close', 'Prev Close', 'Last', 'Change', 'Change %', 
               'Volume (Contracts)', 'Value (₹ Lakhs)']
    
    df = pd.DataFrame(parsed_data, columns=columns)
    
    # Clean the 'Volume (Contracts)' column and convert to numeric
    df['Volume'] = pd.to_numeric(df['Volume (Contracts)'].str.replace(',', ''), errors='coerce')
    df.dropna(subset=['Volume'], inplace=True)
    df['Volume'] = df['Volume'].astype(int)
    
    # --- 2. Analyze Activity by Expiry ---
    unique_expiries = df['Expiry Date'].unique()
    analysis_results = []
    
    print("--- Derivatives Activity Analysis for Trent Ltd. ---")
    
    for expiry in unique_expiries:
        # Filter data for the current expiry date
        expiry_df = df[df['Expiry Date'] == expiry]
        
        # Find the contract with the highest volume
        most_active = expiry_df.loc[expiry_df['Volume'].idxmax()]
        
        # Format the description of the most active contract
        instrument_desc = f"₹{most_active['Strike']} {most_active['Option']}"
        if 'Future' in most_active['Instrument Type']:
            instrument_desc = "Futures Contract"
            
        result = {
            'Expiry': expiry,
            'Instrument': instrument_desc,
            'Volume': most_active['Volume']
        }
        analysis_results.append(result)

        print(f"\n## Analysis for {expiry} Expiry")
        print(f"- Most Active Contract: {result['Instrument']}")
        print(f"- Volume: {result['Volume']:,} contracts")

    return pd.DataFrame(analysis_results)

# --- 3. Generate Visualization ---
def plot_activity(results_df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(results_df['Expiry'], results_df['Volume'], color=['#004c6d', '#69a5c9', '#cce4f2'])
    
    ax.set_title('Trent Ltd. - Most Active Contract Volume by Expiry', fontsize=16)
    ax.set_ylabel('Volume (Number of Contracts)', fontsize=12)
    ax.set_xlabel('Expiration Date', fontsize=12)
    
    # Add volume numbers on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval):,}', va='bottom', ha='center')
        
    plt.tight_layout()
    plt.savefig('trent_activity_analysis.png')
    print("\nGraph has been generated and saved as 'trent_activity_analysis.png'")

# --- Main Execution ---
activity_df = analyze_derivatives_activity(trent_data)
plot_activity(activity_df)

