import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def calculate_max_pain(option_chain_data):
    """
    Parses option chain data and calculates the max pain strike price.
    """
    # --- 1. Data Parsing and Cleaning ---
    lines = option_chain_data.strip().split('\n')
    data = []
    pattern = re.compile(r'([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)')

    for line in lines[2:]:
        match = pattern.search(line)
        if match:
            row = [val.replace(',', '') for val in match.groups()]
            data.append(row)
    
    columns = [
        'CALL_OI', 'CALL_CHNG_IN_OI', 'CALL_VOLUME', 'CALL_IV', 'CALL_LTP', 
        'CALL_CHNG', 'CALL_BID_QTY', 'CALL_BID', 'CALL_ASK', 'CALL_ASK_QTY', 
        'STRIKE_PRICE', 
        'PUT_BID_QTY', 'PUT_BID', 'PUT_ASK', 'PUT_ASK_QTY', 'PUT_CHNG', 
        'PUT_LTP', 'PUT_IV', 'PUT_VOLUME', 'PUT_CHNG_IN_OI', 'PUT_OI'
    ]
    df = pd.DataFrame(data, columns=columns[:len(data[0])])

    # Convert key columns to numeric
    for col in ['STRIKE_PRICE', 'CALL_OI', 'PUT_OI']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['STRIKE_PRICE', 'CALL_OI', 'PUT_OI'])

    # --- 2. Max Pain Calculation ---
    strike_prices = df['STRIKE_PRICE'].unique()
    total_losses = []

    for expiry_price in strike_prices:
        # Calculate loss for call option holders
        # Loss = (Expiry Price - Strike Price) * OI, if Expiry > Strike
        call_loss = ((expiry_price - df['STRIKE_PRICE']) * df['CALL_OI']).clip(lower=0).sum()
        
        # Calculate loss for put option holders
        # Loss = (Strike Price - Expiry Price) * OI, if Strike > Expiry
        put_loss = ((df['STRIKE_PRICE'] - expiry_price) * df['PUT_OI']).clip(lower=0).sum()
        
        total_losses.append({'strike': expiry_price, 'loss': call_loss + put_loss})

    loss_df = pd.DataFrame(total_losses)
    max_pain_strike = loss_df.loc[loss_df['loss'].idxmin()]

    return max_pain_strike, loss_df

# --- Main Execution ---
# Paste the raw option chain data for Trent Ltd.
option_chain_data = """
PASTE THE TRENT LTD OPTION CHAIN DATA HERE
"""

# Placeholder for running the script. Replace with the actual data from the prompt.
try:
    max_pain_info, loss_data = calculate_max_pain(option_chain_data)
    print("--- Max Pain Analysis for Trent Ltd. (Expiry: 30-Sep-2025) ---")
    print(f"Max Pain Strike Price: ₹{max_pain_info['strike']:.2f}")
    print(f"Total Notional Value at Max Pain: ₹{max_pain_info['loss']/1e7:.2f} Crores")

    # --- 3. Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(loss_data['strike'], loss_data['loss'] / 1e7, marker='o', linestyle='-', color='#003f5c', label='Total Notional Value (in Crores)')
    
    # Highlight the Max Pain point
    ax.axvline(x=max_pain_info['strike'], color='#ff6361', linestyle='--', linewidth=2, label=f"Max Pain: ₹{max_pain_info['strike']:.0f}")
    
    ax.set_title('Trent Ltd. - Max Pain Analysis for 30-Sep-2025 Expiry', fontsize=16)
    ax.set_xlabel('Strike Price (₹)', fontsize=12)
    ax.set_ylabel('Total Notional Value of Expiring Options (₹ Crores)', fontsize=12)
    ax.legend()
    ax.grid(True)
    
    # Format y-axis to show "Cr"
    formatter = plt.FuncFormatter(lambda x, pos: f'₹{x:.0f} Cr')
    ax.yaxis.set_major_formatter(formatter)
    
    plt.tight_layout()
    plt.savefig('trent_max_pain.png')
    print("\nGraph has been generated and saved as 'trent_max_pain.png'")

except Exception:
    print("Could not parse the provided data. The script requires the exact text from the option chain.")

