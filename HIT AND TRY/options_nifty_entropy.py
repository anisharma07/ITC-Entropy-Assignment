import pandas as pd
import numpy as np
import re
from datetime import datetime

def calculate_shannon_entropy(prices, bins=10):
    """
    Calculate Shannon entropy of price movements.
    """
    if len(prices) < 2:
        return 0
    
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # Create histogram
    hist, _ = np.histogram(returns, bins=bins, density=True)
    
    # Normalize to get probabilities
    hist = hist / np.sum(hist)
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    # Calculate Shannon entropy
    entropy = -np.sum(hist * np.log2(hist))
    
    return entropy

def calculate_vix_from_chain(option_chain_data, spot_price, valuation_date, risk_free_rate):
    """
    Calculates a VIX-like volatility index from a given option chain.
    """
    # --- 1. Data Preparation ---
    lines = option_chain_data.strip().split('\n')
    data = []
    # This regex is designed to parse the specific format of the option chain
    pattern = re.compile(r'([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)\s+([\d,.-]+)')

    for line in lines[2:]: # Skip headers
        match = pattern.search(line)
        if match:
            # Extract all matched groups which correspond to the columns
            row = [val.replace(',', '') for val in match.groups()]
            data.append(row)
    
    # Define columns based on the provided text format
    columns = [
        'CALL_OI', 'CALL_CHNG_IN_OI', 'CALL_VOLUME', 'CALL_IV', 'CALL_LTP', 
        'CALL_CHNG', 'CALL_BID_QTY', 'CALL_BID', 'CALL_ASK', 'CALL_ASK_QTY', 
        'STRIKE_PRICE', 
        'PUT_BID_QTY', 'PUT_BID', 'PUT_ASK', 'PUT_ASK_QTY', 'PUT_CHNG', 
        'PUT_LTP', 'PUT_IV', 'PUT_VOLUME', 'PUT_CHNG_IN_OI', 'PUT_OI'
    ]
    # Adjust for the actual number of columns matched
    df = pd.DataFrame(data, columns=columns[:len(data[0])])

    # Convert columns to numeric, coercing errors
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['STRIKE_PRICE', 'CALL_BID', 'CALL_ASK', 'PUT_BID', 'PUT_ASK'])

    # --- 2. Select Expirations & Calculate Time to Expiry (T) ---
    near_expiry_date = datetime(2025, 9, 30)
    next_expiry_date = datetime(2025, 10, 7)

    T_near = (near_expiry_date - valuation_date).total_seconds() / (365 * 24 * 3600)
    T_next = (next_expiry_date - valuation_date).total_seconds() / (365 * 24 * 3600)

    # --- 3. Calculate Variance for each Expiration Term ---
    def get_variance_for_term(df_term, T, R, F):
        # Calculate mid-prices
        df_term['CALL_MID'] = (df_term['CALL_BID'] + df_term['CALL_ASK']) / 2
        df_term['PUT_MID'] = (df_term['PUT_BID'] + df_term['PUT_ASK']) / 2

        # Determine K0 - the strike immediately below the forward price F
        K0 = df_term[df_term['STRIKE_PRICE'] < F]['STRIKE_PRICE'].max()

        # Select OTM puts (strike < K0) and OTM calls (strike > K0)
        otm_puts = df_term[df_term['STRIKE_PRICE'] < K0][['STRIKE_PRICE', 'PUT_MID']].copy()
        otm_calls = df_term[df_term['STRIKE_PRICE'] > K0][['STRIKE_PRICE', 'CALL_MID']].copy()
        
        # At-the-money options (strike == K0)
        atm_strike = df_term[df_term['STRIKE_PRICE'] == K0]
        atm_price = (atm_strike['CALL_MID'].iloc[0] + atm_strike['PUT_MID'].iloc[0]) / 2
        
        # Combine options for calculation
        otm_puts.rename(columns={'PUT_MID': 'PRICE'}, inplace=True)
        otm_calls.rename(columns={'CALL_MID': 'PRICE'}, inplace=True)
        atm_option = pd.DataFrame([{'STRIKE_PRICE': K0, 'PRICE': atm_price}])
        
        options_to_sum = pd.concat([otm_puts, otm_calls, atm_option]).sort_values(by='STRIKE_PRICE').reset_index(drop=True)
        
        # Calculate delta K
        options_to_sum['DELTA_K'] = (options_to_sum['STRIKE_PRICE'].shift(-1) - options_to_sum['STRIKE_PRICE'].shift(1)) / 2
        options_to_sum.loc[0, 'DELTA_K'] = options_to_sum.loc[1, 'STRIKE_PRICE'] - options_to_sum.loc[0, 'STRIKE_PRICE']
        options_to_sum.loc[len(options_to_sum)-1, 'DELTA_K'] = options_to_sum.loc[len(options_to_sum)-1, 'STRIKE_PRICE'] - options_to_sum.loc[len(options_to_sum)-2, 'STRIKE_PRICE']

        # Calculate each option's contribution to variance
        options_to_sum['CONTRIBUTION'] = (options_to_sum['DELTA_K'] / (options_to_sum['STRIKE_PRICE']**2)) * np.exp(R * T) * options_to_sum['PRICE']

        # Sum contributions and apply the formula term
        sigma_sq = (2 / T) * options_to_sum['CONTRIBUTION'].sum() - (1 / T) * ((F / K0 - 1)**2)
        return sigma_sq

    # Determine Forward Price for each term
    # F = Strike Price + e^(RT) * (Call Price - Put Price)
    df['CALL_PUT_DIFF'] = abs(df['CALL_BID'] - df['PUT_BID'])
    min_diff_strike = df.loc[df['CALL_PUT_DIFF'].idxmin()]
    
    F_near = min_diff_strike['STRIKE_PRICE'] + np.exp(risk_free_rate * T_near) * (min_diff_strike['CALL_BID'] - min_diff_strike['PUT_BID'])
    F_next = min_diff_strike['STRIKE_PRICE'] + np.exp(risk_free_rate * T_next) * (min_diff_strike['CALL_BID'] - min_diff_strike['PUT_BID'])

    # For this simplified example, we'll use the same full dataframe for both terms
    # as the user did not provide separate chains for each expiry.
    sigma_sq_near = get_variance_for_term(df.copy(), T_near, risk_free_rate, F_near)
    sigma_sq_next = get_variance_for_term(df.copy(), T_next, risk_free_rate, F_next)

    # --- 4. Interpolate to 30 days and Final Calculation ---
    N_t1 = T_near * 365
    N_t2 = T_next * 365
    N_30 = 30
    
    term1 = T_near * sigma_sq_near * ((N_t2 - N_30) / (N_t2 - N_t1))
    term2 = T_next * sigma_sq_next * ((N_30 - N_t1) / (N_t2 - N_t1))
    
    vix_value = 100 * np.sqrt((term1 + term2) * (365 / 30))
    
    return vix_value


# --- Main Execution ---
# Assumed data based on user's provided context
valuation_date = datetime(2025, 9, 26)
spot_price = 24654.70
risk_free_rate = 0.05

# Sample option chain data (replace with actual data)
option_chain_data = """
OI	CHNG IN OI	VOLUME	IV	LTP	CHNG	BID QTY	BID	ASK	ASK QTY	STRIKE PRICE	BID QTY	BID	ASK	ASK QTY	CHNG	LTP	IV	VOLUME	CHNG IN OI	OI
1000	100	500	25.5	150.25	5.25	50	149.50	151.00	25	24500	30	48.75	49.25	40	-2.15	49.00	22.8	300	-50	800
1200	150	750	24.2	125.75	3.50	75	124.90	126.60	35	24550	45	73.25	74.00	50	-1.85	73.60	23.5	450	-75	950
1500	200	1000	23.8	102.50	2.25	100	101.75	103.25	60	24600	65	98.50	99.25	70	-1.25	98.85	24.1	600	-100	1100
1800	250	1250	23.1	81.25	1.75	125	80.50	82.00	80	24650	85	124.75	125.50	90	-0.75	125.10	24.8	750	-125	1300
2000	300	1500	22.7	62.75	1.25	150	62.00	63.50	100	24700	105	152.25	153.00	110	-0.25	152.60	25.2	900	-150	1500
"""

# Process the data and calculate metrics
try:
    print("--- NIFTY Options Analysis: VIX-Like Index & Shannon Entropy ---")
    print(f"Valuation Date: {valuation_date.strftime('%Y-%m-%d')}")
    print(f"Spot Price: ₹{spot_price:,.2f}")
    print(f"Risk-Free Rate: {risk_free_rate:.2%}")
    print("\n" + "="*60 + "\n")
    
    # Calculate VIX-like volatility index
    vix = calculate_vix_from_chain(option_chain_data, spot_price, valuation_date, risk_free_rate)
    print(f"📊 Calculated NIFTY Volatility Index: {vix:.2f}")
    
    # Parse option chain for entropy calculation
    lines = option_chain_data.strip().split('\n')
    if len(lines) > 2:
        # Extract call and put prices for entropy calculation
        call_prices = []
        put_prices = []
        
        for line in lines[2:]:  # Skip headers
            parts = line.split()
            if len(parts) >= 18:
                try:
                    call_ltp = float(parts[4])  # CALL_LTP
                    put_ltp = float(parts[16])  # PUT_LTP
                    if call_ltp > 0:
                        call_prices.append(call_ltp)
                    if put_ltp > 0:
                        put_prices.append(put_ltp)
                except (ValueError, IndexError):
                    continue
        
        # Calculate Shannon entropy for option prices
        if len(call_prices) > 1:
            call_entropy = calculate_shannon_entropy(call_prices)
            print(f"🔢 Call Options Shannon Entropy: {call_entropy:.4f} bits")
        
        if len(put_prices) > 1:
            put_entropy = calculate_shannon_entropy(put_prices)
            print(f"🔢 Put Options Shannon Entropy: {put_entropy:.4f} bits")
        
        # Combined analysis
        all_prices = call_prices + put_prices
        if len(all_prices) > 1:
            combined_entropy = calculate_shannon_entropy(all_prices)
            print(f"🔢 Combined Options Shannon Entropy: {combined_entropy:.4f} bits")
            
            # Interpretation
            print(f"\n📈 Market Analysis:")
            print(f"   • Higher entropy ({combined_entropy:.4f}) indicates more uncertainty/randomness")
            print(f"   • Lower entropy indicates more predictable price patterns")
            print(f"   • VIX-like index ({vix:.2f}) measures implied volatility")
            
            # Market state interpretation
            if combined_entropy > 2.5:
                market_state = "High Uncertainty"
            elif combined_entropy > 1.5:
                market_state = "Moderate Uncertainty"
            else:
                market_state = "Low Uncertainty"
            
            print(f"   • Current Market State: {market_state}")
    
    print(f"\n" + "="*60)
    print("✅ Analysis Complete!")

except Exception as e:
    print("❌ Error in calculation:")
    print(f"   {str(e)}")
    print("\n💡 Note: This script demonstrates entropy calculation concepts.")
    print("   For production use, integrate with real-time option chain APIs.")

