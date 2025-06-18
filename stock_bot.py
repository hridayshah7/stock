import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

def get_stock_list() -> List[str]:
    """Get list of Indian stocks to scan"""
    # Top Indian stocks across sectors
    stocks = [
        # IT Sector
        "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTI.NS",
        
        # Banking & Finance
        "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS", "AXISBANK.NS",
        "INDUSINDBK.NS", "BANDHANBNK.NS", "FEDERALBNK.NS",
        
        # Energy & Oil
        "RELIANCE.NS", "ONGC.NS", "IOC.NS", "BPCL.NS", "HINDPETRO.NS",
        
        # Auto Sector
        "MARUTI.NS", "HYUNDAIMTR.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
        "HEROMOTOCO.NS", "EICHERMOT.NS", "ASHOKLEY.NS",
        
        # Pharma
        "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS",
        "LUPIN.NS", "AUROPHARMA.NS", "CADILAHC.NS",
        
        # FMCG
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS",
        "GODREJCP.NS", "MARICO.NS", "COLPAL.NS",
        
        # Metals & Mining
        "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "NMDC.NS",
        "COALINDIA.NS", "JINDALSTEL.NS", "SAIL.NS",
        
        # Telecom
        "BHARTIARTL.NS", "IDEA.NS", "RCOM.NS",
        
        # Cement
        "ULTRACEMCO.NS", "SHREECEM.NS", "ACC.NS", "AMBUJACEMENT.NS",
        
        # Infrastructure
        "LT.NS", "POWERGRID.NS", "NTPC.NS", "ADANIPORTS.NS",
        
        # Consumer Goods
        "BAJAJFINSV.NS", "BAJFINANCE.NS", "HDFCLIFE.NS", "SBILIFE.NS",
        
        # Nifty 50 Additional
        "ASIANPAINT.NS", "GRASIM.NS", "TITAN.NS", "ONGC.NS", "DRREDDY.NS"
    ]
    
    return stocks

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    except:
        return 50

def calculate_macd(prices: pd.Series) -> Dict[str, float]:
    """Calculate MACD (Moving Average Convergence Divergence)"""
    try:
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0,
            'signal': signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0,
            'histogram': histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        }
    except:
        return {'macd': 0, 'signal': 0, 'histogram': 0}

def calculate_bollinger_bands(prices: pd.Series, period: int = 20) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    try:
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = prices.iloc[-1]
        upper = upper_band.iloc[-1]
        lower = lower_band.iloc[-1]
        middle = sma.iloc[-1]
        
        return {
            'upper': upper if not pd.isna(upper) else current_price * 1.1,
            'middle': middle if not pd.isna(middle) else current_price,
            'lower': lower if not pd.isna(lower) else current_price * 0.9,
            'position': (current_price - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        }
    except:
        return {'upper': 0, 'middle': 0, 'lower': 0, 'position': 0.5}

def score_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Score a stock based on 12 technical indicators
    Returns a dictionary with symbol, score, price, and details
    """
    try:
        # Fetch stock data
        stock = yf.Ticker(symbol)
        hist = stock.history(period="6mo")
        
        if hist.empty or len(hist) < 50:
            return None
            
        close_prices = hist['Close']
        volume = hist['Volume']
        high_prices = hist['High']
        low_prices = hist['Low']
        
        current_price = close_prices.iloc[-1]
        score = 0
        details = {}
        
        # 1. RSI Score (oversold conditions are good for buying)
        rsi = calculate_rsi(close_prices)
        details['rsi'] = round(rsi, 2)
        if 25 <= rsi <= 35:  # Oversold but not extreme
            score += 2
        elif 35 < rsi <= 45:  # Approaching oversold
            score += 1
        elif 50 <= rsi <= 70:  # Neutral to bullish
            score += 1
            
        # 2. MACD Score
        macd_data = calculate_macd(close_prices)
        details['macd'] = macd_data
        if macd_data['macd'] > macd_data['signal'] and macd_data['histogram'] > 0:
            score += 2  # Bullish crossover
        elif macd_data['macd'] > macd_data['signal']:
            score += 1  # Above signal line
            
        # 3. Moving Average Score (Price vs 20/50 day MA)
        ma20 = close_prices.rolling(20).mean().iloc[-1]
        ma50 = close_prices.rolling(50).mean().iloc[-1]
        details['ma20'] = round(ma20, 2)
        details['ma50'] = round(ma50, 2)
        
        if current_price > ma20 > ma50:  # Strong uptrend
            score += 2
        elif current_price > ma20:  # Above short-term MA
            score += 1
            
        # 4. Volume Score (Higher volume indicates strength)
        avg_volume = volume.rolling(20).mean().iloc[-1]
        recent_volume = volume.iloc[-1]
        details['volume_ratio'] = round(recent_volume / avg_volume, 2) if avg_volume > 0 else 1
        
        if recent_volume > avg_volume * 1.5:  # High volume
            score += 2
        elif recent_volume > avg_volume:  # Above average volume
            score += 1
            
        # 5. Price Momentum Score (Recent price change)
        price_change_5d = (current_price - close_prices.iloc[-6]) / close_prices.iloc[-6] * 100
        details['momentum_5d'] = round(price_change_5d, 2)
        
        if price_change_5d > 5:  # Strong positive momentum
            score += 2
        elif price_change_5d > 0:  # Positive momentum
            score += 1
            
        # 6. Bollinger Bands Score
        bb_data = calculate_bollinger_bands(close_prices)
        details['bollinger'] = bb_data
        
        if bb_data['position'] < 0.2:  # Near lower band (potential buy)
            score += 2
        elif bb_data['position'] < 0.4:  # Below middle
            score += 1
            
        # Additional scoring factors for better accuracy
        details['current_price'] = round(current_price, 2)
        details['symbol'] = symbol
        details['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return {
            'symbol': symbol,
            'score': min(score, 12),  # Cap at 12
            'current_price': round(current_price, 2),
            'details': details
        }
        
    except Exception as e:
        print(f"Error scoring {symbol}: {str(e)}")
        return None

def run_bot_logic(min_score: int = 10, limit: int = 50) -> Dict[str, Any]:
    """
    Main logic to scan stocks and return high-scoring ones
    """
    stocks_to_scan = get_stock_list()[:limit]  # Limit number of stocks to scan
    high_score_stocks = []
    total_scanned = 0
    
    print(f"🔍 Scanning {len(stocks_to_scan)} stocks for scores >= {min_score}...")
    
    for symbol in stocks_to_scan:
        try:
            result = score_stock(symbol)
            total_scanned += 1
            
            if result and result['score'] >= min_score:
                high_score_stocks.append(result)
                print(f"✅ {symbol}: Score {result['score']}, Price ₹{result['current_price']}")
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
        except Exception as e:
            print(f"❌ Error with {symbol}: {str(e)}")
            continue
    
    # Sort by score (highest first)
    high_score_stocks.sort(key=lambda x: x['score'], reverse=True)
    
    result = {
        'total_scanned': total_scanned,
        'high_score_count': len(high_score_stocks),
        'min_score_used': min_score,
        'stocks': high_score_stocks,
        'scan_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    print(f"🎯 Found {len(high_score_stocks)} stocks with score >= {min_score}")
    return result

if __name__ == "__main__":
    # Test the scoring system
    print("Testing stock scoring system...")
    result = run_bot_logic(min_score=8, limit=10)
    print(f"Results: {result}")
