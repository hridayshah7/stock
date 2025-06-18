import re
import logging
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.helpers import escape_markdown
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BOT_TOKEN = "8166189610:AAEGeti-NF2BNYd68qY0CEysDKUz6xNNIpg"
AUTHORIZED_USER_ID = 945542175
MAX_WORKERS = 5  # Limit concurrent requests to prevent rate limiting

# Cache for stock data to reduce API calls
DATA_CACHE = {}
CACHE_EXPIRY = 3600  # 1 hour cache expiry

# Improved list of symbols (Nifty 500 stocks)
SYMBOLS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "AMBUJACEM.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ_AUTO.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "BHARTIARTL.NS",
    "BPCL.NS", "BRITANNIA.NS", "CIPLA.NS", "COALINDIA.NS", "DIVISLAB.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS", "HDFCBANK.NS",
    "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS", "HINDUNILVR.NS", "ICICIBANK.NS",
    "IDFCFIRSTB.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS", "JSWSTEEL.NS",
    "KOTAKBANK.NS", "LT.NS", "LTIM.NS", "LTTS.NS", "M&M.NS",
    "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "PIDILITIND.NS",
    "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHREECEM.NS",
    "SUNPHARMA.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS",
    "TECHM.NS", "TITAN.NS", "ULTRACEMCO.NS", "UPL.NS", "WIPRO.NS",
    "ADANIGREEN.NS", "ADANITRANS.NS", "ATGL.NS", "BANKBARODA.NS", "BEL.NS",
    "CHOLAFIN.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "ICICIPRULI.NS", "INDIGO.NS", "JINDALSTEL.NS", "NMDC.NS",
    "PAYTM.NS", "PETRONET.NS", "PNB.NS", "RECLTD.NS", "SAIL.NS",
    "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TRENT.NS", "TVSMOTOR.NS",
    "VEDL.NS", "VOLTAS.NS", "YESBANK.NS", "ZOMATO.NS", "IRCTC.NS",
    "MUTHOOTFIN.NS", "ABB.NS", "CONCOR.NS", "CROMPTON.NS", "NAUKRI.NS",
    "PAGEIND.NS", "BANDHANBNK.NS", "BHEL.NS", "BIOCON.NS", "COLPAL.NS"
]

# Refined weight scoring system with additional indicators
weights = {
    # Momentum indicators (35%)
    "price_change": 3.5,         # Recent price change
    "volume_spike": 3.0,         # Volume increase
    "rsi_signal": 2.5,           # RSI signal (oversold/overbought)
    "macd_signal": 3.0,          # MACD crossing
    "cci_signal": 2.0,           # CCI signal
    
    # Trend indicators (30%)
    "above_ma50": 2.5,           # Price above 50-day MA
    "ma20_above_ma50": 2.5,      # 20-day MA above 50-day MA (golden cross)
    "adx_trend": 2.5,            # ADX trend strength
    "above_avg_close": 1.5,      # Current close above average
    
    # Price action (15%)
    "green_candle": 1.5,         # Closing price above opening price
    "near_high": 2.0,            # Close near day's high
    "higher_high": 1.5,          # Higher high than previous day
    "higher_low": 1.5,           # Higher low than previous day
    "engulfing_bullish": 2.0,    # Bullish engulfing pattern
    
    # Support/Resistance (15%)
    "bounced_support": 3.0,      # Bounced off support
    "broke_resistance": 3.0,     # Broke through resistance
    "fibonacci_support": 2.0,    # At Fibonacci support
    
    # Volatility (5%)
    "low_volatility": 1.0,       # Low volatility (ATR/price ratio)
    "bollinger_signal": 2.0,     # Bollinger Band signal
}

# Market timing factors - adjusted according to market session
market_timing_weights = {
    "morning": {
        "volume_spike": 1.2,      # Higher importance for volume in the morning
        "green_candle": 1.5       # Price action more important in morning
    },
    "afternoon": {
        "ma20_above_ma50": 1.3,   # Trend more important in afternoon
        "adx_trend": 1.3          # Trend strength more important in afternoon
    },
    "close": {
        "near_high": 1.5,         # Closing near high is significant at end of day
        "rsi_signal": 1.3         # Momentum important at close
    }
}

# Sectoral weights - different sectors respond differently to indicators
sector_weights = {
    "IT": {
        "macd_signal": 1.2,       # IT stocks more responsive to MACD
        "rsi_signal": 1.2         # IT stocks more responsive to RSI
    },
    "BANKING": {
        "volume_spike": 1.3,      # Banks more responsive to volume
        "fibonacci_support": 1.3  # Banks respect technical levels more
    },
    "PHARMA": {
        "bollinger_signal": 1.3,  # Pharma stocks more range-bound
        "low_volatility": 1.3     # Low volatility better for pharma
    }
}

# Sector mapping for some key stocks
stock_sectors = {
    "TCS.NS": "IT", "INFY.NS": "IT", "WIPRO.NS": "IT", "HCLTECH.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "BANKING", "ICICIBANK.NS": "BANKING", "SBIN.NS": "BANKING", "AXISBANK.NS": "BANKING", "KOTAKBANK.NS": "BANKING",
    "SUNPHARMA.NS": "PHARMA", "DRREDDY.NS": "PHARMA", "CIPLA.NS": "PHARMA", "DIVISLAB.NS": "PHARMA", "BIOCON.NS": "PHARMA"
}

# Smart fetch with caching for reducing API calls
@lru_cache(maxsize=128)
def fetch_stock_data(symbol, period='100d', interval='1d', retry_count=3):
    """Fetch stock data with caching and retries"""
    cache_key = f"{symbol}_{period}_{interval}"
    
    # Check cache first
    if cache_key in DATA_CACHE:
        cache_entry = DATA_CACHE[cache_key]
        current_time = time.time()
        
        # Return cached data if not expired
        if current_time - cache_entry['timestamp'] < CACHE_EXPIRY:
            logger.info(f"Using cached data for {symbol}")
            return cache_entry['data']
    
    # If not in cache or expired, fetch new data
    for attempt in range(retry_count):
        try:
            logger.info(f"Fetching fresh data for {symbol} (attempt {attempt+1})")
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
            
            if df is not None and not df.empty:
                # Store in cache
                DATA_CACHE[cache_key] = {
                    'data': df,
                    'timestamp': time.time()
                }
                return df
            
            time.sleep(1)  # Small delay before retry
        except Exception as e:
            logger.error(f"Attempt {attempt+1} failed for {symbol}: {e}")
            time.sleep(2)  # Longer delay after error
    
    return None

# Technical analysis functions - optimized for performance
def calculate_rsi(data, window=14):
    """Calculate RSI (Relative Strength Index) - optimized"""
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    
    # Use exponential moving average for more responsive RSI
    ema_up = up.ewm(com=window-1, adjust=False).mean()
    ema_down = down.ewm(com=window-1, adjust=False).mean()
    
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, rolling_mean, lower_band

def calculate_atr(data, window=14):
    """Calculate ATR (Average True Range)"""
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_cci(data, window=20):
    """Calculate Commodity Channel Index"""
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    ma_tp = typical_price.rolling(window=window).mean()
    mean_deviation = abs(typical_price - ma_tp).rolling(window=window).mean()
    cci = (typical_price - ma_tp) / (0.015 * mean_deviation)
    return cci

def calculate_adx(data, window=14):
    """Calculate Average Directional Index for trend strength"""
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    # Calculate +DI and -DI
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    # Only keep values where +DM > -DM for +DI, and -DM > +DM for -DI
    plus_dm_mask = (plus_dm > minus_dm)
    minus_dm_mask = (minus_dm > plus_dm)
    plus_dm[~plus_dm_mask] = 0
    minus_dm[~minus_dm_mask] = 0
    
    # Calculate true range
    tr = calculate_atr(data, window)
    
    # Smooth with EMA
    plus_di = 100 * plus_dm.ewm(span=window, adjust=False).mean() / tr
    minus_di = 100 * minus_dm.ewm(span=window, adjust=False).mean() / tr
    
    # Calculate directional index
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Average directional index is smoothed DX
    adx = dx.ewm(span=window, adjust=False).mean()
    
    return adx, plus_di, minus_di

def find_support_resistance(data, window=20):
    """Find support and resistance levels using multiple techniques"""
    # Method 1: Rolling min/max
    support_sm = data['Low'].rolling(window=window).min()
    resistance_sm = data['High'].rolling(window=window).max()
    
    # Method 2: Local minima/maxima for recent periods
    pivot_points = []
    for i in range(5, len(data)-5):
        # Check if this is a local minimum
        if all(data['Low'].iloc[i] <= data['Low'].iloc[i-j] for j in range(1, 5)) and \
           all(data['Low'].iloc[i] <= data['Low'].iloc[i+j] for j in range(1, 5)):
            pivot_points.append((i, data['Low'].iloc[i], 'support'))
        
        # Check if this is a local maximum
        if all(data['High'].iloc[i] >= data['High'].iloc[i-j] for j in range(1, 5)) and \
           all(data['High'].iloc[i] >= data['High'].iloc[i+j] for j in range(1, 5)):
            pivot_points.append((i, data['High'].iloc[i], 'resistance'))
    
    # Create dynamic DataFrame for support/resistance
    levels = pd.Series(index=data.index, dtype=float)
    for idx, level, level_type in pivot_points:
        for i in range(idx, len(data)):
            if level_type == 'support':
                if pd.isna(levels.iloc[i]) or level < levels.iloc[i]:
                    levels.iloc[i] = level
            else:  # resistance
                if pd.isna(levels.iloc[i]) or level > levels.iloc[i]:
                    levels.iloc[i] = level
    
    # Combine methods
    support = support_sm.combine_first(levels.where(levels < data['Close']))
    resistance = resistance_sm.combine_first(levels.where(levels > data['Close']))
    
    return support, resistance

def calculate_fibonacci_levels(data, trend='uptrend'):
    """Calculate Fibonacci retracement levels"""
    if trend == 'uptrend':
        high = data['High'].max()
        low = data['Low'].min()
    else:  # downtrend
        high = data['High'].iloc[-20:].max()  # Recent high for downtrend
        low = data['Low'].min()
    
    diff = high - low
    
    levels = {
        0.0: high,
        0.236: high - 0.236 * diff,
        0.382: high - 0.382 * diff,
        0.5: high - 0.5 * diff,
        0.618: high - 0.618 * diff,
        0.786: high - 0.786 * diff,
        1.0: low
    }
    
    return levels

def detect_candlestick_patterns(data):
    """Detect common candlestick patterns"""
    patterns = {}
    
    # Get relevant data
    open_prices = data['Open']
    close_prices = data['Close']
    high_prices = data['High']
    low_prices = data['Low']
    
    # Check for doji (open and close are very close)
    doji_threshold = 0.001  # 0.1% difference
    doji = abs(open_prices - close_prices) < (open_prices * doji_threshold)
    
    # Bullish engulfing (current candle engulfs previous)
    bullish_engulfing = (open_prices < close_prices) & \
                        (open_prices.shift(1) > close_prices.shift(1)) & \
                        (open_prices < close_prices.shift(1)) & \
                        (close_prices > open_prices.shift(1))
    
    # Bearish engulfing
    bearish_engulfing = (open_prices > close_prices) & \
                         (open_prices.shift(1) < close_prices.shift(1)) & \
                         (open_prices > close_prices.shift(1)) & \
                         (close_prices < open_prices.shift(1))
    
    # Hammer (bullish reversal pattern)
    body_size = abs(close_prices - open_prices)
    lower_shadow = np.minimum(open_prices, close_prices) - low_prices
    upper_shadow = high_prices - np.maximum(open_prices, close_prices)
    
    hammer = (lower_shadow > 2 * body_size) & \
              (upper_shadow < 0.5 * body_size) & \
              (body_size > 0)  # Positive body
    
    patterns['doji'] = doji
    patterns['bullish_engulfing'] = bullish_engulfing
    patterns['bearish_engulfing'] = bearish_engulfing
    patterns['hammer'] = hammer
    
    return patterns

def calculate_detailed_score(df, symbol=None):
    """Calculate a comprehensive score with buy/sell signals and target prices"""
    try:
        if df is None or df.empty or len(df) < 30:  # Need more data for reliable signals
            return {
                'score': -1,
                'action': 'INSUFFICIENT_DATA',
                'signals': {},
                'entry': None,
                'stop_loss': None,
                'target': None,
                'risk_reward': None
            }

        # Fix for multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df = df.copy()
            df.columns = df.columns.get_level_values(0)
        
        # Make sure we have the required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns. Available: {df.columns}")
            return {
                'score': -1,
                'action': 'MISSING_DATA',
                'signals': {},
                'entry': None,
                'stop_loss': None,
                'target': None,
                'risk_reward': None
            }
        
        # Create a copy of dataframe to avoid modifications on original
        df_analysis = df.copy()
        
        # Calculate technical indicators
        df_analysis['MA20'] = df_analysis['Close'].rolling(window=20).mean()
        df_analysis['MA50'] = df_analysis['Close'].rolling(window=50).mean()
        
        # Calculate RSI - needs to be a Series for proper assignment
        rsi_values = calculate_rsi(df_analysis['Close'])
        df_analysis['RSI'] = rsi_values
        
        # MACD Calculation - now handling Series properly
        close_series = df_analysis['Close']
        macd_line, signal_line, histogram = calculate_macd(close_series)
        
        # Assign MACD components as separate columns
        df_analysis['MACD'] = macd_line
        df_analysis['MACD_Signal'] = signal_line
        df_analysis['MACD_Hist'] = histogram
        
        # Bollinger Bands calculation
        upper_bb, middle_bb, lower_bb = calculate_bollinger_bands(df_analysis['Close'])
        df_analysis['Upper_BB'] = upper_bb
        df_analysis['Middle_BB'] = middle_bb
        df_analysis['Lower_BB'] = lower_bb
        
        # Other indicators
        df_analysis['ATR'] = calculate_atr(df_analysis)
        df_analysis['CCI'] = calculate_cci(df_analysis)
        
        # ADX calculation returns multiple values
        adx_values, plus_di, minus_di = calculate_adx(df_analysis)
        df_analysis['ADX'] = adx_values
        df_analysis['Plus_DI'] = plus_di
        df_analysis['Minus_DI'] = minus_di
        
        # Support and resistance
        support, resistance = find_support_resistance(df_analysis)
        df_analysis['Support'] = support
        df_analysis['Resistance'] = resistance
        
        # Get candlestick patterns
        patterns = detect_candlestick_patterns(df_analysis)
        for pattern_name, pattern_data in patterns.items():
            df_analysis[f'Pattern_{pattern_name}'] = pattern_data
        
        # Get Fibonacci levels for current trend
        if df_analysis['Close'].iloc[-1] > df_analysis['Close'].iloc[-20]:  # Uptrend
            fib_levels = calculate_fibonacci_levels(df_analysis, 'uptrend')
        else:  # Downtrend
            fib_levels = calculate_fibonacci_levels(df_analysis, 'downtrend')
        
        # Get the latest data points
        latest = df_analysis.iloc[-1]
        prev = df_analysis.iloc[-2]
        
        # Initialize signals dict and score
        signals = {}
        score = 0
        
        # Apply sector-specific weight adjustments if applicable
        adjusted_weights = weights.copy()
        if symbol in stock_sectors:
            sector = stock_sectors[symbol]
            for indicator, weight_mult in sector_weights.get(sector, {}).items():
                if indicator in adjusted_weights:
                    adjusted_weights[indicator] *= weight_mult
        
        # Apply time-of-day weight adjustments
        current_hour = datetime.now().hour
        if 9 <= current_hour < 11:  # Morning session
            timing = "morning"
        elif 11 <= current_hour < 14:  # Afternoon session
            timing = "afternoon"
        else:  # Closing session
            timing = "close"
            
        for indicator, weight_mult in market_timing_weights.get(timing, {}).items():
            if indicator in adjusted_weights:
                adjusted_weights[indicator] *= weight_mult
        
        # 1. Recent Price Change
        price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        signals['price_change'] = price_change
        if price_change > 2.0:
            score += adjusted_weights["price_change"] * 1.2  # Extra weight for strong moves
            signals['price_change_signal'] = 'STRONG_BULLISH'
        elif price_change > 1.0:
            score += adjusted_weights["price_change"]
            signals['price_change_signal'] = 'BULLISH'
        elif price_change < -2.0:
            score -= adjusted_weights["price_change"] * 1.2
            signals['price_change_signal'] = 'STRONG_BEARISH'
        elif price_change < -1.0:
            score -= adjusted_weights["price_change"]
            signals['price_change_signal'] = 'BEARISH'
        else:
            signals['price_change_signal'] = 'NEUTRAL'
        
        # 2. Volume Analysis
        avg_vol = df_analysis['Volume'][-10:-1].mean()  # Last 9 days avg
        vol_change = (latest['Volume'] - avg_vol) / avg_vol * 100
        signals['volume_change'] = vol_change
        
        if vol_change > 100 and price_change > 1.5:  # Explosive volume on up move
            score += adjusted_weights["volume_spike"] * 1.5
            signals['volume_signal'] = 'STRONG_BULLISH'
        elif vol_change > 50 and price_change > 0:
            score += adjusted_weights["volume_spike"]
            signals['volume_signal'] = 'BULLISH'
        elif vol_change > 100 and price_change < -1.5:
            score -= adjusted_weights["volume_spike"] * 1.5
            signals['volume_signal'] = 'STRONG_BEARISH'
        elif vol_change > 50 and price_change < 0:
            score -= adjusted_weights["volume_spike"]
            signals['volume_signal'] = 'BEARISH'
        else:
            signals['volume_signal'] = 'NEUTRAL'
        
        # 3. RSI Analysis
        rsi = latest['RSI']
        signals['rsi'] = rsi
        
        if rsi < 30:
            score += adjusted_weights["rsi_signal"]  # Oversold, potential buy
            signals['rsi_signal'] = 'OVERSOLD'
        elif rsi > 70:
            score -= adjusted_weights["rsi_signal"]  # Overbought, potential sell
            signals['rsi_signal'] = 'OVERBOUGHT'
        elif rsi > 50 and rsi < 70:
            score += adjusted_weights["rsi_signal"] * 0.5  # Bullish momentum
            signals['rsi_signal'] = 'BULLISH'
        elif rsi > 30 and rsi < 50:
            score -= adjusted_weights["rsi_signal"] * 0.5  # Bearish momentum
            signals['rsi_signal'] = 'BEARISH'
        else:
            signals['rsi_signal'] = 'NEUTRAL'
        
        # 4. MACD Analysis
        macd = latest['MACD']
        macd_signal = latest['MACD_Signal']
        macd_hist = latest['MACD_Hist']
        
        signals['macd'] = macd
        
        # MACD crossing signal line from below (bullish)
        if macd > macd_signal and prev['MACD'] < prev['MACD_Signal']:
            score += adjusted_weights["macd_signal"] * 1.2
            signals['macd_signal'] = 'STRONG_BULLISH'
        # MACD above signal line (bullish)
        elif macd > macd_signal:
            score += adjusted_weights["macd_signal"]
            signals['macd_signal'] = 'BULLISH'
        # MACD crossing signal line from above (bearish)
        elif macd < macd_signal and prev['MACD'] > prev['MACD_Signal']:
            score -= adjusted_weights["macd_signal"] * 1.2
            signals['macd_signal'] = 'STRONG_BEARISH'
        # MACD below signal line (bearish)
        elif macd < macd_signal:
            score -= adjusted_weights["macd_signal"]
            signals['macd_signal'] = 'BEARISH'
        else:
            signals['macd_signal'] = 'NEUTRAL'
        
        # 5. Trend Analysis - Moving Averages
        # Price above 50-day MA (bullish)
        if latest['Close'] > latest['MA50']:
            score += adjusted_weights["above_ma50"]
            signals['above_ma50'] = True
        else:
            signals['above_ma50'] = False
        
        # 20-day MA crossing above 50-day MA (golden cross - bullish)
        if latest['MA20'] > latest['MA50'] and prev['MA20'] <= prev['MA50']:
            score += adjusted_weights["ma20_above_ma50"] * 1.5
            signals['ma_crossover'] = 'GOLDEN_CROSS'
        # 20-day MA crossing below 50-day MA (death cross - bearish)
        elif latest['MA20'] < latest['MA50'] and prev['MA20'] >= prev['MA50']:
            score -= adjusted_weights["ma20_above_ma50"] * 1.5
            signals['ma_crossover'] = 'DEATH_CROSS'
        # 20-day MA above 50-day MA (bullish trend)
        elif latest['MA20'] > latest['MA50']:
            score += adjusted_weights["ma20_above_ma50"]
            signals['ma_crossover'] = 'BULLISH_TREND'
        # 20-day MA below 50-day MA (bearish trend)
        else:
            score -= adjusted_weights["ma20_above_ma50"]
            signals['ma_crossover'] = 'BEARISH_TREND'
        
        # 6. Price Action - Candlestick
        # Green candle (close > open)
        if latest['Close'] > latest['Open']:
            score += adjusted_weights["green_candle"]
            signals['candle'] = 'GREEN'
        else:
            signals['candle'] = 'RED'
        
        # Close near day's high
        high_ratio = (latest['Close'] - latest['Low']) / (latest['High'] - latest['Low'])
        if high_ratio > 0.8:  # Close in upper 20% of day's range
            score += adjusted_weights["near_high"]
            signals['close_position'] = 'NEAR_HIGH'
        elif high_ratio < 0.2:  # Close in lower 20% of day's range
            score -= adjusted_weights["near_high"]
            signals['close_position'] = 'NEAR_LOW'
        
        # Higher high and higher low (bullish)
        if latest['High'] > prev['High'] and latest['Low'] > prev['Low']:
            score += adjusted_weights["higher_high"] + adjusted_weights["higher_low"]
            signals['price_pattern'] = 'HIGHER_HIGH_HIGHER_LOW'
        # Lower high and lower low (bearish)
        elif latest['High'] < prev['High'] and latest['Low'] < prev['Low']:
            score -= adjusted_weights["higher_high"] + adjusted_weights["higher_low"]
            signals['price_pattern'] = 'LOWER_HIGH_LOWER_LOW'
        
        # Bullish engulfing pattern
        if latest['Pattern_bullish_engulfing']:
            score += adjusted_weights["engulfing_bullish"]
            signals['candlestick_pattern'] = 'BULLISH_ENGULFING'
        # Bearish engulfing pattern
        elif latest['Pattern_bearish_engulfing']:
            score -= adjusted_weights["engulfing_bullish"]
            signals['candlestick_pattern'] = 'BEARISH_ENGULFING'
        
        # 7. Support/Resistance Analysis
        current_price = latest['Close']
        
        # Check if price is near support
        if not pd.isna(latest['Support']) and abs(current_price - latest['Support']) / current_price < 0.03:
            if current_price > latest['Support'] and prev['Close'] < prev['Support']:
                # Bounced off support
                score += adjusted_weights["bounced_support"] * 1.2
                signals['support_resistance'] = 'BOUNCED_SUPPORT'
            elif current_price > latest['Support']:
                # Near support
                score += adjusted_weights["bounced_support"] * 0.7
                signals['support_resistance'] = 'NEAR_SUPPORT'
        
        # Check if price is breaking through resistance
        if not pd.isna(latest['Resistance']) and abs(current_price - latest['Resistance']) / current_price < 0.03:
            if current_price > latest['Resistance'] and prev['Close'] < prev['Resistance']:
                # Breaking resistance
                score += adjusted_weights["broke_resistance"] * 1.5
                signals['support_resistance'] = 'BROKE_RESISTANCE'
            elif current_price < latest['Resistance']:
                # Near resistance
                score -= adjusted_weights["broke_resistance"] * 0.7
                signals['support_resistance'] = 'NEAR_RESISTANCE'
        
        # 8. Check Fibonacci levels
        current_fib_level = None
        for level, price in fib_levels.items():
            if abs(current_price - price) / current_price < 0.03:
                current_fib_level = level
                break
        
        if current_fib_level is not None:
            # At key Fibonacci level
            if current_fib_level in [0.382, 0.5, 0.618]:
                if current_price > prev['Close']:  # Price moving up from Fib level
                    score += adjusted_weights["fibonacci_support"]
                    signals['fibonacci'] = f'BOUNCE_FROM_{current_fib_level}'
        
        # 9. ADX - Trend Strength
        adx = latest['ADX']
        signals['adx'] = adx
        
        if adx > 25:  # Strong trend
            if latest['Plus_DI'] > latest['Minus_DI']:  # Bullish trend
                score += adjusted_weights["adx_trend"]
                signals['adx_signal'] = 'STRONG_TREND_BULLISH'
            else:  # Bearish trend
                score -= adjusted_weights["adx_trend"]
                signals['adx_signal'] = 'STRONG_TREND_BEARISH'
        else:
            signals['adx_signal'] = 'WEAK_TREND'
        
        # 10. Bollinger Bands
        bb_width = (latest['Upper_BB'] - latest['Lower_BB']) / latest['Middle_BB']
        signals['bb_width'] = bb_width
        
        if current_price < latest['Lower_BB']:
            score += adjusted_weights["bollinger_signal"]  # Oversold, potential buy
            signals['bollinger_signal'] = 'BELOW_LOWER_BAND'
        elif current_price > latest['Upper_BB']:
            score -= adjusted_weights["bollinger_signal"]  # Overbought, potential sell
            signals['bollinger_signal'] = 'ABOVE_UPPER_BAND'
        elif bb_width < 0.1:  # Narrow bands, potential breakout
            score += adjusted_weights["bollinger_signal"] * 0.5
            signals['bollinger_signal'] = 'SQUEEZE'
        
        # 11. Volatility (ATR/Price ratio)
        volatility = latest['ATR'] / latest['Close']
        signals['volatility'] = volatility
        
        if volatility < 0.01:  # Low volatility
            score += adjusted_weights["low_volatility"]
            signals['volatility_signal'] = 'LOW'
        elif volatility > 0.03:  # High volatility
            score -= adjusted_weights["low_volatility"] * 0.5
            signals['volatility_signal'] = 'HIGH'
        
        # 12. CCI (Commodity Channel Index)
        cci = latest['CCI']
        signals['cci'] = cci
        
        if cci < -100:  # Oversold
            score += adjusted_weights["cci_signal"]
            signals['cci_signal'] = 'OVERSOLD'
        elif cci > 100:  # Overbought
            score -= adjusted_weights["cci_signal"]
            signals['cci_signal'] = 'OVERBOUGHT'
        
        # 13. Above average close price
        avg_close = df_analysis['Close'][-20:].mean()
        if latest['Close'] > avg_close:
            score += adjusted_weights["above_avg_close"]
            signals['above_avg_close'] = True
        else:
            signals['above_avg_close'] = False
        
        # Calculate target and stop loss
        atr = latest['ATR']
        entry_price = latest['Close']
        
        # Determine action based on score
        if score > 15:
            action = 'STRONG_BUY'
            # More aggressive targets for strong signals
            stop_loss = entry_price - (2.0 * atr)
            target = entry_price + (3.5 * atr)
        elif score > 8:
            action = 'BUY'
            stop_loss = entry_price - (1.5 * atr)
            target = entry_price + (3.0 * atr)
        elif score > 3:
            action = 'WEAK_BUY'
            stop_loss = entry_price - (1.5 * atr)
            target = entry_price + (2.5 * atr)
        elif score < -15:
            action = 'STRONG_SELL'
            stop_loss = entry_price + (2.0 * atr)
            target = entry_price - (3.5 * atr)
        elif score < -8:
            action = 'SELL'
            stop_loss = entry_price + (1.5 * atr)
            target = entry_price - (3.0 * atr)
        elif score < -3:
            action = 'WEAK_SELL'
            stop_loss = entry_price + (1.5 * atr)
            target = entry_price - (2.5 * atr)
        else:
            action = 'NEUTRAL'
            stop_loss = None
            target = None
        
        # Calculate risk-reward ratio if applicable
        if stop_loss is not None and target is not None:
            risk = abs(entry_price - stop_loss)
            reward = abs(target - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
        else:
            risk_reward = None
        
        # Return comprehensive analysis
        return {
            'score': round(score, 2),
            'action': action,
            'signals': signals,
            'entry': round(entry_price, 2) if entry_price is not None else None,
            'stop_loss': round(stop_loss, 2) if stop_loss is not None else None,
            'target': round(target, 2) if target is not None else None,
            'risk_reward': round(risk_reward, 2) if risk_reward is not None else None
        }
    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'score': -1,
            'action': 'ERROR',
            'signals': {'error': str(e)},
            'entry': None,
            'stop_loss': None,
            'target': None,
            'risk_reward': None
        }

# Generate detailed report for a stock
def generate_report(symbol):
    """Generate a detailed analysis report for a stock"""
    try:
        # Get data with different timeframes
        data_daily = fetch_stock_data(symbol, period='100d', interval='1d')
        data_weekly = fetch_stock_data(symbol, period='200d', interval='1wk')
        
        if data_daily is None or data_daily.empty:
            return f"⚠️ Could not retrieve data for {symbol}"
        
        # Get basic info
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get('shortName', symbol)
        sector = info.get('sector', 'Unknown')
        industry = info.get('industry', 'Unknown')
        
        # Calculate scores
        daily_score = calculate_detailed_score(data_daily, symbol)
        weekly_score = calculate_detailed_score(data_weekly, symbol)
        
        # Current price and day change
        current_price = data_daily['Close'].iloc[-1]
        prev_close = data_daily['Close'].iloc[-2]
        day_change = ((current_price - prev_close) / prev_close) * 100
        
        # Format the report with markdown
        report = f"*📊 Analysis for {company_name} ({symbol})*\n\n"
        report += f"*Current Price:* ₹{current_price:.2f} ({day_change:.2f}%)\n"
        report += f"*Sector:* {sector}\n"
        report += f"*Industry:* {industry}\n\n"
        
        # Daily signals
        report += f"*Daily Analysis:*\n"
        report += f"Score: {daily_score['score']} - *{daily_score['action']}*\n"
        
        if daily_score['action'] != 'NEUTRAL' and daily_score['action'] != 'ERROR':
            report += f"Entry: ₹{daily_score['entry']}\n"
            if daily_score['stop_loss']:
                report += f"Stop Loss: ₹{daily_score['stop_loss']}\n"
            if daily_score['target']:
                report += f"Target: ₹{daily_score['target']}\n"
            if daily_score['risk_reward']:
                report += f"Risk/Reward: {daily_score['risk_reward']}\n"
        
        # Weekly context
        report += f"\n*Weekly Trend:*\n"
        report += f"Score: {weekly_score['score']} - *{weekly_score['action']}*\n\n"
        
        # Key signals
        report += "*Key Signals:*\n"
        signals = daily_score['signals']
        
        # RSI
        if 'rsi_signal' in signals:
            report += f"• RSI ({signals.get('rsi', 'N/A'):.1f}): {signals['rsi_signal']}\n"
        
        # MACD
        if 'macd_signal' in signals:
            report += f"• MACD: {signals['macd_signal']}\n"
        
        # MA Crossover
        if 'ma_crossover' in signals:
            report += f"• MA Status: {signals['ma_crossover']}\n"
        
        # Volume
        if 'volume_signal' in signals:
            report += f"• Volume: {signals['volume_signal']}\n"
        
        # Support/Resistance
        if 'support_resistance' in signals:
            report += f"• S/R: {signals['support_resistance']}\n"
        
        # ADX
        if 'adx_signal' in signals:
            report += f"• ADX ({signals.get('adx', 'N/A'):.1f}): {signals['adx_signal']}\n"
        
        # Candlestick pattern
        if 'candlestick_pattern' in signals:
            report += f"• Pattern: {signals['candlestick_pattern']}\n"
        
        return report
    
    except Exception as e:
        logger.error(f"Error generating report for {symbol}: {e}")
        return f"⚠️ Error analyzing {symbol}: {str(e)}"

# Function to scan all stocks
async def scan_stocks():
    """Scan all stocks for trading opportunities"""
    logger.info("Starting market scan...")
    
    buy_signals = []
    sell_signals = []
    
    # Limit concurrent API calls with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Process stocks in batches to prevent overloading
        batch_size = 10
        for i in range(0, len(SYMBOLS), batch_size):
            batch = SYMBOLS[i:i+batch_size]
            
            # Create tasks for each symbol in the batch
            futures = []
            for symbol in batch:
                future = executor.submit(fetch_stock_data, symbol, '100d', '1d')
                futures.append((symbol, future))
            
            # Process results as they complete
            for symbol, future in futures:
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        score_data = calculate_detailed_score(data, symbol)
                        
                        if score_data['score'] > 8:  # Strong buy signals
                            buy_signals.append((symbol, score_data))
                        elif score_data['score'] < -8:  # Strong sell signals
                            sell_signals.append((symbol, score_data))
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
            
            # Small delay between batches to prevent API rate limits
            await asyncio.sleep(1)
    
    # Sort signals by score strength
    buy_signals.sort(key=lambda x: x[1]['score'], reverse=True)
    sell_signals.sort(key=lambda x: x[1]['score'])
    
    return buy_signals, sell_signals

# Telegram commands and handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user_id = update.effective_user.id
    
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return
    
    keyboard = [
        [
            InlineKeyboardButton("Scan Market", callback_data='scan'),
            InlineKeyboardButton("Top Stocks", callback_data='top')
        ],
        [
            InlineKeyboardButton("Buy Signals", callback_data='buy'),
            InlineKeyboardButton("Sell Signals", callback_data='sell')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Welcome to the Stock Analysis Bot! Choose an option:",
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if query.data == 'scan':
        await query.edit_message_text(text="Scanning market... This may take a minute.")
        buy_signals, sell_signals = await scan_stocks()
        
        response = "*Market Scan Results*\n\n"
        response += f"Found {len(buy_signals)} buy signals and {len(sell_signals)} sell signals.\n\n"
        
        # Top 5 Buy Signals
        if buy_signals:
            response += "*Top Buy Signals:*\n"
            for i, (symbol, data) in enumerate(buy_signals[:5]):
                stock = yf.Ticker(symbol)
                name = stock.info.get('shortName', symbol)
                response += f"{i+1}. {name} ({symbol}) - Score: {data['score']}\n"
        
        # Top 5 Sell Signals
        if sell_signals:
            response += "\n*Top Sell Signals:*\n"
            for i, (symbol, data) in enumerate(sell_signals[:5]):
                stock = yf.Ticker(symbol)
                name = stock.info.get('shortName', symbol)
                response += f"{i+1}. {name} ({symbol}) - Score: {data['score']}\n"
        
        await query.edit_message_text(text=response, parse_mode='Markdown')
    
    elif query.data == 'top':
        await query.edit_message_text(text="Fetching top stocks... Please wait.")
        
        # Get top movers
        gainers = []
        losers = []
        volume_leaders = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            for symbol in SYMBOLS[:30]:  # Limit to first 30 stocks for performance
                future = executor.submit(fetch_stock_data, symbol, '5d', '1d')
                futures.append((symbol, future))
            
            for symbol, future in futures:
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        last_price = data['Close'].iloc[-1]
                        prev_price = data['Close'].iloc[-2]
                        change_pct = ((last_price - prev_price) / prev_price) * 100
                        volume = data['Volume'].iloc[-1]
                        
                        stock_info = {
                            'symbol': symbol,
                            'price': last_price,
                            'change_pct': change_pct,
                            'volume': volume
                        }
                        
                        gainers.append(stock_info)
                        losers.append(stock_info)
                        volume_leaders.append(stock_info)
                except Exception as e:
                    logger.error(f"Error processing top movers for {symbol}: {e}")
        
        # Sort lists
        gainers.sort(key=lambda x: x['change_pct'], reverse=True)
        losers.sort(key=lambda x: x['change_pct'])
        volume_leaders.sort(key=lambda x: x['volume'], reverse=True)
        
        response = "*Today's Market Leaders*\n\n"
        
        # Top Gainers
        response += "*Top Gainers:*\n"
        for i, stock in enumerate(gainers[:5]):
            response += f"{i+1}. {stock['symbol']} - ₹{stock['price']:.2f} ({stock['change_pct']:.2f}%)\n"
        
        # Top Losers
        response += "\n*Top Losers:*\n"
        for i, stock in enumerate(losers[:5]):
            response += f"{i+1}. {stock['symbol']} - ₹{stock['price']:.2f} ({stock['change_pct']:.2f}%)\n"
        
        # Volume Leaders
        response += "\n*Volume Leaders:*\n"
        for i, stock in enumerate(volume_leaders[:5]):
            vol_in_cr = stock['volume'] / 10000000  # Convert to Crores for Indian market
            response += f"{i+1}. {stock['symbol']} - ₹{stock['price']:.2f} (Vol: {vol_in_cr:.2f}Cr)\n"
        
        await query.edit_message_text(text=response, parse_mode='Markdown')
    
    elif query.data == 'buy':
        await query.edit_message_text(text="Fetching buy signals... Please wait.")
        buy_signals, _ = await scan_stocks()
        
        if not buy_signals:
            await query.edit_message_text(text="No strong buy signals found today.")
            return
        
        response = "*Top Buy Signals*\n\n"
        for i, (symbol, data) in enumerate(buy_signals[:10]):
            stock = yf.Ticker(symbol)
            name = stock.info.get('shortName', symbol)
            
            response += f"*{i+1}. {name} ({symbol})*\n"
            response += f"Score: {data['score']} - *{data['action']}*\n"
            
            if 'entry' in data and data['entry']:
                response += f"Entry: ₹{data['entry']} | "
            if 'stop_loss' in data and data['stop_loss']:
                response += f"SL: ₹{data['stop_loss']} | "
            if 'target' in data and data['target']:
                response += f"Target: ₹{data['target']}\n"
            if 'risk_reward' in data and data['risk_reward']:
                response += f"R/R: {data['risk_reward']}\n"
            
            # Add a key signal
            signals = data['signals']
            if 'rsi_signal' in signals:
                response += f"RSI: {signals['rsi_signal']}"
            elif 'macd_signal' in signals:
                response += f"MACD: {signals['macd_signal']}"
            
            response += "\n\n"
        
        await query.edit_message_text(text=response, parse_mode='Markdown')
    
    elif query.data == 'sell':
        await query.edit_message_text(text="Fetching sell signals... Please wait.")
        _, sell_signals = await scan_stocks()
        
        if not sell_signals:
            await query.edit_message_text(text="No strong sell signals found today.")
            return
        
        response = "*Top Sell Signals*\n\n"
        for i, (symbol, data) in enumerate(sell_signals[:10]):
            stock = yf.Ticker(symbol)
            name = stock.info.get('shortName', symbol)
            
            response += f"*{i+1}. {name} ({symbol})*\n"
            response += f"Score: {data['score']} - *{data['action']}*\n"
            
            if 'entry' in data and data['entry']:
                response += f"Entry: ₹{data['entry']} | "
            if 'stop_loss' in data and data['stop_loss']:
                response += f"SL: ₹{data['stop_loss']} | "
            if 'target' in data and data['target']:
                response += f"Target: ₹{data['target']}\n"
            if 'risk_reward' in data and data['risk_reward']:
                response += f"R/R: {data['risk_reward']}\n"
            
            # Add a key signal
            signals = data['signals']
            if 'rsi_signal' in signals:
                response += f"RSI: {signals['rsi_signal']}"
            elif 'macd_signal' in signals:
                response += f"MACD: {signals['macd_signal']}"
            
            response += "\n\n"
        
        await query.edit_message_text(text=response, parse_mode='Markdown')

async def analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a specific stock when the command /analyze is issued."""
    user_id = update.effective_user.id
    
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return
    
    if not context.args or len(context.args) == 0:
        await update.message.reply_text("Please provide a stock symbol. Example: /analyze RELIANCE.NS")
        return
    
    symbol = context.args[0].upper()
    
    # Check if the symbol should have .NS suffix
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        # Add .NS by default for Indian stocks
        symbol = f"{symbol}.NS"
    
    await update.message.reply_text(f"Analyzing {symbol}... Please wait.")
    
    try:
        report = generate_report(symbol)
        await update.message.reply_text(report, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        await update.message.reply_text(f"Error analyzing {symbol}: {str(e)}")

async def schedule_market_updates(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send scheduled market updates."""
    # Only send during market hours (9:15 AM to 3:30 PM IST)
    now = datetime.now()
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    
    if now < market_open or now > market_close:
        logger.info("Market closed, skipping scheduled update")
        return
    
    logger.info("Sending scheduled market update")
    
    try:
        buy_signals, sell_signals = await scan_stocks()
        
        # Send update only if there are signals
        if buy_signals or sell_signals:
            message = "*Scheduled Market Update*\n\n"
            
            # Top 3 Buy Signals
            if buy_signals:
                message += "*Hot Buy Signals:*\n"
                for i, (symbol, data) in enumerate(buy_signals[:3]):
                    stock = yf.Ticker(symbol)
                    name = stock.info.get('shortName', symbol)
                    price = data.get('entry', 'N/A')
                    message += f"{i+1}. {name} ({symbol}) - ₹{price} - Score: {data['score']}\n"
            
            # Top 3 Sell Signals
            if sell_signals:
                message += "\n*Hot Sell Signals:*\n"
                for i, (symbol, data) in enumerate(sell_signals[:3]):
                    stock = yf.Ticker(symbol)
                    name = stock.info.get('shortName', symbol)
                    price = data.get('entry', 'N/A')
                    message += f"{i+1}. {name} ({symbol}) - ₹{price} - Score: {data['score']}\n"
            
            # Send to authorized user
            await context.bot.send_message(
                chat_id=AUTHORIZED_USER_ID,
                text=message,
                parse_mode='Markdown'
            )
    except Exception as e:
        logger.error(f"Error sending scheduled update: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    user_id = update.effective_user.id
    
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return
    
    help_text = """
*Stock Analysis Bot Help*

Available commands:

/start - Start the bot and show main menu
/scan - Scan the market for trading opportunities
/analyze SYMBOL - Analyze a specific stock (e.g., /analyze RELIANCE.NS)
/top - Show today's top movers
/buy - Show top buy signals
/sell - Show top sell signals
/help - Show this help message

*Notes:*
- For Indian stocks, you can use either NSE (.NS) or BSE (.BO) suffix
- Analysis includes technical indicators like RSI, MACD, Moving Averages, etc.
- The bot uses a weighted scoring system to rate stocks
- Updates are scheduled hourly during market hours
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Scan the market when the command /scan is issued."""
    user_id = update.effective_user.id
    
    if user_id != AUTHORIZED_USER_ID:
        await update.message.reply_text("Sorry, you are not authorized to use this bot.")
        return
    
    await update.message.reply_text("Scanning market... This may take a minute.")
    
    try:
        buy_signals, sell_signals = await scan_stocks()
        
        response = "*Market Scan Results*\n\n"
        response += f"Found {len(buy_signals)} buy signals and {len(sell_signals)} sell signals.\n\n"
        
        # Top 5 Buy Signals
        if buy_signals:
            response += "*Top Buy Signals:*\n"
            for i, (symbol, data) in enumerate(buy_signals[:5]):
                stock = yf.Ticker(symbol)
                name = stock.info.get('shortName', symbol)
                response += f"{i+1}. {name} ({symbol}) - Score: {data['score']}\n"
        
        # Top 5 Sell Signals
        if sell_signals:
            response += "\n*Top Sell Signals:*\n"
            for i, (symbol, data) in enumerate(sell_signals[:5]):
                stock = yf.Ticker(symbol)
                name = stock.info.get('shortName', symbol)
                response += f"{i+1}. {name} ({symbol}) - Score: {data['score']}\n"
        
        await update.message.reply_text(response, parse_mode='Markdown')
    except Exception as e:
        logger.error(f"Error scanning stocks: {e}")
        await update.message.reply_text(f"Error scanning stocks: {str(e)}")

async def main() -> None:
    """Start the bot."""
    # Create the Application 
    application = Application.builder().token(BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("analyze", analyze))
    application.add_handler(CommandHandler("scan", scan_command))
    
    # Add callback query handler
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Setup scheduler for market updates - use AsyncIOScheduler with the existing event loop
    scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        schedule_market_updates,
        'interval',
        hours=1,
        start_date='2023-01-01 09:30:00',
        end_date='2030-12-31 15:30:00',
        args=[application]
    )
    scheduler.start()
    
    # Run the bot until the user presses Ctrl-C
    await application.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        # Handle graceful shutdown
        pass