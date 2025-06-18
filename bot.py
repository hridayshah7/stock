import logging
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
BOT_TOKEN = "8166189610:AAEGeti-NF2BNYd68qY0CEysDKUz6xNNIpg"
AUTHORIZED_USER_ID = 945542175
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
    "PAGEIND.NS", "BANDHANBNK.NS", "BHEL.NS", "BIOCON.NS", "COLPAL.NS","DIXON.NS","KAYNES.NS"
]

# Advanced weight scoring system
weights = {
    # Momentum indicators
    "price_change": 3,          # Recent price change
    "volume_spike": 2.5,        # Volume increase
    "rsi_signal": 2,            # RSI signal (oversold/overbought)
    "macd_signal": 2.5,         # MACD crossing
    
    # Trend indicators
    "above_ma50": 2,            # Price above 50-day MA
    "ma20_above_ma50": 2,       # 20-day MA above 50-day MA (golden cross)
    "above_avg_close": 1.5,     # Current close above average
    
    # Price action
    "green_candle": 1.5,        # Closing price above opening price
    "near_high": 2,             # Close near day's high
    "higher_high": 2,           # Higher high than previous day
    "higher_low": 1.5,          # Higher low than previous day
    
    # Support/Resistance
    "bounced_support": 3,       # Bounced off support
    "broke_resistance": 3,      # Broke through resistance
    
    # Volatility
    "low_volatility": 1.5,      # Low volatility (ATR/price ratio)
}

# Technical analysis functions
def calculate_rsi(data, window=14):
    # Calculate RSI (Relative Strength Index)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    # Calculate MACD (Moving Average Convergence Divergence)
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_atr(data, window=14):
    # Calculate ATR (Average True Range)
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def find_support_resistance(data, window=10):
    # Simple support and resistance detection based on recent lows and highs
    support = data['Low'].rolling(window=window).min()
    resistance = data['High'].rolling(window=window).max()
    return support, resistance

def calculate_detailed_score(df):
    """Calculate a detailed score with buy/sell signals and target prices"""
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
        
        # Calculate technical indicators
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = calculate_macd(df['Close'])
        df['ATR'] = calculate_atr(df)
        df['Support'], df['Resistance'] = find_support_resistance(df)
        
        # Get the latest data points
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize signals dict and score
        signals = {}
        score = 0
        
        # 1. Recent Price Change
        price_change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
        signals['price_change'] = price_change
        if price_change > 1.5:
            score += weights["price_change"]
            signals['price_change_signal'] = 'BULLISH'
        elif price_change < -1.5:
            score -= weights["price_change"]
            signals['price_change_signal'] = 'BEARISH'
        else:
            signals['price_change_signal'] = 'NEUTRAL'
        
        # 2. Volume Analysis
        avg_vol = df['Volume'][-5:-1].mean()
        vol_change = (latest['Volume'] - avg_vol) / avg_vol * 100
        signals['volume_change'] = vol_change
        if vol_change > 50 and price_change > 0:
            score += weights["volume_spike"]
            signals['volume_signal'] = 'BULLISH'
        elif vol_change > 50 and price_change < 0:
            score -= weights["volume_spike"]
            signals['volume_signal'] = 'BEARISH'
        else:
            signals['volume_signal'] = 'NEUTRAL'
        
        # 3. RSI Analysis
        current_rsi = latest['RSI']
        signals['rsi'] = current_rsi
        if 30 < current_rsi < 50 and current_rsi > prev['RSI']:
            score += weights["rsi_signal"]  # Bullish - coming out of oversold
            signals['rsi_signal'] = 'BULLISH'
        elif 70 > current_rsi > 50 and current_rsi < prev['RSI']:
            score -= weights["rsi_signal"]  # Bearish - coming down from overbought
            signals['rsi_signal'] = 'BEARISH'
        elif current_rsi < 30:
            signals['rsi_signal'] = 'OVERSOLD'
        elif current_rsi > 70:
            signals['rsi_signal'] = 'OVERBOUGHT'
        else:
            signals['rsi_signal'] = 'NEUTRAL'
        
        # 4. MACD Analysis
        macd_current = latest['MACD']
        macd_signal = latest['MACD_Signal']
        macd_prev = prev['MACD']
        macd_signal_prev = prev['MACD_Signal']
        
        signals['macd'] = macd_current
        signals['macd_signal'] = macd_signal
        
        if macd_current > macd_signal and macd_prev < macd_signal_prev:
            score += weights["macd_signal"]  # Bullish crossover
            signals['macd_signal'] = 'BULLISH_CROSS'
        elif macd_current < macd_signal and macd_prev > macd_signal_prev:
            score -= weights["macd_signal"]  # Bearish crossover
            signals['macd_signal'] = 'BEARISH_CROSS'
        elif macd_current > macd_signal:
            signals['macd_signal'] = 'BULLISH'
        elif macd_current < macd_signal:
            signals['macd_signal'] = 'BEARISH'
        else:
            signals['macd_signal'] = 'NEUTRAL'
        
        # 5. Moving Average Analysis
        signals['above_ma50'] = latest['Close'] > latest['MA50']
        signals['ma20_above_ma50'] = latest['MA20'] > latest['MA50']
        
        if signals['above_ma50']:
            score += weights["above_ma50"]
        
        if signals['ma20_above_ma50']:
            score += weights["ma20_above_ma50"]
        
        # 6. Price Action Analysis
        signals['green_candle'] = latest['Close'] > latest['Open']
        
        high_low_range = latest['High'] - latest['Low']
        if high_low_range > 0:
            high_close_ratio = (latest['Close'] - latest['Low']) / high_low_range
            signals['high_close_ratio'] = high_close_ratio
            if high_close_ratio > 0.8:
                score += weights["near_high"]
                signals['near_high'] = True
            else:
                signals['near_high'] = False
        
        signals['higher_high'] = latest['High'] > prev['High']
        signals['higher_low'] = latest['Low'] > prev['Low']
        
        if signals['green_candle']:
            score += weights["green_candle"]
        
        if signals['higher_high']:
            score += weights["higher_high"]
        
        if signals['higher_low']:
            score += weights["higher_low"]
        
        # 7. Support/Resistance Analysis
        support_level = latest['Support']
        resistance_level = latest['Resistance']
        
        signals['support'] = support_level
        signals['resistance'] = resistance_level
        
        # Check if price bounced off support
        if prev['Low'] <= prev['Support'] * 1.01 and latest['Close'] > latest['Open']:
            score += weights["bounced_support"]
            signals['bounced_support'] = True
        else:
            signals['bounced_support'] = False
        
        # Check if price broke through resistance
        if prev['High'] >= prev['Resistance'] * 0.99 and latest['Close'] > latest['Resistance']:
            score += weights["broke_resistance"]
            signals['broke_resistance'] = True
        else:
            signals['broke_resistance'] = False
        
        # 8. Volatility Analysis
        atr_to_price = latest['ATR'] / latest['Close'] * 100
        signals['atr_percent'] = atr_to_price
        
        if atr_to_price < 1.5:  # Low volatility
            score += weights["low_volatility"]
            signals['volatility'] = 'LOW'
        elif atr_to_price > 3:  # High volatility
            signals['volatility'] = 'HIGH'
        else:
            signals['volatility'] = 'MEDIUM'
        
        # Normalize the score to a 0-100 scale
        max_possible_score = sum(weights.values())
        normalized_score = min(max(0, (score + max_possible_score/2) / max_possible_score * 100), 100)
        signals['raw_score'] = score
        
        # Determine action based on score
        if normalized_score >= 75:
            action = 'STRONG_BUY'
        elif normalized_score >= 60:
            action = 'BUY'
        elif normalized_score >= 45:
            action = 'HOLD'
        elif normalized_score >= 30:
            action = 'SELL'
        else:
            action = 'STRONG_SELL'
        
        # Calculate entry, stop loss and target prices
        current_price = latest['Close']
        entry_price = current_price
        
        # For buy signals
        if action in ['BUY', 'STRONG_BUY']:
            # Stop loss at recent support or 2 ATR below entry
            stop_loss = max(support_level, current_price - 2 * latest['ATR'])
            # Target at recent resistance or 3:1 risk-reward ratio
            risk = entry_price - stop_loss
            target = entry_price + (risk * 3)
            risk_reward = 3.0
        
        # For sell signals
        elif action in ['SELL', 'STRONG_SELL']:
            # Stop loss at recent resistance or 2 ATR above entry
            stop_loss = min(resistance_level, current_price + 2 * latest['ATR'])
            # Target at recent support or 3:1 risk-reward ratio
            risk = stop_loss - entry_price
            target = entry_price - (risk * 3)
            risk_reward = 3.0
        
        # For hold signals
        else:
            stop_loss = None
            target = None
            risk_reward = None
        
        # If we're very close to a significant level, adjust the entry price
        if action in ['BUY', 'STRONG_BUY'] and signals.get('broke_resistance'):
            # Wait for pullback to broken resistance
            entry_price = resistance_level
        
        # Format prices to 2 decimal places for readability
        if entry_price is not None:
            entry_price = round(entry_price, 2)
        if stop_loss is not None:
            stop_loss = round(stop_loss, 2)
        if target is not None:
            target = round(target, 2)
        
        return {
            'score': round(normalized_score, 1),
            'action': action,
            'signals': signals,
            'entry': entry_price,
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward': risk_reward,
            'current_price': round(current_price, 2)
        }
    
    except Exception as e:
        logger.error(f"Error in calculate_detailed_score: {e}")
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


def get_analysis(symbol: str) -> dict:
    """Fetch data and analyze a stock symbol"""
    try:
        logger.info(f"Fetching data for {symbol}...")
        # Get 100 days of daily data for better analysis
        df = yf.download(symbol, period='100d', interval='1d', auto_adjust=True)
        
        # Check if data was received
        if df is None or df.empty:
            logger.error(f"No data received for {symbol}")
            return {
                'symbol': symbol,
                'score': -1,
                'action': 'NO_DATA',
                'message': f"No data available for {symbol}"
            }
            
        logger.info(f"[{symbol}] Data received: {len(df)} days")
        
        # Calculate detailed score and signals
        result = calculate_detailed_score(df)
        result['symbol'] = symbol
        
        # Add technical summary
        technical_summary = []
        
        signals = result.get('signals', {})
        if signals.get('rsi_signal') == 'BULLISH':
            technical_summary.append("RSI showing bullish momentum")
        elif signals.get('rsi_signal') == 'OVERSOLD':
            technical_summary.append("RSI indicates oversold conditions")
        
        if signals.get('macd_signal') == 'BULLISH_CROSS':
            technical_summary.append("Recent MACD bullish crossover")
        
        if signals.get('bounced_support'):
            technical_summary.append("Price bounced off support level")
        
        if signals.get('broke_resistance'):
            technical_summary.append("Price broke through resistance")
        
        if signals.get('higher_high') and signals.get('higher_low'):
            technical_summary.append("Making higher highs and higher lows")
        
        result['technical_summary'] = technical_summary
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return {
            'symbol': symbol,
            'score': -1,
            'action': 'ERROR',
            'message': f"Error analyzing {symbol}: {str(e)}"
        }

# Format analysis results into a readable message
def format_analysis_message(analysis):
    if analysis['score'] == -1:
        return f"❌ {analysis['symbol']}: {analysis.get('message', 'Analysis failed')}"
    
    # Get emoji for the action
    action_emojis = {
        'STRONG_BUY': '🔥 STRONG BUY',
        'BUY': '🟢 BUY',
        'HOLD': '⚪ HOLD',
        'SELL': '🔴 SELL',
        'STRONG_SELL': '⛔ STRONG SELL'
    }
    
    action_text = action_emojis.get(analysis['action'], analysis['action'])
    
    # Build the message
    message = f"*{analysis['symbol']}* - Score: *{analysis['score']}*/100\n"
    message += f"Recommendation: *{action_text}*\n\n"
    
    # Current price
    message += f"Current price: ₹{analysis.get('current_price', 'N/A')}\n"
    
    # Add entry/exit strategy if available
    if analysis['action'] in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
        message += "📊 *Trade Setup:*\n"
        message += f"Entry: ₹{analysis.get('entry', 'N/A')}\n"
        message += f"Stop Loss: ₹{analysis.get('stop_loss', 'N/A')}\n"
        message += f"Target: ₹{analysis.get('target', 'N/A')}\n"
        if analysis.get('risk_reward'):
            message += f"Risk/Reward: {analysis.get('risk_reward', 'N/A')}:1\n"
        if analysis.get('time_frame'):
            message += f"Time Frame: {analysis.get('time_frame', 'N/A')}\n"
    
    # Add technical indicators summary
    if analysis.get('technical_summary'):
        message += "\n📈 *Technical Summary:*\n"
        for point in analysis['technical_summary']:
            message += f"• {point}\n"
    
    # Add key signals and calculated values
    if analysis.get('signals'):
        signals = analysis['signals']
        message += "\n🔍 *Key Indicators:*\n"
        
        # Price data
        message += "• *Price Data:* "
        message += f"O: ₹{signals.get('open', 'N/A'):.2f} "
        message += f"H: ₹{signals.get('high', 'N/A'):.2f} "
        message += f"L: ₹{signals.get('low', 'N/A'):.2f} "
        message += f"C: ₹{signals.get('close', 'N/A'):.2f}\n"
        
        if 'price_change' in signals:
            message += f"• Price Change: {signals['price_change']:.2f}%\n"
        
        # Moving Averages
        if 'ma20' in signals and 'ma50' in signals:
            message += f"• MA20: ₹{signals['ma20']:.2f} | MA50: ₹{signals['ma50']:.2f}\n"
            message += f"• Price above MA50: {'Yes' if signals.get('above_ma50') else 'No'}\n"
            message += f"• MA20 above MA50: {'Yes' if signals.get('ma20_above_ma50') else 'No'}\n"
        
        # Support/Resistance
        if 'support' in signals and 'resistance' in signals:
            message += f"• Support: ₹{signals['support']:.2f} | Resistance: ₹{signals['resistance']:.2f}\n"
            if signals.get('broke_resistance') == True:
                message += f"• Broke resistance: Yes\n"
            if signals.get('bounced_support') == True:
                message += f"• Bounced off support: Yes\n"
        
        # RSI
        if 'rsi' in signals:
            rsi_value = signals['rsi']
            if pd.notna(rsi_value):
                message += f"• RSI: {rsi_value:.1f}"
                if rsi_value < 30:
                    message += " (Oversold)"
                elif rsi_value > 70:
                    message += " (Overbought)"
                message += "\n"
        
        # MACD
        if 'macd' in signals and 'macd_signal_line' in signals:
            message += f"• MACD: {signals['macd']:.3f} | Signal: {signals['macd_signal_line']:.3f}\n"
            if signals.get('macd_signal') == 'BULLISH_CROSS':
                message += f"• MACD Bullish Crossover: Yes\n"
            elif signals.get('macd_signal') == 'BEARISH_CROSS':
                message += f"• MACD Bearish Crossover: Yes\n"
        
        # Volatility (ATR)
        if 'atr' in signals:
            message += f"• ATR: {signals['atr']:.2f} ({signals.get('atr_percent', 0):.2f}% of price)\n"
            message += f"• Volatility: {signals.get('volatility', 'N/A')}\n"
        
        # Volume
        if 'volume_change' in signals:
            message += f"• Volume Change: {signals['volume_change']:.2f}%\n"
    
    return message


# Command: /analyze <symbol>
async def analyze_stock(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    if not context.args:
        await update.message.reply_text("Please use: /analyze <symbol>")
        return

    symbol = context.args[0].upper()
    if ".NS" not in symbol:
        symbol += ".NS"

    await update.message.reply_text(f"🔍 Analyzing {symbol}...")
    analysis = get_analysis(symbol)
    
    # Format and send the analysis
    message = format_analysis_message(analysis)
    await update.message.reply_text(message, parse_mode='Markdown')

# Command: /scan - Scan all stocks for opportunities
async def scan_stocks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    await update.message.reply_text("🔍 Scanning all stocks for opportunities...\nThis may take a few minutes.")
    
    # Create a status message to update
    status_message = await update.message.reply_text("⏳ Scan progress: 0%")
    
    buy_opportunities = []
    sell_opportunities = []
    errors = []
    progress = 0
    
    for symbol in SYMBOLS:
        try:
            analysis = get_analysis(symbol)
            
            if analysis['score'] == -1:
                errors.append(symbol)
            elif analysis['action'] in ['BUY', 'STRONG_BUY']:
                buy_opportunities.append(analysis)
            elif analysis['action'] in ['SELL', 'STRONG_SELL']:
                sell_opportunities.append(analysis)
                
            # Update progress every 5%
            new_progress = int((progress / len(SYMBOLS)) * 100)
            if new_progress % 5 == 0 and new_progress > int((progress - 1) / len(SYMBOLS) * 100):
                await status_message.edit_text(f"⏳ Scan progress: {new_progress}%")
                
            progress += 1
            
        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")
            errors.append(symbol)
            progress += 1
    
    # Sort opportunities by score
    buy_opportunities.sort(key=lambda x: x['score'], reverse=True)
    sell_opportunities.sort(key=lambda x: x['score'])
    
    # Build the results message
    if buy_opportunities or sell_opportunities:
        await status_message.edit_text("✅ Scan complete! Preparing results...")
        
        # Send buy opportunities
        if buy_opportunities:
            buy_message = "🟢 *BUY OPPORTUNITIES:*\n\n"
            for i, opportunity in enumerate(buy_opportunities[:5], 1):
                buy_message += f"{i}. *{opportunity['symbol']}* - Score: {opportunity['score']}/100\n"
                buy_message += f"   Price: ₹{opportunity.get('current_price', 'N/A')} | Entry: ₹{opportunity.get('entry', 'N/A')}\n"
                buy_message += f"   Target: ₹{opportunity.get('target', 'N/A')} | Stop: ₹{opportunity.get('stop_loss', 'N/A')}\n"
                buy_message += f"   Time Frame: {opportunity.get('time_frame', 'N/A')} | R:R: {opportunity.get('risk_reward', 'N/A')}:1\n\n"
            
            await update.message.reply_text(buy_message, parse_mode='Markdown')
            
            # Create buttons for detailed analysis
            keyboard = []
            row = []
            for i, opportunity in enumerate(buy_opportunities[:5]):
                symbol = opportunity['symbol'].replace('.NS', '')
                row.append(InlineKeyboardButton(symbol, callback_data=f"analyze_{opportunity['symbol']}"))
                if (i + 1) % 3 == 0 or i == len(buy_opportunities[:5]) - 1:
                    keyboard.append(row)
                    row = []
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Select a stock for detailed analysis:", reply_markup=reply_markup)
        
        # Send sell opportunities
        if sell_opportunities:
            sell_message = "🔴 *SELL OPPORTUNITIES:*\n\n"
            for i, opportunity in enumerate(sell_opportunities[:5], 1):
                sell_message += f"{i}. *{opportunity['symbol']}* - Score: {opportunity['score']}/100\n"
                sell_message += f"   Price: ₹{opportunity.get('current_price', 'N/A')} | Entry: ₹{opportunity.get('entry', 'N/A')}\n"
                sell_message += f"   Target: ₹{opportunity.get('target', 'N/A')} | Stop: ₹{opportunity.get('stop_loss', 'N/A')}\n"
                sell_message += f"   Time Frame: {opportunity.get('time_frame', 'N/A')} | R:R: {opportunity.get('risk_reward', 'N/A')}:1\n\n"
            
            await update.message.reply_text(sell_message, parse_mode='Markdown')
    else:
        await status_message.edit_text("✅ Scan complete! No strong buy or sell signals found.")
    
    if errors:
        error_message = f"⚠️ Could not analyze {len(errors)} symbols: {', '.join(errors[:5])}"
        if len(errors) > 5:
            error_message += f" and {len(errors) - 5} more"
        await update.message.reply_text(error_message)

# Handle callback queries from inline buttons
async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("analyze_"):
        symbol = query.data.replace("analyze_", "")
        await query.message.reply_text(f"🔍 Analyzing {symbol}...")
        analysis = get_analysis(symbol)
        message = format_analysis_message(analysis)
        await query.message.reply_text(message, parse_mode='Markdown')

# Command: /watchlist - Show your watchlist
async def show_watchlist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    # Here we use a simplified watchlist - in a real app you'd store this in a database
    watchlist = [
        "RELIANCE.NS", "INFY.NS", "TCS.NS", "HDFCBANK.NS", "ITC.NS", 
        "TATAMOTORS.NS", "ADANIENT.NS", "WIPRO.NS", "SBIN.NS", "ICICIBANK.NS"
    ]
    
    await update.message.reply_text("🔍 Analyzing your watchlist...")
    
    results = []
    for symbol in watchlist:
        try:
            analysis = get_analysis(symbol)
            results.append(analysis)
        except Exception as e:
            logger.error(f"Error analyzing watchlist item {symbol}: {e}")
    
    # Sort by action (buy first, then hold, then sell)
    action_priority = {
        'STRONG_BUY': 0,
        'BUY': 1,
        'HOLD': 2,
        'SELL': 3,
        'STRONG_SELL': 4
    }
    
    results.sort(key=lambda x: action_priority.get(x.get('action', 'HOLD'), 2))
    
    if results:
        watchlist_message = "📋 *YOUR WATCHLIST:*\n\n"
        for result in results:
            if result['score'] == -1:
                watchlist_message += f"❌ {result['symbol']}: Analysis failed\n"
                continue
                
            action_emojis = {
                'STRONG_BUY': '🔥',
                'BUY': '🟢',
                'HOLD': '⚪',
                'SELL': '🔴',
                'STRONG_SELL': '⛔'
            }
            emoji = action_emojis.get(result['action'], '⚪')
            
            watchlist_message += f"{emoji} *{result['symbol']}* - Score: {result['score']}/100 - {result['action']}\n"
            watchlist_message += f"   Price: ₹{result.get('current_price', 'N/A')}"
            
            if result['action'] in ['BUY', 'STRONG_BUY', 'SELL', 'STRONG_SELL']:
                watchlist_message += f" | Entry: ₹{result.get('entry', 'N/A')}"
                watchlist_message += f" | Target: ₹{result.get('target', 'N/A')}"
                watchlist_message += f" | SL: ₹{result.get('stop_loss', 'N/A')}"
                watchlist_message += f"\n   Time Frame: {result.get('time_frame', 'N/A')} | R:R: {result.get('risk_reward', 'N/A')}:1"
            
            watchlist_message += "\n\n"
        
        await update.message.reply_text(watchlist_message, parse_mode='Markdown')
        
        # Create buttons for detailed analysis
        keyboard = []
        row = []
        for i, result in enumerate(results):
            if result['score'] != -1:  # Only add buttons for valid analyses
                symbol = result['symbol'].replace('.NS', '')
                row.append(InlineKeyboardButton(symbol, callback_data=f"analyze_{result['symbol']}"))
                if (i + 1) % 3 == 0 or i == len(results) - 1:
                    keyboard.append(row)
                    row = []
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text("Select a stock for detailed analysis:", reply_markup=reply_markup)
    else:
        await update.message.reply_text("No valid watchlist items to display.")

# Command: /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    welcome_message = (
        "👋 *Welcome to StockSniperBot Pro!*\n\n"
        "This bot helps you identify trading opportunities in the Indian stock market with specific buy/sell recommendations.\n\n"
        "*Commands:*\n"
        "• /analyze <symbol> - Get detailed analysis for a specific stock\n"
        "• /scan - Scan all stocks for trading opportunities\n"
        "• /watchlist - View analysis of your watchlist\n"
        "• /today - Get today's market overview\n\n"
        "You can also just type a stock symbol to get a quick analysis.\n\n"
        "Auto-scan is active during market hours and will alert you about high-scoring opportunities."
    )
    
    await update.message.reply_text(welcome_message, parse_mode='Markdown')

# Command: /today - Market overview
async def market_today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return
    
    await update.message.reply_text("📊 Fetching today's market overview...")
    
    # Get market index data
    try:
        indices = ["^NSEI", "^BSESN"]  # Nifty 50 and Sensex
        index_data = {}
        
        for index in indices:
            data = yf.download(index, period="2d", interval="1d")
            if data is not None and not data.empty and len(data) >= 2:
                latest = data.iloc[-1]
                prev = data.iloc[-2]
                change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                index_data[index] = {
                    'close': latest['Close'],
                    'change': change,
                    'volume': latest['Volume']
                }
        
        # Get sector performance (sample sectors)
        sectors = {
            "NIFTY BANK": "^NSEBANK",
            "NIFTY IT": "NIFTYIT.NS",
            "NIFTY PHARMA": "NIFTYPHARMA.NS",
            "NIFTY AUTO": "NIFTYAUTO.NS",
            "NIFTY METAL": "NIFTYMETAL.NS"
        }
        
        sector_data = {}
        for name, symbol in sectors.items():
            try:
                data = yf.download(symbol, period="2d", interval="1d")
                if data is not None and not data.empty and len(data) >= 2:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2]
                    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                    sector_data[name] = {
                        'close': latest['Close'],
                        'change': change
                    }
            except Exception as e:
                logger.error(f"Error fetching {name} data: {e}")
        
        # Build the message
        message = "📈 *TODAY'S MARKET OVERVIEW*\n\n"
        
        # Market indices
        message += "*Major Indices:*\n"
        if "^NSEI" in index_data:
            nifty = index_data["^NSEI"]
            emoji = "🟢" if nifty['change'] > 0 else "🔴"
            message += f"{emoji} *Nifty 50:* {nifty['close']:.2f} ({nifty['change']:+.2f}%)\n"
        
        if "^BSESN" in index_data:
            sensex = index_data["^BSESN"]
            emoji = "🟢" if sensex['change'] > 0 else "🔴"
            message += f"{emoji} *Sensex:* {sensex['close']:.2f} ({sensex['change']:+.2f}%)\n"
        
        message += "\n*Sector Performance:*\n"
        for name, data in sector_data.items():
            emoji = "🟢" if data['change'] > 0 else "🔴"
            message += f"{emoji} *{name}:* {data['change']:+.2f}%\n"
        
        # Get top gainers and losers
        all_stocks = []
        for symbol in SYMBOLS[:30]:  # Analyze a subset for performance
            try:
                data = yf.download(symbol, period="2d", interval="1d")
                if data is not None and not data.empty and len(data) >= 2:
                    latest = data.iloc[-1]
                    prev = data.iloc[-2]
                    change = ((latest['Close'] - prev['Close']) / prev['Close']) * 100
                    all_stocks.append({
                        'symbol': symbol,
                        'close': latest['Close'],
                        'change': change
                    })
            except Exception:
                pass
        
        # Sort for top gainers and losers
        gainers = sorted(all_stocks, key=lambda x: x['change'], reverse=True)[:5]
        losers = sorted(all_stocks, key=lambda x: x['change'])[:5]
        
        message += "\n*Top Gainers:*\n"
        for stock in gainers:
            message += f"🟢 *{stock['symbol']}:* {stock['change']:+.2f}%\n"
        
        message += "\n*Top Losers:*\n"
        for stock in losers:
            message += f"🔴 *{stock['symbol']}:* {stock['change']:+.2f}%\n"
        
        # Market insight based on overall performance
        message += "\n*Market Insight:*\n"
        if "^NSEI" in index_data:
            nifty_change = index_data["^NSEI"]['change']
            if nifty_change > 1.5:
                message += "• Strong bullish momentum in the market\n"
            elif nifty_change > 0.5:
                message += "• Market showing positive bias\n"
            elif nifty_change < -1.5:
                message += "• Significant selling pressure in the market\n"
            elif nifty_change < -0.5:
                message += "• Market showing negative bias\n"
            else:
                message += "• Market trading in a range with no clear direction\n"
        
        # Add sector rotation insight
        if sector_data:
            best_sector = max(sector_data.items(), key=lambda x: x[1]['change'])
            worst_sector = min(sector_data.items(), key=lambda x: x[1]['change'])
            message += f"• Rotation towards {best_sector[0]} ({best_sector[1]['change']:+.2f}%)\n"
            message += f"• Weakness in {worst_sector[0]} ({worst_sector[1]['change']:+.2f}%)\n"
        
        await update.message.reply_text(message, parse_mode='Markdown')
    
    except Exception as e:
        logger.error(f"Error in market_today: {e}")
        await update.message.reply_text(f"❌ Error getting market overview: {str(e)}")

# Handle normal messages (manual symbol lookup)
def is_symbol(text: str) -> bool:
    text = text.strip()
    return text.isalpha() and 2 <= len(text) <= 10

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != AUTHORIZED_USER_ID:
        await update.message.reply_text("You are not authorized to use this bot.")
        return

    text = update.message.text.strip().upper()
    if not is_symbol(text):
        await update.message.reply_text("Unknown command or invalid symbol.\nUse /analyze <symbol> or just type a stock name.")
        return

    symbol = text
    if ".NS" not in symbol:
        symbol += ".NS"

    await update.message.reply_text(f"🔍 Analyzing {symbol}...")
    analysis = get_analysis(symbol)
    message = format_analysis_message(analysis)
    await update.message.reply_text(message, parse_mode='Markdown')

# Auto scanner loop - runs during market hours
async def auto_scan(app):
    logger.info("🔁 Running auto-scan for high-potential stocks...")
    
    high_potential_stocks = []
    
    for symbol in SYMBOLS:
        try:
            analysis = get_analysis(symbol)
            
            # Only alert on strong signals
            if analysis['score'] >= 80 and analysis['action'] in ['STRONG_BUY', 'BUY']:
                high_potential_stocks.append(analysis)
            
            # Limit to prevent rate limiting
            if len(high_potential_stocks) >= 5:
                break
                
        except Exception as e:
            logger.error(f"Error in auto_scan for {symbol}: {e}")
    
    # Send alerts if we found high potential stocks
    if high_potential_stocks:
        alert_message = "🚨 *HIGH POTENTIAL STOCKS DETECTED:*\n\n"
        
        for stock in high_potential_stocks:
            alert_message += f"🔥 *{stock['symbol']}* - Score: {stock['score']}/100\n"
            alert_message += f"Action: {stock['action']}\n"
            alert_message += f"Price: ₹{stock.get('current_price', 'N/A')} | Entry: ₹{stock.get('entry', 'N/A')}\n"
            alert_message += f"Target: ₹{stock.get('target', 'N/A')} | Stop: ₹{stock.get('stop_loss', 'N/A')}\n"
            alert_message += f"Time Frame: {stock.get('time_frame', 'N/A')} | R:R: {stock.get('risk_reward', 'N/A')}:1\n"
            
            if stock.get('technical_summary'):
                alert_message += "Key signals: " + ", ".join(stock['technical_summary'][:2]) + "\n"
                
            alert_message += "\n"
        
        await app.bot.send_message(
            chat_id=AUTHORIZED_USER_ID,
            text=alert_message,
            parse_mode='Markdown'
        )
        
        logger.info(f"Sent alerts for {len(high_potential_stocks)} high potential stocks")
    else:
        logger.info("Auto-scan complete. No high-potential stocks found.")

# Main entry point
async def main():
    try:
        application = Application.builder().token(BOT_TOKEN).build()
        
        # Register command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("analyze", analyze_stock))
        application.add_handler(CommandHandler("scan", scan_stocks))
        application.add_handler(CommandHandler("watchlist", show_watchlist))
        application.add_handler(CommandHandler("today", market_today))
        
        # Register message and callback handlers
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
        application.add_handler(CallbackQueryHandler(button_callback))

        await application.initialize()

        # Set up scheduler for auto-scanning
        scheduler = AsyncIOScheduler()
        
        # Auto-scan every hour during market hours
        scheduler.add_job(
            lambda: asyncio.create_task(auto_scan(application)), 
            'cron',
            day_of_week='mon-fri', 
            hour='9-15', 
            minute='30', 
            timezone='Asia/Kolkata'
        )
        
        scheduler.start()

        print("✅ StockSniperBot Pro started successfully!")
        logger.info("Bot and Scheduler started successfully")
        
        # Send startup notification to authorized user
        await application.bot.send_message(
            chat_id=AUTHORIZED_USER_ID, 
            text="✅ *StockSniperBot Pro* is now online!\n\n"
                 "Use the following commands:\n"
                 "• /analyze <symbol> - Get detailed analysis\n"
                 "• /scan - Scan all stocks for opportunities\n"
                 "• /watchlist - View your watchlist\n"
                 "• /today - Get market overview",
            parse_mode='Markdown'
        )
        
        await application.run_polling()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        print(f"❌ Fatal error: {e}")

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())