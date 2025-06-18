import os
import logging
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from typing import Dict, List
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
import talib

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stocksniper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BOT_TOKEN = os.getenv("BOT_TOKEN")
AUTHORIZED_USER_IDS = set(map(int, os.getenv("AUTHORIZED_USER_IDS").split(',')))
SYMBOLS_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
CACHE_EXPIRY_MINUTES = 30
DATA_REFRESH_INTERVAL = 5  # Minutes

# Global variables
symbols_cache = None
last_refreshed = None

# Technical indicator parameters
INDICATOR_CONFIG = {
    "RSI_PERIOD": 14,
    "MACD_FAST": 12,
    "MACD_SLOW": 26,
    "MACD_SIGNAL": 9,
    "BB_PERIOD": 20,
    "BB_STDDEV": 2,
    "ATR_PERIOD": 14,
    "ICHIMOKU_CONVERSION": 9,
    "ICHIMOKU_BASE": 26,
    "ICHIMOKU_SPAN_B": 52
}

# Advanced scoring weights (dynamic adjustments based on market conditions)
DYNAMIC_WEIGHTS = {
    "momentum": {
        "price_change": 3.0,
        "volume_spike": 2.5,
        "rsi_signal": 2.0,
        "macd_signal": 2.5,
        "stochastic_signal": 2.0
    },
    "trend": {
        "ma_cross": 3.0,
        "ichimoku_cloud": 3.5,
        "adx_strength": 2.5
    },
    "volatility": {
        "bb_squeeze": 2.0,
        "atr_ratio": 1.5
    },
    "volume": {
        "obv_trend": 2.0,
        "vwap_signal": 2.5
    }
}

async def fetch_nse_symbols() -> List[str]:
    """Fetch current NSE symbols from official source"""
    global symbols_cache, last_refreshed
    
    if symbols_cache and last_refreshed and \
       (datetime.now() - last_refreshed) < timedelta(minutes=CACHE_EXPIRY_MINUTES):
        return symbols_cache
    
    try:
        logger.info("Refreshing NSE symbol list...")
        df = pd.read_csv(SYMBOLS_URL)
        symbols = df[df[' SERIES'] == 'EQ'][' SYMBOL'].tolist()
        symbols = [f"{s}.NS" for s in symbols]
        symbols_cache = symbols
        last_refreshed = datetime.now()
        logger.info(f"Refreshed {len(symbols)} symbols")
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        return symbols_cache or []

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators"""
    # Momentum indicators
    df['RSI'] = talib.RSI(df['Close'], timeperiod=INDICATOR_CONFIG["RSI_PERIOD"])
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(
        df['Close'], 
        fastperiod=INDICATOR_CONFIG["MACD_FAST"],
        slowperiod=INDICATOR_CONFIG["MACD_SLOW"],
        signalperiod=INDICATOR_CONFIG["MACD_SIGNAL"]
    )
    
    # Trend indicators
    df['MA20'] = talib.SMA(df['Close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['Close'], timeperiod=50)
    df['MA200'] = talib.SMA(df['Close'], timeperiod=200)
    
    # Ichimoku Cloud
    conversion = (talib.MAX(df['High'], INDICATOR_CONFIG["ICHIMOKU_CONVERSION"]) + 
                 talib.MIN(df['Low'], INDICATOR_CONFIG["ICHIMOKU_CONVERSION"])) / 2
    base = (talib.MAX(df['High'], INDICATOR_CONFIG["ICHIMOKU_BASE"]) + 
           talib.MIN(df['Low'], INDICATOR_CONFIG["ICHIMOKU_BASE"])) / 2
    span_a = (conversion + base) / 2
    span_b = (talib.MAX(df['High'], INDICATOR_CONFIG["ICHIMOKU_SPAN_B"]) + 
             talib.MIN(df['Low'], INDICATOR_CONFIG["ICHIMOKU_SPAN_B"])) / 2
    
    df['Ichimoku_Conversion'] = conversion
    df['Ichimoku_Base'] = base
    df['Ichimoku_Span_A'] = span_a
    df['Ichimoku_Span_B'] = span_b
    
    # Volatility indicators
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = talib.BBANDS(
        df['Close'],
        timeperiod=INDICATOR_CONFIG["BB_PERIOD"],
        nbdevup=INDICATOR_CONFIG["BB_STDDEV"],
        nbdevdn=INDICATOR_CONFIG["BB_STDDEV"]
    )
    df['ATR'] = talib.ATR(
        df['High'], df['Low'], df['Close'],
        timeperiod=INDICATOR_CONFIG["ATR_PERIOD"]
    )
    
    # Volume indicators
    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    return df

async def generate_stock_chart(symbol: str, df: pd.DataFrame) -> io.BytesIO:
    """Generate professional stock chart"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Price chart
    ax1.plot(df.index, df['Close'], label='Close', color='#1f77b4')
    ax1.plot(df.index, df['MA20'], label='20 MA', color='orange', linestyle='--', linewidth=1)
    ax1.plot(df.index, df['MA50'], label='50 MA', color='green', linestyle='--', linewidth=1)
    ax1.fill_between(df.index, df['Ichimoku_Span_A'], df['Ichimoku_Span_B'], 
                    where=df['Ichimoku_Span_A'] >= df['Ichimoku_Span_B'],
                    color='rgba(0,255,0,0.2)', interpolate=True)
    ax1.set_title(f'{symbol} Price Analysis')
    ax1.legend(loc='upper left')
    
    # Volume chart
    ax2.bar(df.index, df['Volume'], color=np.where(df['Close'] >= df['Open'], 'g', 'r'))
    ax2.set_title('Volume')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

async def analyze_symbol(symbol: str) -> Dict:
    """Comprehensive analysis with AI-enhanced scoring"""
    try:
        logger.info(f"Analyzing {symbol}")
        df = yf.download(symbol, period='6mo', interval='1d', auto_adjust=True)
        
        if df.empty or len(df) < 50:
            return {'error': 'Insufficient data'}
            
        df = calculate_technical_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Initialize scoring
        score = 0
        signals = {}
        
        # Momentum scoring
        if latest['RSI'] < 30 and latest['RSI'] > prev['RSI']:
            score += DYNAMIC_WEIGHTS['momentum']['rsi_signal']
            signals['rsi'] = 'Bullish reversal'
        
        # MACD crossover
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] < prev['MACD_Signal']:
            score += DYNAMIC_WEIGHTS['momentum']['macd_signal']
            signals['macd'] = 'Bullish crossover'
        
        # Ichimoku Cloud analysis
        if latest['Close'] > latest['Ichimoku_Span_A'] and latest['Close'] > latest['Ichimoku_Span_B']:
            score += DYNAMIC_WEIGHTS['trend']['ichimoku_cloud']
            signals['ichimoku'] = 'Above cloud'
        
        # Generate chart
        chart = await generate_stock_chart(symbol, df.tail(100))
        
        return {
            'symbol': symbol,
            'score': round(score, 1),
            'signals': signals,
            'chart': chart,
            'price': latest['Close'],
            'recommendation': 'STRONG_BUY' if score > 8 else 'BUY' if score > 5 else 'HOLD'
        }
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return {'error': str(e)}

async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Scan market for opportunities with interactive menu"""
    user_id = update.effective_user.id
    if user_id not in AUTHORIZED_USER_IDS:
        await update.message.reply_text("❌ Unauthorized access")
        return
    
    keyboard = [
        [InlineKeyboardButton("Top Gainers", callback_data='scan_gainers'),
         InlineKeyboardButton("Top Losers", callback_data='scan_losers')],
        [InlineKeyboardButton("Breakouts", callback_data='scan_breakouts'),
         InlineKeyboardButton("Volume Surge", callback_data='scan_volume')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🔍 Select scan type:",
        reply_markup=reply_markup
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle interactive button presses"""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith('scan_'):
        scan_type = query.data.split('_')[1]
        await handle_scan_type(query, scan_type)
    elif query.data.startswith('analyze_'):
        symbol = query.data.split('_')[1]
        await analyze_command(query, symbol)

async def alert_user(context: ContextTypes.DEFAULT_TYPE, message: str, chart: io.BytesIO = None):
    """Send alert to authorized users"""
    for user_id in AUTHORIZED_USER_IDS:
        try:
            if chart:
                await context.bot.send_photo(chat_id=user_id, photo=chart, caption=message)
            else:
                await context.bot.send_message(chat_id=user_id, text=message)
        except Exception as e:
            logger.error(f"Error sending alert to {user_id}: {e}")

async def market_open_check():
    """Check if Indian market is open"""
    now = datetime.now().astimezone(pytz.timezone("Asia/Kolkata"))
    if now.weekday() >= 5:  # Weekend
        return False
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

async def auto_scanner(context: ContextTypes.DEFAULT_TYPE):
    """Automated market scanner"""
    if not await market_open_check():
        return
    
    logger.info("Running auto scanner...")
    symbols = await fetch_nse_symbols()
    
    for symbol in symbols[:10]:  # Scan first 10 for demo
        analysis = await analyze_symbol(symbol)
        if analysis.get('recommendation') in ['STRONG_BUY', 'STRONG_SELL']:
            message = (f"🚨 {symbol} {analysis['recommendation']}\n"
                       f"Score: {analysis['score']}\n"
                       f"Price: {analysis['price']}")
            await alert_user(context, message, analysis.get('chart'))

def main():
    """Main application setup"""
    application = Application.builder().token(BOT_TOKEN).build()
    
    # Command setup
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("scan", "Market scanner"),
        BotCommand("analyze", "Analyze specific stock"),
        BotCommand("alerts", "Manage price alerts")
    ]
    application.bot.set_my_commands(commands)
    
    # Handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("scan", scan_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    
    # Scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        auto_scanner, 
        'interval', 
        minutes=DATA_REFRESH_INTERVAL,
        args=[application]
    )
    scheduler.start()
    
    # Start bot
    application.run_polling()

if __name__ == "__main__":
    main()