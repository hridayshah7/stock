
import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import yfinance as yf
import pandas as pd
import ta
import requests
from bs4 import BeautifulSoup

BOT_TOKEN = "8166189610:AAEGeti-NF2BNYd68qY0CEysDKUz6xNNIpg"

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

weights = {
    "price_above_200ema": 1.5,
    "macd_bullish": 1,
    "rsi_good": 1,
    "volume_spike": 1.2,
    "bullish_candle": 1,
    "sma_crossover": 1.3,
    "adx_trend": 1,
    "price_above_prev_high": 1,
    "news_sentiment": 1,
    "price_consolidation": 1,
    "sector_strength": 1,
    "no_event_today": 1
}

def get_sector_strength():
    return True  # Mocked as positive sector

def get_news_sentiment(symbol):
    url = f"https://news.google.com/search?q={symbol}+stock"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.content, "html.parser")
        articles = soup.find_all("article")
        if not articles:
            return True
        headlines = [a.get_text().lower() for a in articles[:5]]
        negative_keywords = ["fraud", "loss", "scam", "fire", "strike", "fall", "drop", "layoff"]
        return not any(kw in h for h in headlines for kw in negative_keywords)
    except:
        return True  # Assume positive if error

def has_earnings_today(symbol):
    return False  # Placeholder: replace with NSE event parser if needed

def analyze_stock(symbol):
    try:
        df = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True)  # Set auto_adjust=True
        if df.empty:
            return f"⚠️ No data found for {symbol}"

        df.dropna(inplace=True)
        df["EMA_200"] = ta.trend.ema_indicator(df["Close"], window=200)

        macd_indicator = ta.trend.MACD(df["Close"])
        macd_diff = macd_indicator.macd_diff().iloc[-1]  # Get the last MACD value (single scalar value)

        adx_indicator = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
        adx = adx_indicator.adx().iloc[-1]  # Get the last ADX value (single scalar value)

        rsi = ta.momentum.rsi(df["Close"], window=14)
        df["Volume_SMA20"] = df["Volume"].rolling(window=20).mean()

        latest = df.iloc[-1]
        score = 0
        breakdown = []

        def add(condition, key, label):
            nonlocal score
            if condition:
                score += weights[key]
                breakdown.append(f"🟩 {label} ✅ (+{weights[key]})")
            else:
                breakdown.append(f"🟥 {label} ❌ (+0)")

        add(latest["Close"] > latest["EMA_200"], "price_above_200ema", "Price above 200 EMA")
        add(macd_diff > 0, "macd_bullish", "MACD Bullish")
        add(adx > 20, "adx_trend", "Trending ADX")
        add(30 < rsi.iloc[-1] < 70, "rsi_good", "RSI Normal")
        add(latest["Volume"] > latest["Volume_SMA20"], "volume_spike", "Volume Spike")
        add(latest["Close"] > latest["Open"], "bullish_candle", "Bullish Candle")
        add(df["Close"].iloc[-1] > df["Close"].iloc[-20], "sma_crossover", "Price Above 20D SMA")
        add(latest["Close"] > df["High"].iloc[-2], "price_above_prev_high", "Breakout of Prev High")
        add(get_news_sentiment(symbol), "news_sentiment", "News Sentiment Positive")
        add((df["High"].rolling(10).max().iloc[-1] - df["Low"].rolling(10).min().iloc[-1]) / latest["Close"] < 0.05,
            "price_consolidation", "Price Consolidation")
        add(get_sector_strength(), "sector_strength", "Sector Strength Positive")
        add(not has_earnings_today(symbol), "no_event_today", "No Earnings/Events Today")

        final_score = round(score, 2)
        result = f"📊 Stock: {symbol.upper()}\n" + "\n".join(breakdown) + f"\n\n🧮 Final Score: {final_score} / 12"

        if final_score >= 10:
            result += "\n🔥 Setup Detected! Watch closely."
        elif final_score >= 8:
            result += "\n⚠️ Decent setup, but wait for confirmation."

        return result
    except Exception as e:
        return f"❌ Error analyzing {symbol}: {str(e)}"



async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Welcome to StockSniperBot! Use /manual <symbol> to get a score.")

async def manual(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Please provide a stock symbol. Usage: /manual TCS")
        return

    symbol = context.args[0].upper()
    await update.message.reply_text(f"📥 Analyzing {symbol}...")
    result = analyze_stock(symbol)
    await update.message.reply_text(result)

if __name__ == "__main__":
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("manual", manual))
    print("🤖 Bot is running...")
    app.run_polling()
