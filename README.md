# 🎯 Stock Sniper API

A FastAPI-based stock scanning service that analyzes Indian stocks using technical indicators and returns high-scoring investment opportunities.

## 🚀 Features

- **Technical Analysis**: 12-point scoring system using RSI, MACD, Moving Averages, Volume, and Bollinger Bands
- **Real-time Data**: Fetches live stock data using Yahoo Finance
- **RESTful API**: Clean endpoints for integration with n8n, browsers, or other services
- **Filtering**: Customizable minimum score thresholds
- **Indian Stocks**: Pre-configured with top Indian stocks across sectors

## 📁 Project Structure

```
D:/stock/
├── api/
│   └── main.py          # FastAPI application
├── stock_bot.py         # Stock analysis logic
├── requirements.txt     # Python dependencies
├── vercel.json         # Vercel deployment config
├── start_server.bat    # Windows batch file to start server
├── test_local.py       # Local testing script
└── README.md           # This file
```

## 🛠️ Local Setup & Testing

### Quick Start (Windows)
1. Double-click `start_server.bat`
2. Wait for server to start
3. Open browser to `http://localhost:8000`

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Test the API
```bash
# Run test script
python test_local.py
```

## 🌐 API Endpoints

### Health Check
```
GET /
GET /health
```

### Scan Stocks
```
GET /scan?min_score=10&limit=50
```
**Parameters:**
- `min_score` (1-12): Minimum score threshold (default: 10)
- `limit` (1-200): Maximum stocks to scan (default: 50)

**Response:**
```json
{
  "total_scanned": 50,
  "high_score_count": 5,
  "min_score_used": 10,
  "stocks": [
    {
      "symbol": "TCS.NS",
      "score": 11,
      "current_price": 3245.50,
      "details": {
        "rsi": 45.2,
        "macd": {...},
        "ma20": 3200.0,
        "ma50": 3150.0,
        "volume_ratio": 1.8,
        "momentum_5d": 2.5
      }
    }
  ],
  "scan_time": "2025-06-19 10:30:00"
}
```

### Get Stock List
```
GET /stocks
```

### Scan Single Stock
```
GET /scan/{symbol}
```
Example: `GET /scan/TCS.NS`

## 🎯 Scoring System (12 Points Max)

1. **RSI Analysis** (0-2 points)
   - Oversold conditions: 2 points
   - Approaching oversold: 1 point
   - Neutral/bullish: 1 point

2. **MACD Signals** (0-2 points)
   - Bullish crossover: 2 points
   - Above signal line: 1 point

3. **Moving Averages** (0-2 points)
   - Price > MA20 > MA50: 2 points
   - Price > MA20: 1 point

4. **Volume Analysis** (0-2 points)
   - High volume (>1.5x avg): 2 points
   - Above average volume: 1 point

5. **Price Momentum** (0-2 points)
   - Strong momentum (>5%): 2 points
   - Positive momentum: 1 point

6. **Bollinger Bands** (0-2 points)
   - Near lower band: 2 points
   - Below middle: 1 point

## 🚀 Deploy to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy:**
   ```bash
   cd D:/stock
   vercel --prod
   ```

3. **Access your API:**
   ```
   https://your-project-name.vercel.app/scan?min_score=10
   ```

## 🔗 Integration Examples

### n8n HTTP Node
```
Method: GET
URL: https://your-api.vercel.app/scan?min_score=10&limit=30
```

### Python Client
```python
import requests

response = requests.get('http://localhost:8000/scan?min_score=8')
data = response.json()

for stock in data['stocks']:
    print(f"{stock['symbol']}: Score {stock['score']}")
```

### JavaScript/Browser
```javascript
fetch('http://localhost:8000/scan?min_score=10')
  .then(response => response.json())
  .then(data => {
    console.log(`Found ${data.high_score_count} high-scoring stocks`);
    data.stocks.forEach(stock => {
      console.log(`${stock.symbol}: ${stock.score} points`);
    });
  });
```

## 📊 Stock Coverage

The API covers major Indian stocks across sectors:
- **IT**: TCS, Infosys, Wipro, HCL Tech, Tech Mahindra
- **Banking**: HDFC Bank, ICICI Bank, SBI, Kotak Bank, Axis Bank
- **Energy**: Reliance, ONGC, IOC, BPCL
- **Auto**: Maruti, Hyundai Motor, Tata Motors, M&M
- **Pharma**: Sun Pharma, Dr. Reddy's, Cipla, Divis Labs
- **FMCG**: HUL, ITC, Nestle India, Britannia
- **And more...**

## ⚡ Performance Notes

- **Local Testing**: Processes ~50 stocks in 30-60 seconds
- **Vercel Deployment**: 10-second timeout limit (use smaller limit parameter)
- **Rate Limiting**: Built-in delays to respect Yahoo Finance API limits
- **Caching**: Consider implementing Redis for production use

## 🛡️ Error Handling

The API includes comprehensive error handling:
- Invalid stock symbols are skipped
- Network timeouts are handled gracefully
- Malformed data is filtered out
- HTTP error codes for client errors

## 🔧 Customization

### Add New Stocks
Edit the `get_stock_list()` function in `stock_bot.py`:
```python
def get_stock_list() -> List[str]:
    stocks = [
        "YOUR_STOCK.NS",  # Add your stocks here
        # ... existing stocks
    ]
    return stocks
```

### Modify Scoring Logic
Update the `score_stock()` function in `stock_bot.py` to adjust scoring criteria.

## 📝 License

This project is for educational and personal use. Please ensure compliance with data provider terms of service.

## 🆘 Troubleshooting

### Common Issues

1. **Import Error**: Make sure all files are in correct directories
2. **API Timeout**: Reduce the `limit` parameter for faster scans
3. **Missing Data**: Some stocks may not have sufficient historical data
4. **Rate Limiting**: Built-in delays should handle this automatically

### Support

For issues or questions, check the error logs in the console when running locally.

---

**Ready to find your next winning stock? 🎯**
