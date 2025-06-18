from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path to import stock_bot
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stock_bot import run_bot_logic, get_stock_list

app = FastAPI(
    title="Stock Sniper API",
    description="API for scanning and scoring stocks based on technical indicators",
    version="1.0.0"
)

# Add CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockResult(BaseModel):
    symbol: str
    score: int
    current_price: float
    details: Dict[str, Any]

class ScanResponse(BaseModel):
    total_scanned: int
    high_score_count: int
    min_score_used: int
    stocks: List[StockResult]
    scan_time: str

@app.get("/")
def read_root():
    return {
        "message": "🎯 Stock Sniper API is live!",
        "endpoints": {
            "/scan": "Scan stocks with minimum score filter",
            "/stocks": "Get list of available stocks to scan",
            "/health": "Health check endpoint"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Stock Sniper API"}

@app.get("/scan", response_model=ScanResponse)
def scan_stocks(
    min_score: int = Query(10, description="Minimum score to return stocks (1-12)", ge=1, le=12),
    limit: int = Query(50, description="Maximum number of stocks to scan", ge=1, le=200)
):
    """
    Scan stocks and return those with scores >= min_score
    
    - **min_score**: Minimum score threshold (1-12)
    - **limit**: Maximum number of stocks to scan
    """
    try:
        results = run_bot_logic(min_score=min_score, limit=limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning stocks: {str(e)}")

@app.get("/stocks")
def get_available_stocks():
    """Get list of all available stocks that can be scanned"""
    try:
        stocks = get_stock_list()
        return {
            "total_stocks": len(stocks),
            "stocks": stocks[:50],  # Return first 50 for preview
            "note": "This is a preview. Full list used in scanning."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stock list: {str(e)}")

@app.get("/scan/{symbol}")
def scan_single_stock(symbol: str):
    """Scan a specific stock symbol"""
    try:
        from stock_bot import score_stock
        result = score_stock(symbol.upper())
        if result is None:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found or data unavailable")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scanning {symbol}: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
