# 🚀 Crypto IQ Burst Trading Bot v3.0

An advanced cryptocurrency trading bot with **real-time WebSocket integration** to Crypto IQ's burst detection service, combined with AI-powered decision making and pattern-specific trading strategies.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📊 Strategy Overview

The bot implements the **Crypto IQ Burst v3.0** strategy with **direct WebSocket connection** to Crypto IQ's real-time burst detection service. It identifies three distinct pattern types (ABSORPTION, BREAKOUT, REVERSAL) with pattern-specific risk parameters and position sizing.

### Key Features

- **🌐 Real-Time WebSocket**: Direct connection to Crypto IQ's burst detection service (matrix.cryptoiq.com)
- **🎯 Pattern-Specific Trading**: Three distinct patterns (ABSORPTION, BREAKOUT, REVERSAL) with optimized parameters
- **📊 Smart Position Sizing**: Pattern-based position multipliers (ABSORPTION: 1.2x, BREAKOUT: 1.0x, REVERSAL: 0.8x)
- **🤖 AI-Powered Decisions**: DeepSeek AI validates signals and analyzes market context
- **💰 Fee-Aware Trading**: WooX exchange fees (0.03% maker/taker) integrated into P&L calculations
- **🛡️ Pattern-Specific Risk**: Customized stop-loss and take-profit levels per pattern type
- **📈 Real-Time Market Data**: Binance API integration for price feeds and technical indicators

## 🎯 Performance Targets

- **Win Rate**: 60-70% (vs 40% in traditional strategies)
- **Risk/Reward**: 1:1.5 to 1:2
- **Max Drawdown**: <10%
- **Monthly Return**: 10-20% in favorable conditions

## 🚦 How It Works

### Pattern Types & Trading Logic

**1. ABSORPTION Pattern** (85% confidence, highest priority)
```
- High PV (Price-Volume) with low delta
- Indicates liquidation/accumulation zones
- INVERTED LOGIC: Positive delta = SHORT, Negative delta = LONG
- Position size: 60% of capital (1.2x multiplier)
- Risk: 2.0% stop-loss, 4.5% take-profit
```

**2. BREAKOUT Pattern** (80% confidence)
```
- High delta with high strings count
- Momentum confirmation required
- Standard directional trading
- Position size: 50% of capital (1.0x base)
- Risk: 3.0% stop-loss, 5.0% take-profit
```

**3. REVERSAL Pattern** (75% confidence)
```
- Delta direction change detected
- Requires confirmation from RSI/BB
- Medium position size
- Position size: 40% of capital (0.8x multiplier)
- Risk: 2.5% stop-loss, 4.0% take-profit
```

The bot maintains a **WebSocket connection** to Crypto IQ's service and monitors BTC, ETH, SOL, BNB, XRP, DOGE in real-time.

## 🔧 Installation

### Prerequisites

- Python 3.8+
- Binance account (optional for live trading)
- Deepseek API key for AI analysis

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/crypto-iq-burst-bot.git
cd crypto-iq-burst-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
```bash
export DEEPSEEK_API_KEY='your_deepseek_api_key'
export BINANCE_API_KEY='your_binance_key'  # Optional
```

4. Run the bot:
```bash
python3 crypto_iq_bot.py
```

Or use the automated setup script:
```bash
chmod +x run_bot.sh
./run_bot.sh
```

## 📁 Project Structure

```
crypto-iq-burst-bot/
│
├── crypto_iq_bot.py        # ⭐ MAIN BOT - v3.0 with WebSocket integration
├── demo_iq_burst.py        # Interactive demo of IQ Burst detection
├── test_bot.py             # Test version for live market analysis
├── dashboard_server.py     # Real-time web dashboard
├── dashboard.html          # Dashboard UI
├── run_bot.sh              # Setup and launch script
├── requirements.txt        # Python dependencies
├── bot_state.json          # Real-time bot state (auto-generated)
├── trading_bot.db          # SQLite database (auto-generated)
└── README.md               # This file
```

**Important**: Always run `crypto_iq_bot.py` - this is the latest v3.0 version with all features!

## ⚙️ Configuration

Edit the configuration in `crypto_iq_bot.py` (Config class):

```python
# Trading Parameters
INITIAL_CAPITAL = 200.0          # Starting capital
MAX_POSITIONS = 2                # Max concurrent trades
POSITION_SIZE_PCT = 0.5          # 50% base position size

# Pattern-Specific Position Multipliers
ABSORPTION_SIZE_MULTIPLIER = 1.2  # 60% for absorption (highest confidence)
BREAKOUT_SIZE_MULTIPLIER = 1.0    # 50% for breakout (base)
REVERSAL_SIZE_MULTIPLIER = 0.8    # 40% for reversal (lower confidence)

# Pattern-Specific Risk Parameters
ABSORPTION_STOP_LOSS_PCT = 2.0   # Tight stops for liquidation patterns
ABSORPTION_TAKE_PROFIT_PCT = 4.5

BREAKOUT_STOP_LOSS_PCT = 3.0     # Wider stops for momentum
BREAKOUT_TAKE_PROFIT_PCT = 5.0

REVERSAL_STOP_LOSS_PCT = 2.5     # Medium stops
REVERSAL_TAKE_PROFIT_PCT = 4.0

# Crypto IQ WebSocket
TOTAL_TARGET = 100               # PV threshold for burst detection
STRINGS_TARGET = 10              # Strings threshold
MIN_DELTA_FOR_SIGNAL = 50        # Minimum delta for trade signals

# WooX Exchange Fees
MAKER_FEE = 0.0003              # 0.03% maker fee
TAKER_FEE = 0.0003              # 0.03% taker fee
```

## 📊 Trading Signals

### Pattern-Based Signals (Real-Time WebSocket)

1. **ABSORPTION Pattern** (Confidence: 85% - Highest Priority)
   - Real-time burst from Crypto IQ WebSocket
   - High PV (>500) with low delta (<30)
   - **INVERTED LOGIC**: Positive delta → SHORT, Negative delta → LONG
   - RSI confirmation required (oversold/overbought)
   - Position: 60% of capital

2. **BREAKOUT Pattern** (Confidence: 80%)
   - High delta (>100) with high strings (>15)
   - Momentum confirmation from order imbalance
   - Standard directional entry
   - Position: 50% of capital

3. **REVERSAL Pattern** (Confidence: 75%)
   - Delta direction change detected
   - Bollinger Band extremes required
   - Second-half delta must be >50
   - Position: 40% of capital

### AI Enhancement
- DeepSeek AI validates all signals
- Can recommend SKIP or REDUCE_SIZE
- Analyzes sentiment, risk factors, and market context
- Minimum 70% AI confidence required

## 🧪 Testing

Run the demo to see IQ Burst detection in action:

```bash
python demo_iq_burst.py
```

This will:
- Generate sample data with IQ Burst patterns
- Show detection algorithm in action
- Display trading signals
- Explain the strategy

## 📈 Performance Monitoring

**Real-Time Dashboard**
```bash
python3 dashboard_server.py
# Then open http://localhost:8000 in your browser
```

The bot generates reports every 15 minutes including:
- Portfolio equity and returns with fee calculations
- Open positions with live P&L (fee-adjusted)
- Win rate statistics by pattern type
- Recent burst detections with patterns
- AI analysis history
- WebSocket connection status

Data is saved to:
- `bot_state.json` - Real-time state (updated every 5 seconds)
- `trading_bot.db` - SQLite database for historical analysis
- `crypto_iq_bot.log` - Detailed logging

## 🤖 AI Integration

The bot uses Deepseek AI to:
- Analyze market sentiment
- Validate technical signals
- Identify correlation risks
- Suggest optimal entries/exits

AI analysis runs every 5 minutes and requires 70%+ confidence for trade execution.

## ⚠️ Risk Management

Built-in safety features:
- Maximum 2 concurrent positions
- 30% capital allocation per trade
- 2% stop-loss on all trades
- Trailing stops lock in profits
- No leverage by default

## 📝 Database Schema

All data stored in SQLite:

```sql
trades          - Completed trades with P&L
signals         - All generated signals
performance     - Equity curve and metrics
```

## 🐛 Troubleshooting

### WebSocket not connecting
- Check internet connection
- Verify Crypto IQ service is online (wss://matrix.cryptoiq.com)
- Check firewall settings
- Review logs for connection errors

### No signals generated
- Verify WebSocket is connected (check logs for "✅ Crypto IQ WebSocket connected")
- Adjust TOTAL_TARGET and STRINGS_TARGET if too restrictive
- Check MIN_DELTA_FOR_SIGNAL threshold
- Monitor burst detections in logs (🎯 SIGNIFICANT BURST messages)

### AI not working
- Verify DEEPSEEK_API_KEY is set correctly
- Check API rate limits
- Review logs in `crypto_iq_bot.log`
- AI will use default analysis if unavailable (bot continues running)

## 📚 Learn More

- [Understanding IQ Bursts](TRADING_BOT_README.md)
- [Strategy Optimization Guide](TRADING_BOT_README.md#strategy-optimization-tips)
- [API Documentation](TRADING_BOT_README.md#configuration)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚡ Disclaimer

**IMPORTANT**: This bot is for educational purposes only. Cryptocurrency trading carries significant risk. Always:
- Test with paper trading first
- Start with small amounts
- Never invest more than you can afford to lose
- Understand the code before running
- Monitor performance regularly

## 🌟 Acknowledgments

- Inspired by institutional volume analysis techniques
- Bollinger Band strategy by John Bollinger
- AI integration powered by Deepseek

---

**Version**: 3.0
**Main File**: `crypto_iq_bot.py` ⭐
**Last Updated**: November 2024

### What's New in v3.0
- ✅ Real-time WebSocket integration with Crypto IQ service
- ✅ Pattern-specific trading (ABSORPTION, BREAKOUT, REVERSAL)
- ✅ Pattern-based position sizing and risk management
- ✅ Fee-aware P&L calculations (WooX exchange)
- ✅ CRITICAL FIX: Inverted ABSORPTION logic (liquidation theory)
- ✅ Real-time dashboard with live burst detection
- ✅ Enhanced logging and state management

For support, please open an issue on GitHub.
