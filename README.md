# 🚀 Crypto IQ Burst Trading Bot

An advanced cryptocurrency trading bot that combines volume absorption analysis (IQ Bursts), Bollinger Bands, and AI-powered decision making to identify high-probability trading opportunities.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

## 📊 Strategy Overview

The bot implements the **Crypto IQ Burst** strategy - detecting high-volume absorption patterns that typically precede significant price movements. When institutional traders accumulate positions, they create distinctive volume spikes with minimal price movement. This "absorption" pattern is a powerful predictor of upcoming trends.

### Key Features

- **🎯 IQ Burst Detection**: Identifies volume spikes (>2.5x average) with minimal price change (<0.3%)
- **📈 Bollinger Band Analysis**: Catches volatility breakouts and squeeze patterns
- **🤖 AI-Powered Decisions**: Uses Deepseek AI to validate signals and analyze market context
- **🔄 Reversal Patterns**: Detects trend reversals using RSI divergences
- **⚡ Real-time Trading**: Connects to Binance API for live market data
- **🛡️ Risk Management**: Automatic stop-loss, take-profit, and trailing stops

## 🎯 Performance Targets

- **Win Rate**: 60-70% (vs 40% in traditional strategies)
- **Risk/Reward**: 1:1.5 to 1:2
- **Max Drawdown**: <10%
- **Monthly Return**: 10-20% in favorable conditions

## 🚦 How It Works

### IQ Burst Signal Example
```
1. Volume spikes to 3x average
2. Price moves only 0.2%
3. RSI oversold at 28
4. Price below Bollinger lower band
→ HIGH PROBABILITY LONG SIGNAL!
```

The bot continuously monitors 6 major cryptocurrencies (BTC, ETH, SOL, BNB, XRP, DOGE) for these patterns and executes trades automatically when conditions align.

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
chmod +x run_bot.sh
./run_bot.sh
```

## 📁 Project Structure

```
crypto-iq-burst-bot/
│
├── trading_bot.py          # Main trading bot with full features
├── demo_iq_burst.py        # Interactive demo of IQ Burst detection
├── test_bot.py             # Test version for live market analysis
├── run_bot.sh              # Setup and launch script
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variables template
└── README.md              # This file
```

## ⚙️ Configuration

Edit the configuration in `trading_bot.py` or use environment variables:

```python
# Trading Parameters
INITIAL_CAPITAL = 200.0          # Starting capital
MAX_POSITIONS = 2                # Max concurrent trades
POSITION_SIZE_PCT = 0.3          # 30% per position

# Risk Management  
STOP_LOSS_PCT = 2.0              # 2% stop loss
TAKE_PROFIT_PCT = 3.0            # 3% take profit
TRAILING_STOP_PCT = 1.5          # 1.5% trailing stop

# Signal Generation
AI_CONFIDENCE_THRESHOLD = 70     # Min confidence for trades
VOLUME_SPIKE_MULTIPLIER = 2.5    # For IQ Burst detection
```

## 📊 Trading Signals

### Signal Types

1. **IQ_BURST_LONG/SHORT** (Confidence: 85%)
   - High volume absorption pattern detected
   - Price at Bollinger Band extreme
   - RSI confirming oversold/overbought

2. **REVERSAL_LONG/SHORT** (Confidence: 75%)
   - Swing high/low with RSI divergence
   - Volume confirmation

3. **BB_SQUEEZE_LONG/SHORT** (Confidence: 70%)
   - Bollinger Band width contraction
   - Breakout direction confirmed

4. **AI_LONG/SHORT** (Confidence: Variable)
   - AI analysis of market conditions
   - Multiple factor confirmation

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

The bot generates reports every 15 minutes including:
- Portfolio equity and returns
- Open positions with P&L
- Win rate statistics
- AI market analysis
- Risk metrics

Reports are saved to:
- `trading_report.txt` - Regular updates
- `final_trading_report.txt` - Session summary
- SQLite database for historical analysis

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

### No signals generated
- Check market volatility
- Verify API keys
- Lower confidence threshold temporarily

### Too many false signals
- Increase AI_CONFIDENCE_THRESHOLD
- Adjust VOLUME_SPIKE_MULTIPLIER
- Reduce MAX_POSITIONS

### AI not working
- Verify DEEPSEEK_API_KEY
- Check rate limits
- Review logs in `trading_bot.log`

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

**Version**: 2.0  
**Author**: Advanced Trading Systems  
**Last Updated**: November 2024

For support, please open an issue on GitHub.
