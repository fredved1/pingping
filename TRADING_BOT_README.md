# Advanced Crypto Trading Bot v2.0

## Overview
This trading bot combines multiple advanced strategies to identify high-probability trading opportunities in cryptocurrency markets:

- **Crypto IQ Bursts**: Detects high-volume absorption patterns that often precede significant price movements
- **Bollinger Band Analysis**: Identifies volatility-based entry points and squeeze breakouts
- **AI-Powered Decision Making**: Uses Deepseek AI to analyze market conditions and confirm signals
- **Reversal Pattern Recognition**: Identifies potential trend reversals using RSI divergences

## Key Improvements from Previous Version

### 1. Signal Quality Enhancement
- **IQ Burst Detection**: The bot now specifically looks for high-volume bars with minimal price movement (absorption), which often indicates accumulation or distribution before a major move
- **Multiple Confirmation**: Signals require convergence of multiple indicators (IQ Burst + BB position + RSI extreme)
- **AI Filtering**: All signals are validated by AI analysis for market context

### 2. Reduced Overtrading
- **Higher Confidence Threshold**: Minimum 70% confidence required (vs. no threshold before)
- **Limited Positions**: Maximum 2 concurrent positions (vs. unlimited)
- **Signal Prioritization**: Only takes the highest confidence signals

### 3. Better Risk Management
- **Trailing Stops**: Automatically trails stop loss when in profit
- **Dynamic Position Sizing**: 30% of available capital per position
- **Separate Stop/Target**: Each signal has calculated stop loss and take profit levels

## Trading Strategy Details

### Crypto IQ Bursts (Primary Strategy)
IQ Bursts are detected when:
1. Volume spike > 2.5x the 20-period average
2. Price change < 0.3% (absorption)
3. Combined with extreme RSI or BB position

**Entry Signals:**
- **Long IQ Burst**: IQ Burst + Price below BB lower + RSI < 30
- **Short IQ Burst**: IQ Burst + Price above BB upper + RSI > 70

### Reversal Patterns
The bot identifies reversals through:
1. Swing high/low detection (10-bar lookback)
2. RSI divergence confirmation
3. Volume confirmation

**Reversal Signals:**
- **Bullish Reversal**: Price makes lower low, RSI makes higher low
- **Bearish Reversal**: Price makes higher high, RSI makes lower high

### Bollinger Band Squeeze
Detects low volatility periods that often precede breakouts:
1. BB width < 20-period average - 0.5 standard deviations
2. Price breaks above/below middle band
3. RSI confirms direction

### AI Analysis Layer
Every 5 minutes, the AI:
1. Analyzes all market data across symbols
2. Evaluates technical signals
3. Considers market correlation
4. Provides trade recommendations with confidence scores

## Configuration

### Required API Keys
```bash
export DEEPSEEK_API_KEY='your_deepseek_api_key'
export BINANCE_API_KEY='optional_for_public_data'
```

### Key Parameters
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
RSI_OVERSOLD = 30                # RSI oversold level
RSI_OVERBOUGHT = 70              # RSI overbought level
```

## Installation & Usage

### 1. Install Dependencies
```bash
pip3 install pandas numpy requests --break-system-packages
```

### 2. Set API Keys
```bash
export DEEPSEEK_API_KEY='your_api_key_here'
```

### 3. Run the Bot
```bash
# Using the setup script
chmod +x run_bot.sh
./run_bot.sh

# Or directly
python3 trading_bot.py
```

## Signal Examples

### High-Confidence IQ Burst Signal
```
Signal: BTC IQ_BURST_LONG
Confidence: 85%
Entry: $67,500
Stop Loss: $66,150 (-2%)
Take Profit: $69,525 (+3%)
Reason: IQ Burst detected at BB lower band with oversold RSI
Indicators: RSI=28, Volume Spike=3.2x, BB Position=below_lower
```

### AI-Confirmed Reversal Signal
```
Signal: ETH REVERSAL_LONG
Confidence: 75%
Entry: $3,700
Stop Loss: $3,626 (-2%)
Take Profit: $3,811 (+3%)
Reason: Bullish reversal pattern with RSI divergence
AI Analysis: Market showing accumulation, sentiment turning bullish
```

## Performance Monitoring

The bot generates reports every 15 minutes showing:
- Current equity and returns
- Open positions with P&L
- Win rate and trade statistics
- Latest AI market analysis
- Risk metrics

## Database Schema

All trades and signals are stored in SQLite:

### Tables
1. **trades**: Completed trades with entry/exit prices and P&L
2. **signals**: All generated signals (executed and non-executed)
3. **performance**: Equity curve and performance metrics

## Troubleshooting

### Common Issues

1. **No signals generated**
   - Check if markets are volatile enough
   - Verify API keys are set correctly
   - Lower confidence threshold temporarily

2. **Too many false signals**
   - Increase confidence threshold
   - Reduce position size
   - Check if IQ Burst parameters need adjustment

3. **AI not working**
   - Verify DEEPSEEK_API_KEY is correct
   - Check API rate limits
   - Review AI analysis logs

## Best Practices

1. **Start with paper trading**: Test the bot without real money first
2. **Monitor regularly**: Check reports and logs frequently
3. **Adjust parameters**: Fine-tune based on market conditions
4. **Risk management**: Never risk more than you can afford to lose
5. **Market conditions**: Bot works best in volatile, trending markets

## Strategy Optimization Tips

### For Better IQ Burst Detection
- Adjust `VOLUME_SPIKE_MULTIPLIER` based on the asset's typical volume patterns
- Lower `PRICE_ABSORPTION_THRESHOLD` for more sensitive detection
- Increase lookback period for volume averaging in low-liquidity markets

### For Improved Win Rate
- Increase `AI_CONFIDENCE_THRESHOLD` to 80%+
- Reduce `MAX_POSITIONS` to 1 for more selective trading
- Add time-of-day filters (avoid low-volume hours)

### For Higher Returns
- Increase `TAKE_PROFIT_PCT` during strong trends
- Use tighter stops (`STOP_LOSS_PCT`) in ranging markets
- Enable `TRAILING_STOP_PCT` earlier in profitable trades

## Expected Performance

Based on the strategy design:
- **Target Win Rate**: 55-65%
- **Average Win**: 2-3%
- **Average Loss**: 1-2%
- **Risk/Reward Ratio**: 1.5:1 to 2:1
- **Monthly Return Target**: 10-20% (in favorable conditions)

## Disclaimer

**IMPORTANT**: This bot is for educational purposes. Cryptocurrency trading carries significant risk. Past performance does not guarantee future results. Always:
- Test thoroughly before using real funds
- Start with small amounts
- Never invest more than you can afford to lose
- Understand the code before running it
- Monitor the bot's performance regularly

## Support & Updates

For issues or improvements:
1. Check the logs in `trading_bot.log`
2. Review the database for trade history
3. Adjust parameters in the Config class
4. Monitor AI analysis quality

---

**Version**: 2.0
**Last Updated**: November 2025
**Strategy Focus**: IQ Bursts + Bollinger Bands + AI Analysis
