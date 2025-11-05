#!/bin/bash

# Trading Bot Setup and Run Script

echo "=================================="
echo "CRYPTO TRADING BOT v2.0 SETUP"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Python version: $python_version"

# Install required packages
echo ""
echo "Installing required packages..."
pip3 install pandas numpy requests aiohttp python-dotenv --break-system-packages

# Create necessary directories
echo ""
echo "Setting up directories..."
mkdir -p logs
mkdir -p data
mkdir -p reports

# Check for API keys
echo ""
echo "Checking API keys..."

if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  DEEPSEEK_API_KEY not set!"
    echo "   To enable AI analysis, run:"
    echo "   export DEEPSEEK_API_KEY='your_api_key_here'"
else
    echo "✓ DEEPSEEK_API_KEY is set"
fi

if [ -z "$BINANCE_API_KEY" ]; then
    echo "ℹ️  BINANCE_API_KEY not set (using public API)"
else
    echo "✓ BINANCE_API_KEY is set"
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file template..."
    cat > .env << 'EOF'
# Trading Bot Configuration

# API Keys
DEEPSEEK_API_KEY=your_deepseek_api_key_here
BINANCE_API_KEY=optional_for_public_data
BINANCE_API_SECRET=optional_for_public_data

# Trading Parameters
INITIAL_CAPITAL=200
MAX_POSITIONS=2
POSITION_SIZE_PCT=0.3

# Risk Management
STOP_LOSS_PCT=2.0
TAKE_PROFIT_PCT=3.0
TRAILING_STOP_PCT=1.5

# AI Settings
AI_CONFIDENCE_THRESHOLD=70
EOF
    echo "✓ Created .env file template"
fi

echo ""
echo "=================================="
echo "SETUP COMPLETE!"
echo "=================================="
echo ""
echo "To run the bot:"
echo "1. Edit .env file with your API keys"
echo "2. Run: python3 trading_bot.py"
echo ""
echo "Features:"
echo "✓ Crypto IQ Burst Detection"
echo "✓ Bollinger Band Analysis"
echo "✓ AI-Powered Decision Making"
echo "✓ Reversal Pattern Recognition"
echo "✓ Risk Management with Trailing Stops"
echo "✓ Multi-Symbol Trading"
echo ""
echo "Starting bot in 5 seconds..."
sleep 5

# Run the trading bot
python3 trading_bot.py
