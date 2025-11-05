#!/usr/bin/env python3
"""
Trading Bot Test/Demo Version
Shows how IQ Burst detection and signal generation works
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import time

# ============================================
# IQ BURST DETECTION DEMO
# ============================================

def fetch_binance_data(symbol='BTCUSDT', interval='5m', limit=100):
    """Fetch real market data from Binance"""
    url = f"https://api.binance.com/api/v3/klines"
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        
        return df
    return None

def calculate_indicators(df):
    """Calculate technical indicators"""
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std * 2)
    df['bb_lower'] = df['bb_middle'] - (std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = df['volume'] / df['volume_sma']
    
    # Price change
    df['price_change_pct'] = df['close'].pct_change() * 100
    
    return df

def detect_iq_bursts(df, volume_multiplier=2.5, price_threshold=0.3):
    """
    Detect Crypto IQ Bursts
    High volume with minimal price movement = absorption before big move
    """
    
    # IQ Burst conditions
    high_volume = df['volume_spike'] > volume_multiplier
    low_price_change = abs(df['price_change_pct']) < price_threshold
    
    df['iq_burst'] = high_volume & low_price_change
    
    # Determine burst direction based on price position
    df['iq_burst_long'] = (
        df['iq_burst'] & 
        (df['close'] < df['bb_lower']) & 
        (df['rsi'] < 30)
    )
    
    df['iq_burst_short'] = (
        df['iq_burst'] & 
        (df['close'] > df['bb_upper']) & 
        (df['rsi'] > 70)
    )
    
    return df

def detect_reversals(df, lookback=10):
    """Detect reversal patterns"""
    
    # Find swing points
    df['swing_high'] = df['high'] == df['high'].rolling(window=lookback).max()
    df['swing_low'] = df['low'] == df['low'].rolling(window=lookback).min()
    
    # RSI divergence
    rsi_lookback = df['rsi'].rolling(window=lookback)
    
    # Bullish reversal: Price low + RSI higher low
    df['bullish_reversal'] = (
        df['swing_low'] & 
        (df['rsi'] > rsi_lookback.min() + 5) &
        (df['rsi'] < 40)
    )
    
    # Bearish reversal: Price high + RSI lower high
    df['bearish_reversal'] = (
        df['swing_high'] & 
        (df['rsi'] < rsi_lookback.max() - 5) &
        (df['rsi'] > 60)
    )
    
    return df

def generate_signals(df):
    """Generate trading signals from indicators"""
    signals = []
    
    # Get latest data
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # IQ Burst Signals
    if latest['iq_burst_long']:
        signals.append({
            'type': 'IQ_BURST_LONG',
            'confidence': 85,
            'reason': f"IQ Burst detected! Volume spike {latest['volume_spike']:.1f}x with only {abs(latest['price_change_pct']):.2f}% price change. RSI oversold at {latest['rsi']:.0f}",
            'entry': latest['close'],
            'stop_loss': latest['close'] * 0.98,
            'take_profit': latest['close'] * 1.03
        })
    
    elif latest['iq_burst_short']:
        signals.append({
            'type': 'IQ_BURST_SHORT', 
            'confidence': 85,
            'reason': f"IQ Burst detected! Volume spike {latest['volume_spike']:.1f}x with only {abs(latest['price_change_pct']):.2f}% price change. RSI overbought at {latest['rsi']:.0f}",
            'entry': latest['close'],
            'stop_loss': latest['close'] * 1.02,
            'take_profit': latest['close'] * 0.97
        })
    
    # Reversal Signals
    if latest['bullish_reversal']:
        signals.append({
            'type': 'REVERSAL_LONG',
            'confidence': 75,
            'reason': f"Bullish reversal pattern detected. Swing low with RSI divergence at {latest['rsi']:.0f}",
            'entry': latest['close'],
            'stop_loss': latest['low'],
            'take_profit': latest['close'] * 1.025
        })
    
    elif latest['bearish_reversal']:
        signals.append({
            'type': 'REVERSAL_SHORT',
            'confidence': 75,
            'reason': f"Bearish reversal pattern detected. Swing high with RSI divergence at {latest['rsi']:.0f}",
            'entry': latest['close'],
            'stop_loss': latest['high'],
            'take_profit': latest['close'] * 0.975
        })
    
    # Bollinger Band Squeeze Breakout
    if latest['bb_width'] < 0.02:
        if latest['close'] > latest['bb_middle'] and latest['rsi'] > 50:
            signals.append({
                'type': 'BB_SQUEEZE_LONG',
                'confidence': 70,
                'reason': f"Bollinger Band squeeze breakout. Width at {latest['bb_width']:.3f} (narrow)",
                'entry': latest['close'],
                'stop_loss': latest['bb_lower'],
                'take_profit': latest['bb_upper']
            })
    
    return signals

def print_market_analysis(symbol, df):
    """Print market analysis"""
    latest = df.iloc[-1]
    
    print("\n" + "=" * 60)
    print(f"MARKET ANALYSIS - {symbol}")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Price: ${latest['close']:.2f}")
    print(f"RSI: {latest['rsi']:.1f}")
    print(f"Volume Spike: {latest['volume_spike']:.2f}x average")
    print(f"Price Change: {latest['price_change_pct']:.2f}%")
    print(f"BB Position: ", end="")
    
    if latest['close'] > latest['bb_upper']:
        print("Above Upper Band")
    elif latest['close'] < latest['bb_lower']:
        print("Below Lower Band")
    else:
        pct_in_band = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
        print(f"{pct_in_band:.1%} in band")
    
    print(f"BB Width: {latest['bb_width']:.3f}")
    
    # Check for patterns
    print("\nPattern Detection:")
    if latest['iq_burst']:
        print("✓ IQ BURST DETECTED! High volume absorption")
    if latest['bullish_reversal']:
        print("✓ BULLISH REVERSAL pattern")
    if latest['bearish_reversal']:
        print("✓ BEARISH REVERSAL pattern")
    if latest['bb_width'] < 0.02:
        print("✓ BOLLINGER SQUEEZE detected")

def main():
    """Main demo function"""
    print("=" * 60)
    print("CRYPTO IQ BURST TRADING BOT - DEMO")
    print("=" * 60)
    
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print("\nFetching market data...")
    
    all_signals = []
    
    for symbol in symbols:
        # Fetch data
        df = fetch_binance_data(symbol, '5m', 100)
        if df is None:
            print(f"Failed to fetch data for {symbol}")
            continue
        
        # Calculate indicators
        df = calculate_indicators(df)
        df = detect_iq_bursts(df)
        df = detect_reversals(df)
        
        # Print analysis
        print_market_analysis(symbol, df)
        
        # Generate signals
        signals = generate_signals(df)
        
        if signals:
            print(f"\n🎯 SIGNALS for {symbol}:")
            for signal in signals:
                all_signals.append((symbol, signal))
                print(f"  • {signal['type']} (Confidence: {signal['confidence']}%)")
                print(f"    {signal['reason']}")
                print(f"    Entry: ${signal['entry']:.2f}, Stop: ${signal['stop_loss']:.2f}, Target: ${signal['take_profit']:.2f}")
        else:
            print(f"\n  No signals for {symbol}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    
    if all_signals:
        # Sort by confidence
        all_signals.sort(key=lambda x: x[1]['confidence'], reverse=True)
        
        print(f"Total signals generated: {len(all_signals)}")
        print("\nTop Opportunities:")
        
        for i, (symbol, signal) in enumerate(all_signals[:3], 1):
            print(f"\n{i}. {symbol} - {signal['type']}")
            print(f"   Confidence: {signal['confidence']}%")
            print(f"   Entry: ${signal['entry']:.2f}")
            print(f"   Risk/Reward: 1:{(signal['take_profit']-signal['entry'])/(signal['entry']-signal['stop_loss']):.1f}")
    else:
        print("No trading signals at this time. Market is quiet.")
        print("\nIQ Burst signals appear when:")
        print("• High volume spike (>2.5x average)")
        print("• Minimal price movement (<0.3%)")
        print("• Combined with extreme RSI or BB position")
        print("\nThis indicates absorption before a potential big move!")
    
    print("\n" + "=" * 60)
    print("\nNote: This is a demonstration of signal detection.")
    print("The full bot includes:")
    print("• AI analysis with Deepseek")
    print("• Position management")
    print("• Risk management with stops")
    print("• Performance tracking")
    print("• Automated execution")

if __name__ == "__main__":
    # Run once
    main()
    
    # Optionally run continuously
    # while True:
    #     main()
    #     time.sleep(300)  # Check every 5 minutes
