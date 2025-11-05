#!/usr/bin/env python3
"""
Crypto IQ Burst Trading Bot v3.0
Integrates with Crypto IQ's WebSocket service for real-time burst detection
Enhanced with DeepSeek AI analysis
"""

import os
import json
import time
import asyncio
import websocket
import threading
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque
import sqlite3

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Trading bot configuration"""

    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')

    # Crypto IQ WebSocket Configuration
    CRYPTO_IQ_WS_URL = "wss://matrix.cryptoiq.com/api/sentinel/ws"
    CRYPTO_IQ_PING_INTERVAL = 30  # seconds

    # Trading Parameters
    INITIAL_CAPITAL = 200.0
    MAX_POSITIONS = 2
    POSITION_SIZE_PCT = 0.5  # Base position size (50% of capital)

    # Pattern-specific position size multipliers
    # ABSORPTION: Larger positions (85% confidence - highest priority)
    ABSORPTION_SIZE_MULTIPLIER = 1.2  # 60% of capital

    # BREAKOUT: Normal positions (80% confidence)
    BREAKOUT_SIZE_MULTIPLIER = 1.0  # 50% of capital (base)

    # REVERSAL: Smaller positions (75% confidence)
    REVERSAL_SIZE_MULTIPLIER = 0.8  # 40% of capital

    # WooX Exchange Fees
    MAKER_FEE = 0.0003  # 0.03%
    TAKER_FEE = 0.0003  # 0.03%
    TRADING_FEE = 0.0006  # Total round-trip (entry + exit)

    # Burst Detection Thresholds
    TOTAL_TARGET = 100  # PV (price-volume) threshold
    STRINGS_TARGET = 10  # Strings threshold for detection
    MIN_DELTA_FOR_SIGNAL = 50  # Minimum delta for trade signal

    # Risk Management (adjusted for fees) - Pattern-specific
    # ABSORPTION: Tighter stops (liquidation = quick reversal)
    ABSORPTION_STOP_LOSS_PCT = 2.0
    ABSORPTION_TAKE_PROFIT_PCT = 4.5

    # BREAKOUT: Wider stops (momentum can continue)
    BREAKOUT_STOP_LOSS_PCT = 3.0
    BREAKOUT_TAKE_PROFIT_PCT = 5.0

    # REVERSAL: Medium stops (confirmation needed)
    REVERSAL_STOP_LOSS_PCT = 2.5
    REVERSAL_TAKE_PROFIT_PCT = 4.0

    TRAILING_STOP_PCT = 2.0

    # Symbols to trade
    TRADING_SYMBOLS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE']

    # Timeframes
    KLINE_INTERVAL = 5 * 60 * 1000  # 5 minutes in milliseconds

    # Binance WebSocket
    BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"

    # AI Analysis
    AI_CONFIDENCE_THRESHOLD = 70
    AI_ANALYSIS_INTERVAL = 300  # 5 minutes

# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class BurstData:
    """Burst data from Crypto IQ WebSocket"""
    s: Optional[str]  # Symbol
    ts: int  # Timestamp
    pv: float  # Price-Volume metric
    cv: float  # Cumulative Volume
    delta: float  # Delta value
    strings: int  # Number of strings

@dataclass
class BurstMarker(BurstData):
    """Extended burst data with kline alignment"""
    kline_ts: int  # Aligned kline timestamp
    x: Optional[float] = None
    y: Optional[float] = None
    radius: Optional[float] = None

@dataclass
class TradingSignal:
    """Trading signal generated from burst"""
    timestamp: datetime
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    burst_type: str  # 'ABSORPTION', 'BREAKOUT', 'REVERSAL'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    burst_data: BurstMarker
    market_context: Dict[str, Any]
    ai_analysis: Optional[Dict[str, Any]] = None

# ============================================
# DEEPSEEK AI ANALYST
# ============================================

class DeepSeekAnalyst:
    """AI-powered market analysis using DeepSeek"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def analyze_burst_signal(self, signal: TradingSignal, market_data: Dict) -> Dict:
        """Analyze burst signal with AI"""
        try:
            prompt = self._create_burst_analysis_prompt(signal, market_data)

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are an expert crypto trader analyzing IQ Burst patterns."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 300
                },
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']

                # Strip markdown code blocks if present
                if content.strip().startswith('```'):
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    elif content.startswith('```'):
                        content = content[3:]
                    if '```' in content:
                        content = content[:content.rfind('```')]
                    content = content.strip()

                analysis = json.loads(content)
                logging.info(f"✅ AI Analysis: {analysis.get('sentiment', 'UNKNOWN')} - Confidence: {analysis.get('confidence', 0)}%")
                return analysis
            else:
                logging.error(f"AI API error: {response.status_code}")
                return self._default_analysis()

        except Exception as e:
            logging.error(f"AI analysis error: {str(e)}")
            return self._default_analysis()

    def _create_burst_analysis_prompt(self, signal: TradingSignal, market_data: Dict) -> str:
        """Create analysis prompt"""
        return f"""Analyze this Crypto IQ Burst signal:

BURST SIGNAL:
Symbol: {signal.symbol}
Direction: {signal.direction}
Burst Type: {signal.burst_type}
Entry Price: ${signal.entry_price:.2f}

BURST METRICS:
PV (Price-Volume): {signal.burst_data.pv:.2f}
Delta: {signal.burst_data.delta:.2f}
Strings: {signal.burst_data.strings}
CV (Cumulative Volume): {signal.burst_data.cv:.2f}

MARKET CONTEXT:
RSI: {signal.market_context.get('rsi', 50):.1f}
BB Position: {signal.market_context.get('bb_position', 'middle')}
Order Imbalance: {signal.market_context.get('order_imbalance', 0):.2f}

Analyze if this is a strong signal and provide:
1. Sentiment (BULLISH/BEARISH/NEUTRAL)
2. Confidence (0-100)
3. Key reasons
4. Risk factors

Respond in JSON:
{{
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "confidence": 0-100,
    "reasoning": "brief explanation",
    "risk_factors": ["risk1", "risk2"],
    "recommendation": "TAKE/SKIP/REDUCE_SIZE"
}}"""

    def _default_analysis(self) -> Dict:
        """Default analysis when AI unavailable"""
        return {
            "sentiment": "NEUTRAL",
            "confidence": 50,
            "reasoning": "AI analysis unavailable",
            "risk_factors": [],
            "recommendation": "TAKE"
        }

# ============================================
# CRYPTO IQ WEBSOCKET CLIENT
# ============================================

class CryptoIQWebSocketClient:
    """Handles connection to Crypto IQ burst detection service"""

    def __init__(self, symbols: List[str], on_burst_callback=None):
        self.symbols = symbols
        self.on_burst_callback = on_burst_callback
        self.ws = None
        self.running = False

        # Burst data storage
        self.burst_markers = {symbol: deque(maxlen=1000) for symbol in symbols}
        self.recent_bursts = {symbol: deque(maxlen=100) for symbol in symbols}

        # Configuration
        self.interval = Config.KLINE_INTERVAL
        self.total_target = Config.TOTAL_TARGET
        self.strings_target = Config.STRINGS_TARGET

        # Threading
        self.ws_thread = None
        self.ping_thread = None

        logging.info(f"Crypto IQ WebSocket client initialized for {symbols}")

    def start(self):
        """Start WebSocket connection"""
        self.running = True

        # Start WebSocket in thread
        self.ws_thread = threading.Thread(target=self._run_websocket)
        self.ws_thread.daemon = True
        self.ws_thread.start()

        # Start ping thread
        self.ping_thread = threading.Thread(target=self._ping_loop)
        self.ping_thread.daemon = True
        self.ping_thread.start()

        logging.info("Crypto IQ WebSocket started")

    def _run_websocket(self):
        """Run WebSocket connection"""
        def on_open(ws):
            logging.info("✅ Crypto IQ WebSocket connected")
            # Subscribe to all symbols
            for symbol in self.symbols:
                subscription = {
                    "op": "sub",
                    "symbols": f"{symbol}USDT",
                    "pv": self.total_target,
                    "strings": self.strings_target
                }
                ws.send(json.dumps(subscription))
                logging.info(f"📊 Subscribed to {symbol}USDT bursts (PV>{self.total_target}, Strings>{self.strings_target})")

        def on_message(ws, message):
            try:
                data = json.loads(message)

                # Handle ping response
                if data.get('op') == 'pong':
                    return

                # Process burst data
                burst_data = BurstData(
                    s=data.get('s'),
                    ts=data.get('ts', 0),
                    pv=data.get('pv', 0),
                    cv=data.get('cv', 0),
                    delta=data.get('delta', 0),
                    strings=data.get('strings', 0)
                )

                self._handle_burst_data(burst_data)

            except Exception as e:
                logging.error(f"Error processing message: {e}")

        def on_error(ws, error):
            logging.error(f"Crypto IQ WebSocket error: {error}")

        def on_close(ws, close_status_code, close_msg):
            logging.info(f"Crypto IQ WebSocket closed: {close_msg}")
            if self.running:
                logging.info("Reconnecting in 5 seconds...")
                time.sleep(5)
                self._run_websocket()  # Reconnect

        self.ws = websocket.WebSocketApp(
            Config.CRYPTO_IQ_WS_URL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        self.ws.run_forever()

    def _handle_burst_data(self, burst_data: BurstData):
        """Process incoming burst data"""
        if not burst_data.s:
            return

        # Extract symbol (remove USDT suffix)
        symbol = burst_data.s.replace('USDT', '')

        if symbol not in self.symbols:
            return

        # Align to kline interval
        kline_ts = (burst_data.ts // self.interval) * self.interval

        # Create burst marker
        burst_marker = BurstMarker(
            s=burst_data.s,
            ts=burst_data.ts,
            pv=burst_data.pv,
            cv=burst_data.cv,
            delta=burst_data.delta,
            strings=burst_data.strings,
            kline_ts=kline_ts
        )

        # Store burst
        self.burst_markers[symbol].append(burst_marker)
        self.recent_bursts[symbol].append(burst_marker)

        # Log significant bursts
        if abs(burst_data.delta) > Config.MIN_DELTA_FOR_SIGNAL:
            logging.info(f"🎯 SIGNIFICANT BURST - {symbol}")
            logging.info(f"   PV: {burst_data.pv:.2f}, CV: {burst_data.cv:.2f}")
            logging.info(f"   Delta: {burst_data.delta:.2f}, Strings: {burst_data.strings}")

            # Trigger callback
            if self.on_burst_callback:
                self.on_burst_callback(symbol, burst_marker)

    def _ping_loop(self):
        """Send periodic pings to keep connection alive"""
        while self.running:
            if self.ws and self.ws.sock and self.ws.sock.connected:
                try:
                    self.ws.send(json.dumps({"op": "ping"}))
                except Exception as e:
                    logging.error(f"Ping error: {e}")

            time.sleep(Config.CRYPTO_IQ_PING_INTERVAL)

    def get_recent_bursts(self, symbol: str, minutes: int = 30) -> List[BurstMarker]:
        """Get recent bursts for a symbol"""
        cutoff_time = int(time.time() * 1000) - (minutes * 60 * 1000)
        return [b for b in self.recent_bursts.get(symbol, []) if b.ts > cutoff_time]

    def analyze_burst_pattern(self, symbol: str) -> Dict[str, Any]:
        """Analyze burst patterns for trading signals"""
        recent = self.get_recent_bursts(symbol, minutes=15)

        if len(recent) < 3:
            return {'pattern': 'insufficient_data'}

        # Calculate metrics
        total_pv = sum(b.pv for b in recent)
        avg_delta = np.mean([b.delta for b in recent])
        max_strings = max(b.strings for b in recent)

        # Detect patterns
        pattern = {
            'pattern': 'none',
            'confidence': 0,
            'direction': None,
            'metrics': {
                'total_pv': total_pv,
                'avg_delta': avg_delta,
                'max_strings': max_strings,
                'burst_count': len(recent)
            }
        }

        # Absorption pattern: High PV with low delta
        # CRITICAL: INVERT delta for absorption (liquidation theory)
        # Positive delta = liquidations above = TOP → SHORT
        # Negative delta = liquidations below = BOTTOM → LONG
        if total_pv > 500 and abs(avg_delta) < 30:
            pattern['pattern'] = 'absorption'
            pattern['confidence'] = 85
            pattern['direction'] = 'SHORT' if avg_delta > 0 else 'LONG'  # INVERSED!

        # Breakout pattern: High delta with high strings
        elif abs(avg_delta) > 100 and max_strings > 15:
            pattern['pattern'] = 'breakout'
            pattern['confidence'] = 80
            pattern['direction'] = 'LONG' if avg_delta > 0 else 'SHORT'

        # Reversal pattern: Change in delta direction
        elif len(recent) > 5:
            first_half = recent[:len(recent)//2]
            second_half = recent[len(recent)//2:]

            first_delta = np.mean([b.delta for b in first_half])
            second_delta = np.mean([b.delta for b in second_half])

            if first_delta * second_delta < 0 and abs(second_delta) > 50:
                pattern['pattern'] = 'reversal'
                pattern['confidence'] = 75
                pattern['direction'] = 'LONG' if second_delta > 0 else 'SHORT'

        return pattern

    def stop(self):
        """Stop WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()
        logging.info("Crypto IQ WebSocket stopped")

# ============================================
# BINANCE INTEGRATION
# ============================================

class BinanceDataFeed:
    """Fetches price data from Binance to complement burst signals"""

    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self.ws_url = Config.BINANCE_WS_BASE
        self.price_cache = {}
        self.orderbook_cache = {}

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            response = requests.get(
                f"{self.base_url}/ticker/price",
                params={'symbol': f"{symbol}USDT"}
            )
            if response.status_code == 200:
                price = float(response.json()['price'])
                self.price_cache[symbol] = price
                return price
        except Exception as e:
            logging.error(f"Error fetching price for {symbol}: {e}")

        return self.price_cache.get(symbol, 0)

    def get_orderbook(self, symbol: str, limit: int = 10) -> Dict:
        """Get order book for a symbol"""
        try:
            response = requests.get(
                f"{self.base_url}/depth",
                params={'symbol': f"{symbol}USDT", 'limit': limit}
            )
            if response.status_code == 200:
                data = response.json()
                self.orderbook_cache[symbol] = data
                return data
        except Exception as e:
            logging.error(f"Error fetching orderbook for {symbol}: {e}")

        return self.orderbook_cache.get(symbol, {})

    def get_klines(self, symbol: str, interval: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Get historical klines"""
        try:
            response = requests.get(
                f"{self.base_url}/klines",
                params={
                    'symbol': f"{symbol}USDT",
                    'interval': interval,
                    'limit': limit
                }
            )
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
        except Exception as e:
            logging.error(f"Error fetching klines for {symbol}: {e}")

        return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate technical indicators"""
        if df.empty or len(df) < 20:
            return {}

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_middle = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)

        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'bb_upper': bb_upper.iloc[-1],
            'bb_middle': bb_middle.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'volume_avg': df['volume'].rolling(20).mean().iloc[-1]
        }

# ============================================
# SIGNAL GENERATOR
# ============================================

class SignalGenerator:
    """Generates trading signals from Crypto IQ bursts and market data"""

    def __init__(self, ai_analyst: Optional[DeepSeekAnalyst] = None):
        self.binance = BinanceDataFeed()
        self.ai_analyst = ai_analyst
        self.active_signals = {}
        self.signal_history = deque(maxlen=1000)

    def generate_signal(self, symbol: str, burst_marker: BurstMarker,
                       burst_pattern: Dict) -> Optional[TradingSignal]:
        """Generate trading signal from burst and market data"""

        # Get current price
        current_price = self.binance.get_current_price(symbol)
        if current_price == 0:
            return None

        # Get market data
        df = self.binance.get_klines(symbol)
        if df.empty:
            return None

        # Calculate indicators
        indicators = self.binance.calculate_indicators(df)
        if not indicators:
            return None

        # Get order book
        orderbook = self.binance.get_orderbook(symbol)

        # Analyze market context
        market_context = self._analyze_market_context(
            symbol, current_price, indicators, orderbook
        )

        # Determine if signal should be generated
        if not self._should_generate_signal(burst_pattern, market_context):
            return None

        # Create trading signal
        direction = burst_pattern['direction']
        pattern_type = burst_pattern['pattern']

        # Get pattern-specific risk parameters
        if pattern_type == 'absorption':
            stop_loss_pct = Config.ABSORPTION_STOP_LOSS_PCT
            take_profit_pct = Config.ABSORPTION_TAKE_PROFIT_PCT
        elif pattern_type == 'breakout':
            stop_loss_pct = Config.BREAKOUT_STOP_LOSS_PCT
            take_profit_pct = Config.BREAKOUT_TAKE_PROFIT_PCT
        elif pattern_type == 'reversal':
            stop_loss_pct = Config.REVERSAL_STOP_LOSS_PCT
            take_profit_pct = Config.REVERSAL_TAKE_PROFIT_PCT
        else:
            # Fallback to reversal params
            stop_loss_pct = Config.REVERSAL_STOP_LOSS_PCT
            take_profit_pct = Config.REVERSAL_TAKE_PROFIT_PCT

        # Calculate entry, stop, and target
        if direction == 'LONG':
            entry_price = current_price
            stop_loss = entry_price * (1 - stop_loss_pct / 100)
            take_profit = entry_price * (1 + take_profit_pct / 100)
        else:  # SHORT
            entry_price = current_price
            stop_loss = entry_price * (1 + stop_loss_pct / 100)
            take_profit = entry_price * (1 - take_profit_pct / 100)

        signal = TradingSignal(
            timestamp=datetime.now(),
            symbol=symbol,
            direction=direction,
            burst_type=burst_pattern['pattern'].upper(),
            confidence=burst_pattern['confidence'],
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            burst_data=burst_marker,
            market_context=market_context
        )

        # Get AI analysis if available
        if self.ai_analyst:
            ai_analysis = self.ai_analyst.analyze_burst_signal(signal, market_context)
            signal.ai_analysis = ai_analysis

            # Adjust confidence based on AI
            if ai_analysis.get('recommendation') == 'SKIP':
                logging.info(f"❌ AI recommends skipping signal for {symbol}")
                return None
            elif ai_analysis.get('recommendation') == 'REDUCE_SIZE':
                logging.info(f"⚠️  AI recommends reducing size for {symbol}")
                signal.confidence = min(signal.confidence * 0.7, 70)

        # Store signal
        self.active_signals[symbol] = signal
        self.signal_history.append(signal)

        # Log signal
        self._log_signal(signal)

        return signal

    def _analyze_market_context(self, symbol: str, price: float,
                               indicators: Dict, orderbook: Dict) -> Dict:
        """Analyze market context for signal generation"""
        context = {
            'price': price,
            'rsi': indicators.get('rsi', 50),
            'bb_position': 'middle',
            'volume_ratio': 1.0,
            'bid_ask_spread': 0,
            'order_imbalance': 0
        }

        # BB position
        if price > indicators.get('bb_upper', price):
            context['bb_position'] = 'above_upper'
        elif price < indicators.get('bb_lower', price):
            context['bb_position'] = 'below_lower'

        # Order book analysis
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            bids = orderbook['bids'][:5]
            asks = orderbook['asks'][:5]

            if bids and asks:
                best_bid = float(bids[0][0])
                best_ask = float(asks[0][0])

                context['bid_ask_spread'] = (best_ask - best_bid) / best_bid * 100

                # Calculate order imbalance
                bid_volume = sum(float(b[1]) for b in bids)
                ask_volume = sum(float(a[1]) for a in asks)

                if bid_volume + ask_volume > 0:
                    context['order_imbalance'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)

        return context

    def _should_generate_signal(self, burst_pattern: Dict, market_context: Dict) -> bool:
        """Determine if signal should be generated"""

        # Check confidence threshold
        if burst_pattern['confidence'] < 70:
            return False

        # Check pattern type
        if burst_pattern['pattern'] == 'none':
            return False

        # Additional filters based on pattern
        if burst_pattern['pattern'] == 'absorption':
            # For absorption, we want RSI extremes
            if burst_pattern['direction'] == 'LONG' and market_context['rsi'] > 40:
                return False
            if burst_pattern['direction'] == 'SHORT' and market_context['rsi'] < 60:
                return False

        elif burst_pattern['pattern'] == 'breakout':
            # For breakout, we want momentum confirmation
            if abs(market_context['order_imbalance']) < 0.2:
                return False

        elif burst_pattern['pattern'] == 'reversal':
            # For reversal, we want BB extremes
            if market_context['bb_position'] == 'middle':
                return False

        return True

    def _log_signal(self, signal: TradingSignal):
        """Log trading signal"""
        logging.info(f"=" * 60)
        logging.info(f"📊 TRADING SIGNAL GENERATED - {signal.symbol}")
        logging.info(f"Direction: {signal.direction}")
        logging.info(f"Burst Type: {signal.burst_type}")
        logging.info(f"Confidence: {signal.confidence}%")
        logging.info(f"Entry: ${signal.entry_price:.2f}")
        logging.info(f"Stop Loss: ${signal.stop_loss:.2f}")
        logging.info(f"Take Profit: ${signal.take_profit:.2f}")
        logging.info(f"Burst Data: PV={signal.burst_data.pv:.1f}, Delta={signal.burst_data.delta:.1f}, Strings={signal.burst_data.strings}")
        logging.info(f"Market: RSI={signal.market_context['rsi']:.1f}, BB={signal.market_context['bb_position']}")

        if signal.ai_analysis:
            logging.info(f"AI Analysis: {signal.ai_analysis.get('sentiment')} - {signal.ai_analysis.get('reasoning')}")

        logging.info(f"=" * 60)

# ============================================
# MAIN TRADING BOT
# ============================================

class CryptoIQTradingBot:
    """Main trading bot that coordinates all components"""

    def __init__(self):
        self.config = Config()
        self.iq_client = None
        self.signal_generator = None
        self.ai_analyst = None
        self.positions = {}
        self.performance = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0
        }
        self.trade_history = []
        self.recent_bursts = []
        self.recent_ai_analyses = []
        self.state_file = 'bot_state.json'
        self.db_path = 'trading_bot.db'

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('crypto_iq_bot.log'),
                logging.StreamHandler()
            ]
        )

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                entry_time TEXT,
                exit_time TEXT,
                pnl REAL,
                pnl_pct REAL,
                exit_reason TEXT
            )
        ''')

        # Create signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                timestamp TEXT,
                symbol TEXT,
                signal_type TEXT,
                direction TEXT,
                confidence REAL,
                executed INTEGER
            )
        ''')

        # Create performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                timestamp TEXT,
                equity REAL,
                positions INTEGER,
                daily_pnl REAL,
                win_rate REAL
            )
        ''')

        conn.commit()
        conn.close()
        logging.info("Database initialized")

    def _save_trade(self, trade_data: Dict):
        """Save closed trade to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Generate unique trade ID
            trade_id = f"{trade_data['symbol']}_{trade_data['exit_time'].replace(':', '').replace('-', '').replace('T', '_')}"

            cursor.execute('''
                INSERT INTO trades (id, symbol, direction, entry_price, exit_price,
                                  quantity, entry_time, exit_time, pnl, pnl_pct, exit_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_id,
                trade_data['symbol'],
                trade_data['direction'],
                trade_data['entry_price'],
                trade_data['exit_price'],
                trade_data['quantity'],
                trade_data['entry_time'],
                trade_data['exit_time'],
                trade_data['pnl'],
                trade_data['pnl_pct'],
                trade_data['exit_reason']
            ))

            conn.commit()
            conn.close()
            logging.info(f"Trade saved to database: {trade_id}")

        except Exception as e:
            logging.error(f"Error saving trade to database: {e}")

    def start(self):
        """Start the trading bot"""
        logging.info("=" * 60)
        logging.info("CRYPTO IQ BURST TRADING BOT v3.0")
        logging.info("=" * 60)
        logging.info(f"Trading symbols: {', '.join(self.config.TRADING_SYMBOLS)}")
        logging.info(f"Burst thresholds: PV={self.config.TOTAL_TARGET}, Strings={self.config.STRINGS_TARGET}")
        logging.info(f"Initial capital: ${self.config.INITIAL_CAPITAL}")

        # Initialize DeepSeek AI if API key available
        if self.config.DEEPSEEK_API_KEY:
            self.ai_analyst = DeepSeekAnalyst(self.config.DEEPSEEK_API_KEY)
            logging.info("✅ DeepSeek AI analyst initialized")
        else:
            logging.warning("⚠️  No DeepSeek API key - AI analysis disabled")

        # Initialize signal generator
        self.signal_generator = SignalGenerator(ai_analyst=self.ai_analyst)

        # Initialize Crypto IQ WebSocket client
        self.iq_client = CryptoIQWebSocketClient(
            symbols=self.config.TRADING_SYMBOLS,
            on_burst_callback=self.on_burst_detected
        )

        # Start WebSocket
        self.iq_client.start()

        # Start monitoring loop
        self.run_monitoring_loop()

    def on_burst_detected(self, symbol: str, burst_marker: BurstMarker):
        """Called when a significant burst is detected"""
        try:
            # Analyze burst pattern first
            pattern = self.iq_client.analyze_burst_pattern(symbol)

            # Store burst for dashboard with pattern
            self.recent_bursts.append({
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'pv': burst_marker.pv,
                'delta': burst_marker.delta,
                'strings': burst_marker.strings,
                'cv': burst_marker.cv,
                'pattern': pattern['pattern']
            })
            # Keep only last 50
            if len(self.recent_bursts) > 50:
                self.recent_bursts = self.recent_bursts[-50:]

            if pattern['pattern'] != 'none':
                logging.info(f"🔥 Pattern detected: {pattern['pattern'].upper()} for {symbol} (Confidence: {pattern['confidence']}%)")

                # Generate trading signal
                signal = self.signal_generator.generate_signal(
                    symbol, burst_marker, pattern
                )

                if signal:
                    # Store AI analysis for dashboard
                    if signal.ai_analysis:
                        self.recent_ai_analyses.append({
                            'symbol': symbol,
                            'timestamp': datetime.now().isoformat(),
                            'sentiment': signal.ai_analysis.get('sentiment', 'UNKNOWN'),
                            'confidence': signal.ai_analysis.get('confidence', 0),
                            'reasoning': signal.ai_analysis.get('reasoning', ''),
                            'recommendation': signal.ai_analysis.get('recommendation', 'SKIP'),
                            'risk_factors': signal.ai_analysis.get('risk_factors', []),
                            'pattern': pattern['pattern'],
                            'direction': signal.direction
                        })
                        # Keep only last 20
                        if len(self.recent_ai_analyses) > 20:
                            self.recent_ai_analyses = self.recent_ai_analyses[-20:]

                    # Execute trade (in production, this would place actual orders)
                    self.execute_trade(signal)

        except Exception as e:
            logging.error(f"Error processing burst: {e}")

    def execute_trade(self, signal: TradingSignal):
        """Execute trade based on signal"""
        # Check if we already have a position
        if signal.symbol in self.positions:
            logging.info(f"Already have position in {signal.symbol}, skipping")
            return

        # Check max positions
        if len(self.positions) >= self.config.MAX_POSITIONS:
            logging.info(f"Max positions reached ({self.config.MAX_POSITIONS}), skipping")
            return

        # Create position (in production, this would place actual orders)
        position = {
            'symbol': signal.symbol,
            'direction': signal.direction,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'quantity': self._calculate_position_size(signal.entry_price, signal.burst_type),
            'entry_time': signal.timestamp,
            'burst_type': signal.burst_type
        }

        self.positions[signal.symbol] = position

        # Get multiplier for logging
        multiplier = 1.0
        if signal.burst_type == 'ABSORPTION':
            multiplier = self.config.ABSORPTION_SIZE_MULTIPLIER
        elif signal.burst_type == 'BREAKOUT':
            multiplier = self.config.BREAKOUT_SIZE_MULTIPLIER
        elif signal.burst_type == 'REVERSAL':
            multiplier = self.config.REVERSAL_SIZE_MULTIPLIER

        logging.info(f"✅ TRADE EXECUTED: {signal.symbol} {signal.direction} ({signal.burst_type})")
        logging.info(f"   Position size: {position['quantity']:.4f} (multiplier: {multiplier}x)")
        logging.info(f"   Entry: ${position['entry_price']:.2f}")
        logging.info(f"   Stop Loss: ${position['stop_loss']:.2f}")
        logging.info(f"   Take Profit: ${position['take_profit']:.2f}")

    def _calculate_position_size(self, price: float, burst_type: str) -> float:
        """Calculate position size based on available capital and pattern type"""
        # Get pattern-specific multiplier
        if burst_type == 'ABSORPTION':
            multiplier = self.config.ABSORPTION_SIZE_MULTIPLIER
        elif burst_type == 'BREAKOUT':
            multiplier = self.config.BREAKOUT_SIZE_MULTIPLIER
        elif burst_type == 'REVERSAL':
            multiplier = self.config.REVERSAL_SIZE_MULTIPLIER
        else:
            multiplier = 1.0  # Default

        available_capital = self.config.INITIAL_CAPITAL * self.config.POSITION_SIZE_PCT * multiplier
        return available_capital / price

    def run_monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while True:
                # Update positions
                self.update_positions()

                # Save state to JSON for dashboard
                self.save_state()

                # Generate report
                if datetime.now().minute % 15 == 0:
                    self.generate_report()

                time.sleep(5)  # Check every 5 seconds

        except KeyboardInterrupt:
            logging.info("Shutting down...")
            self.shutdown()

    def update_positions(self):
        """Update open positions with current prices"""
        for symbol, position in list(self.positions.items()):
            current_price = self.signal_generator.binance.get_current_price(symbol)

            if current_price == 0:
                continue

            # Calculate raw PnL
            if position['direction'] == 'LONG':
                raw_pnl = (current_price - position['entry_price']) * position['quantity']
            else:  # SHORT
                raw_pnl = (position['entry_price'] - current_price) * position['quantity']

            # Calculate fees for live P&L display
            position_value = position['entry_price'] * position['quantity']
            entry_fee = position_value * self.config.MAKER_FEE
            exit_fee = current_price * position['quantity'] * self.config.TAKER_FEE
            total_fees = entry_fee + exit_fee

            # Net PnL after fees
            pnl = raw_pnl - total_fees
            pnl_pct = (pnl / position_value) * 100

            # Check exit conditions
            if position['direction'] == 'LONG':
                if current_price <= position['stop_loss']:
                    self.close_position(symbol, current_price, 'STOP_LOSS')
                elif current_price >= position['take_profit']:
                    self.close_position(symbol, current_price, 'TAKE_PROFIT')
            else:  # SHORT
                if current_price >= position['stop_loss']:
                    self.close_position(symbol, current_price, 'STOP_LOSS')
                elif current_price <= position['take_profit']:
                    self.close_position(symbol, current_price, 'TAKE_PROFIT')

    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close a position"""
        position = self.positions.get(symbol)
        if not position:
            return

        # Calculate raw PnL
        if position['direction'] == 'LONG':
            raw_pnl = (exit_price - position['entry_price']) * position['quantity']
        else:
            raw_pnl = (position['entry_price'] - exit_price) * position['quantity']

        # Calculate WooX fees (entry + exit)
        position_value = position['entry_price'] * position['quantity']
        entry_fee = position_value * self.config.MAKER_FEE
        exit_fee = exit_price * position['quantity'] * self.config.TAKER_FEE
        total_fees = entry_fee + exit_fee

        # Net PnL after fees
        pnl = raw_pnl - total_fees
        pnl_pct = (pnl / position_value) * 100

        # Update performance
        self.performance['total_trades'] += 1
        if pnl > 0:
            self.performance['wins'] += 1
        else:
            self.performance['losses'] += 1
        self.performance['total_pnl'] += pnl

        # Add to history
        closed_trade = {
            'symbol': symbol,
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'quantity': position['quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'fees': total_fees,
            'exit_reason': reason,
            'entry_time': position['entry_time'].isoformat() if isinstance(position['entry_time'], datetime) else position['entry_time'],
            'exit_time': datetime.now().isoformat(),
            'burst_type': position.get('burst_type', 'UNKNOWN')
        }
        self.trade_history.append(closed_trade)
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]

        # Save to database
        self._save_trade(closed_trade)

        # Remove position
        del self.positions[symbol]

        logging.info(f"📊 POSITION CLOSED: {symbol}")
        logging.info(f"   Exit reason: {reason}")
        logging.info(f"   Exit price: ${exit_price:.2f}")
        logging.info(f"   Raw P&L: ${raw_pnl:.2f}")
        logging.info(f"   Fees: ${total_fees:.4f}")
        logging.info(f"   Net P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")

    def save_state(self):
        """Save bot state to JSON file for dashboard"""
        try:
            # Calculate current P&L for open positions (with fees)
            positions_list = []
            for symbol, pos in self.positions.items():
                current_price = self.signal_generator.binance.get_current_price(symbol)
                if current_price > 0:
                    # Calculate raw P&L
                    if pos['direction'] == 'LONG':
                        raw_pnl = (current_price - pos['entry_price']) * pos['quantity']
                    else:
                        raw_pnl = (pos['entry_price'] - current_price) * pos['quantity']

                    # Calculate fees
                    position_value = pos['entry_price'] * pos['quantity']
                    entry_fee = position_value * self.config.MAKER_FEE
                    exit_fee = current_price * pos['quantity'] * self.config.TAKER_FEE
                    total_fees = entry_fee + exit_fee

                    # Net P&L after fees
                    pnl = raw_pnl - total_fees
                    pnl_pct = (pnl / position_value) * 100

                    positions_list.append({
                        'symbol': symbol,
                        'direction': pos['direction'],
                        'entry_price': pos['entry_price'],
                        'current_price': current_price,
                        'quantity': pos['quantity'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'fees': total_fees,
                        'stop_loss': pos['stop_loss'],
                        'take_profit': pos['take_profit'],
                        'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else pos['entry_time'],
                        'burst_type': pos.get('burst_type', 'UNKNOWN')
                    })

            # Calculate metrics
            win_rate = 0
            if self.performance['total_trades'] > 0:
                win_rate = (self.performance['wins'] / self.performance['total_trades']) * 100

            state = {
                'timestamp': datetime.now().isoformat(),
                'positions': positions_list,
                'trade_history': self.trade_history[-20:],  # Last 20 trades
                'recent_bursts': self.recent_bursts[-20:],  # Last 20 bursts
                'ai_analyses': self.recent_ai_analyses[-20:],  # Last 20 AI analyses
                'performance': {
                    'total_trades': self.performance['total_trades'],
                    'wins': self.performance['wins'],
                    'losses': self.performance['losses'],
                    'total_pnl': self.performance['total_pnl'],
                    'capital': self.config.INITIAL_CAPITAL + self.performance['total_pnl'],
                    'win_rate': round(win_rate, 1),
                    'best_trade': max([t.get('pnl', 0) for t in self.trade_history] + [0]),
                    'worst_trade': min([t.get('pnl', 0) for t in self.trade_history] + [0])
                },
                'system_status': {
                    'websocket_connected': self.iq_client.running if self.iq_client else False,
                    'ai_enabled': self.ai_analyst is not None,
                    'burst_count': len(self.recent_bursts),
                    'signal_count': len(self.signal_generator.signal_history) if self.signal_generator else 0
                }
            }

            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            logging.error(f"Error saving state: {e}")

    def generate_report(self):
        """Generate performance report"""
        logging.info("\n" + "=" * 60)
        logging.info("PERFORMANCE REPORT")
        logging.info("=" * 60)
        logging.info(f"Total trades: {self.performance['total_trades']}")
        logging.info(f"Wins: {self.performance['wins']}")
        logging.info(f"Losses: {self.performance['losses']}")

        if self.performance['total_trades'] > 0:
            win_rate = (self.performance['wins'] / self.performance['total_trades']) * 100
            logging.info(f"Win rate: {win_rate:.1f}%")

        logging.info(f"Total PnL: ${self.performance['total_pnl']:.2f}")
        logging.info(f"Open positions: {len(self.positions)}")

        for symbol, pos in self.positions.items():
            logging.info(f"  - {symbol}: {pos['direction']} from ${pos['entry_price']:.2f} ({pos['burst_type']})")

        logging.info("=" * 60 + "\n")

    def shutdown(self):
        """Shutdown the bot"""
        if self.iq_client:
            self.iq_client.stop()

        logging.info("Bot shut down successfully")

# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point"""
    bot = CryptoIQTradingBot()
    bot.start()

if __name__ == "__main__":
    main()
