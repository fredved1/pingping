#!/usr/bin/env python3
"""
Advanced Crypto Trading Bot v2.0
Combines Crypto IQ Bursts, Bollinger Bands, and AI Analysis
"""

import os
import json
import time
import sqlite3
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURATION
# ============================================

class Config:
    """Trading bot configuration"""
    
    # API Keys (set as environment variables)
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    
    # Trading Parameters
    INITIAL_CAPITAL = 200.0
    MAX_POSITIONS = 2  # Maximum concurrent positions
    POSITION_SIZE_PCT = 0.3  # 30% of available capital per position
    MAX_LEVERAGE = 1  # No leverage for safety
    
    # Risk Management
    STOP_LOSS_PCT = 2.0  # 2% stop loss
    TAKE_PROFIT_PCT = 3.0  # 3% take profit
    TRAILING_STOP_PCT = 1.5  # 1.5% trailing stop
    
    # Technical Indicators
    BB_PERIOD = 20  # Bollinger Band period
    BB_STD_DEV = 2  # Standard deviations for bands
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    
    # Crypto IQ Burst Detection
    VOLUME_SPIKE_MULTIPLIER = 2.5  # Volume must be 2.5x average
    PRICE_ABSORPTION_THRESHOLD = 0.3  # Max 0.3% price change on high volume
    REVERSAL_CONFIRMATION_BARS = 3  # Bars to confirm reversal
    
    # AI Analysis
    AI_CONFIDENCE_THRESHOLD = 70  # Minimum confidence for trades
    AI_ANALYSIS_INTERVAL = 300  # Analyze every 5 minutes
    
    # Symbols to trade
    TRADING_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'DOGEUSDT']
    
    # Timeframes
    PRIMARY_TIMEFRAME = '5m'
    SECONDARY_TIMEFRAME = '15m'
    
    # Database
    DB_PATH = 'trading_bot.db'
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILE = 'trading_bot.log'

# ============================================
# DATA STRUCTURES
# ============================================

class SignalType(Enum):
    """Types of trading signals"""
    IQ_BURST_LONG = "IQ_BURST_LONG"
    IQ_BURST_SHORT = "IQ_BURST_SHORT"
    BB_SQUEEZE_LONG = "BB_SQUEEZE_LONG"
    BB_SQUEEZE_SHORT = "BB_SQUEEZE_SHORT"
    REVERSAL_LONG = "REVERSAL_LONG"
    REVERSAL_SHORT = "REVERSAL_SHORT"
    AI_LONG = "AI_LONG"
    AI_SHORT = "AI_SHORT"

@dataclass
class MarketData:
    """Market data for a symbol"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    rsi: float = 0.0
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    volume_sma: float = 0.0
    price_change_pct: float = 0.0
    is_iq_burst: bool = False
    is_reversal: bool = False

@dataclass
class TradingSignal:
    """Trading signal with metadata"""
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    direction: str  # 'LONG' or 'SHORT'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    indicators: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Active trading position"""
    id: str
    symbol: str
    direction: str
    entry_price: float
    current_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    entry_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0

# ============================================
# TECHNICAL INDICATORS
# ============================================

class TechnicalAnalyzer:
    """Technical analysis and indicator calculations"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower
    
    @staticmethod
    def detect_iq_burst(data: pd.DataFrame, volume_multiplier: float = 2.5, 
                       price_threshold: float = 0.3) -> pd.Series:
        """
        Detect Crypto IQ Bursts (high volume absorption)
        - High volume with minimal price movement indicates absorption
        - Often precedes significant moves
        """
        volume_sma = data['volume'].rolling(window=20).mean()
        price_change = data['close'].pct_change() * 100
        
        # IQ Burst: High volume + Low price change
        iq_burst = (
            (data['volume'] > volume_sma * volume_multiplier) &
            (abs(price_change) < price_threshold)
        )
        
        return iq_burst
    
    @staticmethod
    def detect_reversal(data: pd.DataFrame, lookback: int = 10) -> Tuple[pd.Series, pd.Series]:
        """
        Detect potential reversal points
        - Price makes new high/low
        - RSI divergence
        - Volume confirmation
        """
        # Calculate indicators
        rsi = TechnicalAnalyzer.calculate_rsi(data['close'])
        
        # Find swing highs and lows
        high_rolling = data['high'].rolling(window=lookback)
        low_rolling = data['low'].rolling(window=lookback)
        
        swing_high = data['high'] == high_rolling.max()
        swing_low = data['low'] == low_rolling.min()
        
        # RSI divergence
        rsi_rolling = rsi.rolling(window=lookback)
        
        # Bearish reversal: Price high + RSI not confirming
        bearish_reversal = swing_high & (rsi < rsi_rolling.max())
        
        # Bullish reversal: Price low + RSI not confirming  
        bullish_reversal = swing_low & (rsi > rsi_rolling.min())
        
        return bullish_reversal, bearish_reversal
    
    @staticmethod
    def calculate_bb_squeeze(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Detect Bollinger Band squeeze
        - Bands are narrow (low volatility)
        - Often precedes breakout
        """
        upper, middle, lower = TechnicalAnalyzer.calculate_bollinger_bands(
            data['close'], period=period
        )
        
        bb_width = upper - lower
        bb_width_sma = bb_width.rolling(window=50).mean()
        bb_width_std = bb_width.rolling(window=50).std()
        
        # Squeeze when width is below average - 0.5 std
        squeeze = bb_width < (bb_width_sma - 0.5 * bb_width_std)
        
        return squeeze

# ============================================
# AI ANALYSIS WITH DEEPSEEK
# ============================================

class AIAnalyst:
    """AI-powered market analysis using Deepseek"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def analyze_market(self, market_data: Dict[str, MarketData], 
                      signals: List[TradingSignal],
                      positions: List[Position]) -> Dict:
        """
        Comprehensive AI analysis of market conditions
        """
        try:
            # Prepare context
            context = self._prepare_context(market_data, signals, positions)
            
            # Create prompt
            prompt = f"""You are an expert crypto trader analyzing real-time market data.

CURRENT MARKET STATE:
{context}

TRADING RULES:
1. Only suggest trades with HIGH confidence (>70%)
2. Look for convergence of multiple signals
3. Consider market correlation and sentiment
4. Avoid overtrading - quality over quantity
5. Focus on IQ Bursts and reversal patterns

Analyze the current setup and provide:
1. Market sentiment (BULLISH/BEARISH/NEUTRAL)
2. Best trading opportunity if any (symbol, direction, confidence)
3. Key reasons for the decision
4. Risk factors to watch

Respond in JSON format:
{{
    "sentiment": "BULLISH/BEARISH/NEUTRAL",
    "trade": {{
        "symbol": "BTCUSDT",
        "direction": "LONG/SHORT/NONE",
        "confidence": 0-100,
        "entry_zone": [price1, price2],
        "stop_loss": price,
        "take_profit": price
    }},
    "reasoning": "detailed explanation",
    "risks": ["risk1", "risk2"]
}}"""

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "You are a professional crypto trading analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                # Parse the JSON content from the AI response
                try:
                    # Strip markdown code blocks if present
                    if content.strip().startswith('```'):
                        content = content.strip()
                        # Remove ```json or ``` at start
                        if content.startswith('```json'):
                            content = content[7:]
                        elif content.startswith('```'):
                            content = content[3:]
                        # Remove ``` at end
                        if '```' in content:
                            content = content[:content.rfind('```')]
                        content = content.strip()

                    analysis = json.loads(content)
                    logging.info(f"✅ AI analysis: {analysis.get('sentiment', 'UNKNOWN')} - {analysis.get('trade', {}).get('symbol', 'NONE')} {analysis.get('trade', {}).get('direction', '')}")
                    return analysis
                except json.JSONDecodeError as e:
                    logging.error(f"Failed to parse AI response: {content[:200]}")
                    return self._default_analysis()
            else:
                logging.error(f"AI API error: {response.status_code} - {response.text[:200]}")
                return self._default_analysis()
                
        except Exception as e:
            logging.error(f"AI analysis error: {str(e)}")
            return self._default_analysis()
    
    def _prepare_context(self, market_data: Dict, signals: List, positions: List) -> str:
        """Prepare market context for AI"""
        context_lines = []
        
        # Market data summary
        for symbol, data in market_data.items():
            context_lines.append(
                f"{symbol}: "
                f"Price=${data.close:.2f} "
                f"RSI={data.rsi:.1f} "
                f"BB_Width={data.bb_width:.2%} "
                f"Volume={'HIGH' if data.volume > data.volume_sma * 2 else 'NORMAL'} "
                f"IQ_Burst={'YES' if data.is_iq_burst else 'NO'} "
                f"Reversal={'YES' if data.is_reversal else 'NO'}"
            )
        
        # Recent signals
        if signals:
            context_lines.append("\nRECENT SIGNALS:")
            for signal in signals[-5:]:
                context_lines.append(
                    f"- {signal.symbol} {signal.direction} "
                    f"(Confidence: {signal.confidence:.0f}%)"
                )
        
        # Open positions
        if positions:
            context_lines.append("\nOPEN POSITIONS:")
            for pos in positions:
                context_lines.append(
                    f"- {pos.symbol} {pos.direction} "
                    f"PnL: {pos.pnl_pct:.2f}%"
                )
        
        return "\n".join(context_lines)
    
    def _default_analysis(self) -> Dict:
        """Default analysis when AI is unavailable"""
        return {
            "sentiment": "NEUTRAL",
            "trade": {
                "symbol": "",
                "direction": "NONE",
                "confidence": 0,
                "entry_zone": [0, 0],
                "stop_loss": 0,
                "take_profit": 0
            },
            "reasoning": "AI analysis unavailable",
            "risks": ["AI system offline"]
        }

# ============================================
# SIGNAL GENERATOR
# ============================================

class SignalGenerator:
    """Generate trading signals from multiple sources"""
    
    def __init__(self, config: Config):
        self.config = config
        self.analyzer = TechnicalAnalyzer()
        self.ai_analyst = AIAnalyst(config.DEEPSEEK_API_KEY)
        self.last_ai_analysis = {}
        self.last_ai_time = datetime.now() - timedelta(seconds=config.AI_ANALYSIS_INTERVAL)
    
    def generate_signals(self, market_data: Dict[str, pd.DataFrame], 
                        positions: List[Position]) -> List[TradingSignal]:
        """Generate trading signals from all sources"""
        signals = []
        
        # Process each symbol
        processed_data = {}
        for symbol, df in market_data.items():
            # Calculate indicators
            data = self._process_market_data(symbol, df)
            processed_data[symbol] = data
            
            # Generate technical signals
            tech_signals = self._generate_technical_signals(data)
            signals.extend(tech_signals)
        
        # AI Analysis (rate limited)
        if (datetime.now() - self.last_ai_time).seconds > self.config.AI_ANALYSIS_INTERVAL:
            ai_signals = self._generate_ai_signals(processed_data, signals, positions)
            signals.extend(ai_signals)
            self.last_ai_time = datetime.now()
        
        # Filter and rank signals
        filtered_signals = self._filter_signals(signals, positions)
        
        return filtered_signals
    
    def _process_market_data(self, symbol: str, df: pd.DataFrame) -> MarketData:
        """Process raw market data into structured format"""
        
        # Calculate indicators
        df['rsi'] = self.analyzer.calculate_rsi(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.analyzer.calculate_bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['price_change_pct'] = df['close'].pct_change() * 100
        df['iq_burst'] = self.analyzer.detect_iq_burst(df)
        df['bullish_reversal'], df['bearish_reversal'] = self.analyzer.detect_reversal(df)
        df['bb_squeeze'] = self.analyzer.calculate_bb_squeeze(df)
        
        # Get latest values
        latest = df.iloc[-1]
        
        return MarketData(
            symbol=symbol,
            timestamp=datetime.now(),
            open=latest['open'],
            high=latest['high'],
            low=latest['low'],
            close=latest['close'],
            volume=latest['volume'],
            rsi=latest['rsi'],
            bb_upper=latest['bb_upper'],
            bb_middle=latest['bb_middle'],
            bb_lower=latest['bb_lower'],
            bb_width=latest['bb_width'],
            volume_sma=latest['volume_sma'],
            price_change_pct=latest['price_change_pct'],
            is_iq_burst=latest['iq_burst'],
            is_reversal=latest['bullish_reversal'] or latest['bearish_reversal']
        )
    
    def _generate_technical_signals(self, data: MarketData) -> List[TradingSignal]:
        """Generate signals from technical indicators"""
        signals = []
        
        # IQ Burst Signals
        if data.is_iq_burst:
            volume_multiplier = data.volume / data.volume_sma if data.volume_sma > 0 else 0
            logging.info(f"🔥 IQ BURST DETECTED: {data.symbol} - Volume: {volume_multiplier:.2f}x, Price: ${data.close:.2f}, RSI: {data.rsi:.1f}")

            # Determine direction based on price position
            if data.close < data.bb_lower and data.rsi < self.config.RSI_OVERSOLD:
                logging.info(f"✅ IQ BURST LONG signal generated for {data.symbol}")
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.IQ_BURST_LONG,
                    direction="LONG",
                    confidence=85,
                    entry_price=data.close,
                    stop_loss=data.close * (1 - self.config.STOP_LOSS_PCT/100),
                    take_profit=data.close * (1 + self.config.TAKE_PROFIT_PCT/100),
                    reason="IQ Burst detected at BB lower band with oversold RSI",
                    indicators={
                        "rsi": data.rsi,
                        "bb_position": "below_lower",
                        "volume_spike": data.volume / data.volume_sma
                    }
                ))
            elif data.close > data.bb_upper and data.rsi > self.config.RSI_OVERBOUGHT:
                logging.info(f"✅ IQ BURST SHORT signal generated for {data.symbol}")
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.IQ_BURST_SHORT,
                    direction="SHORT",
                    confidence=85,
                    entry_price=data.close,
                    stop_loss=data.close * (1 + self.config.STOP_LOSS_PCT/100),
                    take_profit=data.close * (1 - self.config.TAKE_PROFIT_PCT/100),
                    reason="IQ Burst detected at BB upper band with overbought RSI",
                    indicators={
                        "rsi": data.rsi,
                        "bb_position": "above_upper",
                        "volume_spike": data.volume / data.volume_sma
                    }
                ))
            else:
                logging.info(f"⚠️  IQ BURST for {data.symbol} but no entry conditions met (RSI: {data.rsi:.1f}, BB pos: {data.close:.2f} vs [{data.bb_lower:.2f}, {data.bb_upper:.2f}])")
        
        # Reversal Signals
        if data.is_reversal:
            if data.rsi < self.config.RSI_OVERSOLD:
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.REVERSAL_LONG,
                    direction="LONG",
                    confidence=75,
                    entry_price=data.close,
                    stop_loss=data.close * (1 - self.config.STOP_LOSS_PCT/100),
                    take_profit=data.close * (1 + self.config.TAKE_PROFIT_PCT/100),
                    reason="Bullish reversal pattern detected",
                    indicators={"rsi": data.rsi, "pattern": "bullish_reversal"}
                ))
            elif data.rsi > self.config.RSI_OVERBOUGHT:
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.REVERSAL_SHORT,
                    direction="SHORT",
                    confidence=75,
                    entry_price=data.close,
                    stop_loss=data.close * (1 + self.config.STOP_LOSS_PCT/100),
                    take_profit=data.close * (1 - self.config.TAKE_PROFIT_PCT/100),
                    reason="Bearish reversal pattern detected",
                    indicators={"rsi": data.rsi, "pattern": "bearish_reversal"}
                ))
        
        # Bollinger Band Squeeze Breakout
        if data.bb_width < 0.02:  # Tight squeeze
            if data.close > data.bb_middle and data.rsi > 50:
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.BB_SQUEEZE_LONG,
                    direction="LONG",
                    confidence=70,
                    entry_price=data.close,
                    stop_loss=data.bb_lower,
                    take_profit=data.bb_upper,
                    reason="BB squeeze breakout to upside",
                    indicators={"bb_width": data.bb_width, "rsi": data.rsi}
                ))
            elif data.close < data.bb_middle and data.rsi < 50:
                signals.append(TradingSignal(
                    timestamp=data.timestamp,
                    symbol=data.symbol,
                    signal_type=SignalType.BB_SQUEEZE_SHORT,
                    direction="SHORT",
                    confidence=70,
                    entry_price=data.close,
                    stop_loss=data.bb_upper,
                    take_profit=data.bb_lower,
                    reason="BB squeeze breakout to downside",
                    indicators={"bb_width": data.bb_width, "rsi": data.rsi}
                ))
        
        return signals
    
    def _generate_ai_signals(self, market_data: Dict[str, MarketData], 
                           tech_signals: List[TradingSignal],
                           positions: List[Position]) -> List[TradingSignal]:
        """Generate signals from AI analysis"""
        signals = []
        
        # Get AI analysis
        analysis = self.ai_analyst.analyze_market(market_data, tech_signals, positions)
        self.last_ai_analysis = analysis
        
        # Convert AI recommendation to signal
        if analysis['trade']['direction'] != 'NONE':
            trade = analysis['trade']
            
            # Determine signal type
            signal_type = SignalType.AI_LONG if trade['direction'] == 'LONG' else SignalType.AI_SHORT
            
            # Use mid-point of entry zone
            entry_price = (trade['entry_zone'][0] + trade['entry_zone'][1]) / 2
            
            signals.append(TradingSignal(
                timestamp=datetime.now(),
                symbol=trade['symbol'],
                signal_type=signal_type,
                direction=trade['direction'],
                confidence=trade['confidence'],
                entry_price=entry_price,
                stop_loss=trade['stop_loss'],
                take_profit=trade['take_profit'],
                reason=analysis['reasoning'],
                indicators={"ai_sentiment": analysis['sentiment']}
            ))
        
        return signals
    
    def _filter_signals(self, signals: List[TradingSignal], 
                       positions: List[Position]) -> List[TradingSignal]:
        """Filter and prioritize signals"""
        
        # Filter by confidence threshold
        filtered = [s for s in signals if s.confidence >= self.config.AI_CONFIDENCE_THRESHOLD]
        
        # Remove signals for symbols with open positions
        open_symbols = {pos.symbol for pos in positions}
        filtered = [s for s in filtered if s.symbol not in open_symbols]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Limit to max positions
        max_new = self.config.MAX_POSITIONS - len(positions)
        filtered = filtered[:max_new]
        
        return filtered

# ============================================
# TRADING ENGINE
# ============================================

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.signal_generator = SignalGenerator(config)
        self.positions: List[Position] = []
        self.closed_trades = []
        self.capital = config.INITIAL_CAPITAL
        self.available_capital = config.INITIAL_CAPITAL
        
        # Initialize database
        self._init_database()
        
        # Setup logging
        logging.basicConfig(
            level=config.LOG_LEVEL,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(config.LOG_FILE),
                logging.StreamHandler()
            ]
        )
    
    def _init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.config.DB_PATH)
        cursor = conn.cursor()
        
        # Create tables
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
    
    def fetch_market_data(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> pd.DataFrame:
        """Fetch market data from Binance"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
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
                
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                return df
            else:
                logging.error(f"Failed to fetch data for {symbol}: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error fetching market data: {str(e)}")
            return pd.DataFrame()
    
    def execute_trade(self, signal: TradingSignal) -> Optional[Position]:
        """Execute a trade based on signal"""
        try:
            # Calculate position size
            position_value = self.available_capital * self.config.POSITION_SIZE_PCT
            quantity = position_value / signal.entry_price
            
            # Create position
            position = Position(
                id=f"{signal.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                symbol=signal.symbol,
                direction=signal.direction,
                entry_price=signal.entry_price,
                current_price=signal.entry_price,
                quantity=quantity,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                trailing_stop=None,
                entry_time=signal.timestamp,
                highest_price=signal.entry_price if signal.direction == "LONG" else 0,
                lowest_price=signal.entry_price if signal.direction == "SHORT" else float('inf')
            )
            
            # Update capital
            self.available_capital -= position_value
            self.positions.append(position)
            
            # Log trade
            logging.info(f"TRADE EXECUTED: {signal.symbol} {signal.direction} @ {signal.entry_price:.4f}")
            logging.info(f"Reason: {signal.reason}")
            logging.info(f"Confidence: {signal.confidence:.0f}%")
            
            # Save to database
            self._save_signal(signal, executed=True)
            
            return position
            
        except Exception as e:
            logging.error(f"Failed to execute trade: {str(e)}")
            return None
    
    def update_positions(self, market_data: Dict[str, pd.DataFrame]):
        """Update open positions with current prices"""
        for position in self.positions:
            if position.symbol in market_data:
                df = market_data[position.symbol]
                if not df.empty:
                    current_price = df['close'].iloc[-1]
                    position.current_price = current_price
                    
                    # Update PnL
                    if position.direction == "LONG":
                        position.pnl = (current_price - position.entry_price) * position.quantity
                        position.pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                        
                        # Update highest price for trailing stop
                        position.highest_price = max(position.highest_price, current_price)
                        
                        # Check exit conditions
                        if current_price <= position.stop_loss:
                            self._close_position(position, current_price, "STOP_LOSS")
                        elif current_price >= position.take_profit:
                            self._close_position(position, current_price, "TAKE_PROFIT")
                        elif position.trailing_stop and current_price <= position.trailing_stop:
                            self._close_position(position, current_price, "TRAILING_STOP")
                        
                        # Update trailing stop if in profit
                        if position.pnl_pct > self.config.TRAILING_STOP_PCT:
                            position.trailing_stop = position.highest_price * (1 - self.config.TRAILING_STOP_PCT/100)
                    
                    else:  # SHORT
                        position.pnl = (position.entry_price - current_price) * position.quantity
                        position.pnl_pct = ((position.entry_price - current_price) / position.entry_price) * 100
                        
                        # Update lowest price for trailing stop
                        position.lowest_price = min(position.lowest_price, current_price)
                        
                        # Check exit conditions
                        if current_price >= position.stop_loss:
                            self._close_position(position, current_price, "STOP_LOSS")
                        elif current_price <= position.take_profit:
                            self._close_position(position, current_price, "TAKE_PROFIT")
                        elif position.trailing_stop and current_price >= position.trailing_stop:
                            self._close_position(position, current_price, "TRAILING_STOP")
                        
                        # Update trailing stop if in profit
                        if position.pnl_pct > self.config.TRAILING_STOP_PCT:
                            position.trailing_stop = position.lowest_price * (1 + self.config.TRAILING_STOP_PCT/100)
    
    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position"""
        try:
            # Calculate final PnL
            if position.direction == "LONG":
                final_pnl = (exit_price - position.entry_price) * position.quantity
            else:
                final_pnl = (position.entry_price - exit_price) * position.quantity
            
            final_pnl_pct = (final_pnl / (position.entry_price * position.quantity)) * 100
            
            # Update capital
            self.available_capital += (position.entry_price * position.quantity) + final_pnl
            
            # Log closure
            logging.info(f"POSITION CLOSED: {position.symbol} {position.direction}")
            logging.info(f"Exit Price: {exit_price:.4f}, Reason: {reason}")
            logging.info(f"PnL: ${final_pnl:.2f} ({final_pnl_pct:.2f}%)")
            
            # Save to database
            self._save_closed_trade(position, exit_price, final_pnl, final_pnl_pct, reason)
            
            # Remove from active positions
            self.positions.remove(position)
            self.closed_trades.append(position)
            
        except Exception as e:
            logging.error(f"Error closing position: {str(e)}")
    
    def _save_signal(self, signal: TradingSignal, executed: bool):
        """Save signal to database"""
        conn = sqlite3.connect(self.config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (timestamp, symbol, signal_type, direction, confidence, executed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            signal.timestamp.isoformat(),
            signal.symbol,
            signal.signal_type.value,
            signal.direction,
            signal.confidence,
            1 if executed else 0
        ))
        
        conn.commit()
        conn.close()
    
    def _save_closed_trade(self, position: Position, exit_price: float, 
                          pnl: float, pnl_pct: float, reason: str):
        """Save closed trade to database"""
        conn = sqlite3.connect(self.config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (id, symbol, direction, entry_price, exit_price, 
                              quantity, entry_time, exit_time, pnl, pnl_pct, exit_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            position.id,
            position.symbol,
            position.direction,
            position.entry_price,
            exit_price,
            position.quantity,
            position.entry_time.isoformat(),
            datetime.now().isoformat(),
            pnl,
            pnl_pct,
            reason
        ))
        
        conn.commit()
        conn.close()
    
    def save_performance(self):
        """Save current performance metrics"""
        # Calculate metrics
        total_equity = self.available_capital + sum(p.pnl for p in self.positions)
        
        # Calculate win rate from closed trades
        if self.closed_trades:
            wins = sum(1 for t in self.closed_trades if t.pnl > 0)
            win_rate = (wins / len(self.closed_trades)) * 100
        else:
            win_rate = 0
        
        # Daily PnL
        daily_pnl = total_equity - self.config.INITIAL_CAPITAL
        
        # Save to database
        conn = sqlite3.connect(self.config.DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance (timestamp, equity, positions, daily_pnl, win_rate)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            total_equity,
            len(self.positions),
            daily_pnl,
            win_rate
        ))
        
        conn.commit()
        conn.close()
    
    def generate_report(self) -> str:
        """Generate trading report"""
        report = []
        report.append("=" * 60)
        report.append("TRADING BOT PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Portfolio summary
        total_equity = self.available_capital + sum(p.pnl for p in self.positions)
        report.append("PORTFOLIO SUMMARY:")
        report.append(f"Initial Capital: ${self.config.INITIAL_CAPITAL:.2f}")
        report.append(f"Current Equity: ${total_equity:.2f}")
        report.append(f"Available Cash: ${self.available_capital:.2f}")
        report.append(f"Open Positions: {len(self.positions)}")
        report.append(f"Total Return: {((total_equity - self.config.INITIAL_CAPITAL) / self.config.INITIAL_CAPITAL) * 100:.2f}%")
        report.append("")
        
        # Open positions
        if self.positions:
            report.append("OPEN POSITIONS:")
            for pos in self.positions:
                report.append(
                    f"- {pos.symbol} {pos.direction}: "
                    f"Entry=${pos.entry_price:.4f}, "
                    f"Current=${pos.current_price:.4f}, "
                    f"PnL={pos.pnl_pct:.2f}%"
                )
            report.append("")
        
        # Trading statistics
        if self.closed_trades:
            wins = sum(1 for t in self.closed_trades if t.pnl > 0)
            losses = len(self.closed_trades) - wins
            win_rate = (wins / len(self.closed_trades)) * 100
            
            report.append("TRADING STATISTICS:")
            report.append(f"Total Trades: {len(self.closed_trades)}")
            report.append(f"Wins: {wins}, Losses: {losses}")
            report.append(f"Win Rate: {win_rate:.1f}%")
            
            if wins > 0:
                avg_win = sum(t.pnl for t in self.closed_trades if t.pnl > 0) / wins
                report.append(f"Average Win: ${avg_win:.2f}")
            
            if losses > 0:
                avg_loss = sum(t.pnl for t in self.closed_trades if t.pnl < 0) / losses
                report.append(f"Average Loss: ${avg_loss:.2f}")
            
            report.append("")
        
        # Recent AI analysis
        if hasattr(self.signal_generator, 'last_ai_analysis') and self.signal_generator.last_ai_analysis:
            analysis = self.signal_generator.last_ai_analysis
            report.append("LATEST AI ANALYSIS:")
            report.append(f"Market Sentiment: {analysis.get('sentiment', 'N/A')}")
            if analysis.get('trade', {}).get('direction') != 'NONE':
                trade = analysis['trade']
                report.append(f"Recommendation: {trade['symbol']} {trade['direction']} (Confidence: {trade['confidence']}%)")
            report.append(f"Reasoning: {analysis.get('reasoning', 'N/A')[:200]}...")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    async def run(self):
        """Main trading loop"""
        logging.info("Trading bot started")
        logging.info(f"Initial capital: ${self.config.INITIAL_CAPITAL}")
        logging.info(f"Trading symbols: {', '.join(self.config.TRADING_SYMBOLS)}")
        
        while True:
            try:
                # Fetch market data
                market_data = {}
                for symbol in self.config.TRADING_SYMBOLS:
                    df = self.fetch_market_data(symbol, self.config.PRIMARY_TIMEFRAME)
                    if not df.empty:
                        market_data[symbol] = df
                
                if market_data:
                    # Update existing positions
                    self.update_positions(market_data)
                    
                    # Generate new signals
                    signals = self.signal_generator.generate_signals(market_data, self.positions)
                    
                    # Execute trades
                    for signal in signals:
                        if len(self.positions) < self.config.MAX_POSITIONS:
                            self.execute_trade(signal)
                    
                    # Save performance
                    self.save_performance()
                    
                    # Generate and display report
                    if datetime.now().minute % 15 == 0:  # Every 15 minutes
                        report = self.generate_report()
                        print(report)
                        
                        # Save report to file
                        with open('trading_report.txt', 'w') as f:
                            f.write(report)
                
                # Sleep before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logging.info("Shutting down...")
                break
            except Exception as e:
                logging.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(60)

# ============================================
# MAIN ENTRY POINT
# ============================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("ADVANCED CRYPTO TRADING BOT v2.0")
    print("Combining IQ Bursts, Bollinger Bands & AI Analysis")
    print("=" * 60)
    
    # Check for API keys
    config = Config()
    
    if not config.DEEPSEEK_API_KEY:
        print("WARNING: DEEPSEEK_API_KEY not set. AI analysis will be disabled.")
        print("Set it with: export DEEPSEEK_API_KEY='your_key_here'")
    
    # Create and run trading engine
    engine = TradingEngine(config)
    
    # Run async event loop
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\nBot stopped by user")
        # Generate final report
        report = engine.generate_report()
        print(report)
        
        # Save final report
        with open('final_trading_report.txt', 'w') as f:
            f.write(report)

if __name__ == "__main__":
    main()
