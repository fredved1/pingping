#!/usr/bin/env python3
"""
Crypto IQ Dashboard Server
Real-time web dashboard for monitoring the trading bot
"""

from flask import Flask, render_template, jsonify
from flask_cors import CORS
import json
import time
import os
import sqlite3
from datetime import datetime
from collections import deque
import threading

app = Flask(__name__)
CORS(app)

# Global data store
class DashboardData:
    def __init__(self):
        self.state_file = 'bot_state.json'
        self.db_path = 'trading_bot.db'
        self.start_time = time.time()
        self.burst_stats = {
            'patterns': {'absorption': 0, 'breakout': 0, 'reversal': 0},
            'symbols': {}
        }

    def update_from_state(self):
        """Read bot state from JSON file"""
        if not os.path.exists(self.state_file):
            # Return empty state if file doesn't exist yet
            return {
                'positions': [],
                'recent_bursts': [],
                'trade_history': [],
                'performance': {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'capital': 200.0,
                    'win_rate': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                },
                'system_status': {
                    'websocket_connected': False,
                    'ai_enabled': False,
                    'last_update': datetime.now().isoformat(),
                    'uptime_seconds': int(time.time() - self.start_time),
                    'burst_count': 0,
                    'signal_count': 0
                }
            }

        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Update burst stats from bursts
            for burst in state.get('recent_bursts', []):
                pattern = burst.get('pattern', 'none')
                if pattern in self.burst_stats['patterns']:
                    self.burst_stats['patterns'][pattern] += 1

            # Add uptime
            state['system_status']['uptime_seconds'] = int(time.time() - self.start_time)
            state['system_status']['last_update'] = datetime.now().isoformat()

            return state

        except Exception as e:
            print(f"Error reading state file: {e}")
            return None

    def get_trades_from_db(self, limit=20):
        """Get recent trades from database"""
        if not os.path.exists(self.db_path):
            return []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT symbol, direction, entry_price, exit_price, quantity,
                       entry_time, exit_time, pnl, pnl_pct, exit_reason
                FROM trades
                ORDER BY exit_time DESC
                LIMIT ?
            ''', (limit,))

            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'symbol': row[0],
                    'direction': row[1],
                    'entry_price': row[2],
                    'exit_price': row[3],
                    'quantity': row[4],
                    'entry_time': row[5],
                    'exit_time': row[6],
                    'pnl': row[7],
                    'pnl_pct': row[8],
                    'exit_reason': row[9]
                })

            conn.close()
            return trades

        except Exception as e:
            print(f"Error reading trades from database: {e}")
            return []

    def calculate_performance_from_db(self):
        """Calculate performance metrics from all database trades"""
        if not os.path.exists(self.db_path):
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'capital': 200.0,
                'win_rate': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get all trades
            cursor.execute('SELECT pnl, pnl_pct FROM trades')
            trades = cursor.fetchall()
            conn.close()

            if not trades:
                return {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'capital': 200.0,
                    'win_rate': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                }

            total_trades = len(trades)
            wins = sum(1 for t in trades if t[0] > 0)
            losses = total_trades - wins
            total_pnl = sum(t[0] for t in trades)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            best_trade = max(t[0] for t in trades) if trades else 0
            worst_trade = min(t[0] for t in trades) if trades else 0
            capital = 200.0 + total_pnl

            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'total_pnl': round(total_pnl, 2),
                'capital': round(capital, 2),
                'win_rate': round(win_rate, 2),
                'best_trade': round(best_trade, 2),
                'worst_trade': round(worst_trade, 2)
            }

        except Exception as e:
            print(f"Error calculating performance: {e}")
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'capital': 200.0,
                'win_rate': 0,
                'best_trade': 0,
                'worst_trade': 0
            }

    def get_dashboard_data(self):
        """Get all dashboard data"""
        state = self.update_from_state()
        if not state:
            return {
                'positions': [],
                'recent_bursts': [],
                'trade_history': [],
                'performance': {
                    'total_trades': 0,
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0,
                    'capital': 200.0,
                    'win_rate': 0,
                    'best_trade': 0,
                    'worst_trade': 0
                },
                'market_data': {},
                'ai_analysis': {},
                'system_status': {
                    'websocket_connected': False,
                    'ai_enabled': False,
                    'last_update': datetime.now().isoformat(),
                    'uptime_seconds': 0,
                    'burst_count': 0,
                    'signal_count': 0
                },
                'burst_stats': self.burst_stats,
                'timestamp': datetime.now().isoformat()
            }

        # Load trades from database (combines historical + recent)
        db_trades = self.get_trades_from_db(limit=20)

        # Merge with recent trades from state if not in DB yet
        state_trades = state.get('trade_history', [])
        all_trades = db_trades if db_trades else state_trades[-10:]

        # Calculate performance from ALL database trades
        performance = self.calculate_performance_from_db()

        # Calculate burst stats from recent bursts (reset each time to avoid accumulation)
        burst_stats = {
            'patterns': {'absorption': 0, 'breakout': 0, 'reversal': 0, 'none': 0},
            'symbols': {}
        }

        for burst in state.get('recent_bursts', []):
            pattern = burst.get('pattern', 'none')
            if pattern in burst_stats['patterns']:
                burst_stats['patterns'][pattern] += 1

            symbol = burst.get('symbol', 'unknown')
            if symbol not in burst_stats['symbols']:
                burst_stats['symbols'][symbol] = 0
            burst_stats['symbols'][symbol] += 1

        return {
            'positions': state.get('positions', []),
            'recent_bursts': state.get('recent_bursts', []),
            'trade_history': all_trades,
            'ai_analyses': state.get('ai_analyses', []),
            'performance': performance,
            'market_data': {},
            'ai_analysis': {},
            'system_status': state.get('system_status', {}),
            'burst_stats': burst_stats,
            'timestamp': datetime.now().isoformat()
        }

dashboard_data = DashboardData()

# No background thread needed - data read from JSON on each API call

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint for dashboard data"""
    return jsonify(dashboard_data.get_dashboard_data())

@app.route('/api/logs')
def get_logs():
    """Get recent logs"""
    try:
        with open(dashboard_data.log_file, 'r') as f:
            lines = f.readlines()
        return jsonify({
            'logs': lines[-100:],  # Last 100 lines
            'total_lines': len(lines)
        })
    except:
        return jsonify({'logs': [], 'total_lines': 0})

if __name__ == '__main__':
    print("=" * 60)
    print("CRYPTO IQ DASHBOARD SERVER")
    print("=" * 60)
    print("Dashboard URL: http://localhost:5001")
    print("API Endpoint: http://localhost:5001/api/data")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=5001, threaded=True)
