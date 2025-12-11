"""
Self-Learning Strategy Optimizer
Automatically tests different strategies, keeps what works, and continuously improves
Includes advanced learning: time-based, volatility, market regime, patterns, correlations, etc.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import config
from ai_analyzer import AIAnalyzer
import json
import os


class Strategy:
    """Represents a trading strategy with its parameters"""
    def __init__(self, name: str, params: Dict):
        self.name = name
        self.params = params
        self.performance = {
            'win_rate': 0.0,
            'total_signals': 0,
            'profit': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'last_tested': None,
            'tests_count': 0
        }
    
    def to_dict(self):
        return {
            'name': self.name,
            'params': self.params,
            'performance': self.performance
        }
    
    @classmethod
    def from_dict(cls, data):
        strategy = cls(data['name'], data['params'])
        strategy.performance = data.get('performance', strategy.performance)
        return strategy


class StrategyOptimizer:
    """
    Self-learning system that:
    1. Tests different strategy combinations
    2. Backtests each one
    3. Keeps strategies that work
    4. Continuously improves
    """
    
    def __init__(self, ai_analyzer: AIAnalyzer):
        self.ai_analyzer = ai_analyzer
        self.strategies_file = 'models/strategies.json'
        self.strategies: List[Strategy] = []
        self.best_strategies: List[Strategy] = []
        self.coin_specific_params: Dict[str, Dict] = {}  # Coin-specific TP/SL
        self.error_patterns: List[Dict] = []  # Learned error patterns
        self.load_strategies()
        
        # Strategy components to test
        self.strategy_components = {
            'timeframes': ['1m', '5m', '15m', '1h'],
            'indicators': ['rsi', 'macd', 'bollinger', 'stochastic', 'volume'],
            'entry_rules': ['breakout', 'pullback', 'momentum', 'reversal'],
            'exit_rules': ['fixed', 'atr_based', 'trailing', 'time_based'],
            'filters': ['volume_spike', 'correlation', 'regime', 'multi_tf']
        }
        
        # Time-based learning data
        self.time_performance: Dict[str, Dict] = {}  # hour -> performance stats
        self.day_performance: Dict[str, Dict] = {}  # day_of_week -> performance stats
        
        # Market regime detection
        self.market_regime_history: List[Dict] = []
    
    def load_strategies(self):
        """Load saved strategies from disk"""
        if os.path.exists(self.strategies_file):
            try:
                with open(self.strategies_file, 'r') as f:
                    data = json.load(f)
                    self.strategies = [Strategy.from_dict(s) for s in data.get('strategies', [])]
                    self.best_strategies = [Strategy.from_dict(s) for s in data.get('best_strategies', [])]
                    self.coin_specific_params = data.get('coin_specific_params', {})
                    self.error_patterns = data.get('error_patterns', [])
                    self.time_performance = data.get('time_performance', {})
                    self.day_performance = data.get('day_performance', {})
                    print(f"✅ Loaded {len(self.strategies)} strategies ({len(self.best_strategies)} best)")
                    print(f"   📊 Loaded {len(self.coin_specific_params)} coin-specific params")
                    print(f"   ⚠️  Loaded {len(self.error_patterns)} error patterns")
            except Exception as e:
                print(f"Error loading strategies: {e}")
                self.strategies = []
                self.best_strategies = []
    
    def save_strategies(self):
        """Save strategies to disk with all learning data"""
        try:
            os.makedirs('models', exist_ok=True)
            data = {
                'strategies': [s.to_dict() for s in self.strategies],
                'best_strategies': [s.to_dict() for s in self.best_strategies],
                'coin_specific_params': self.coin_specific_params,
                'error_patterns': self.error_patterns[-100:],  # Keep last 100 errors
                'time_performance': self.time_performance,
                'day_performance': self.day_performance,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.strategies_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving strategies: {e}")
    
    def generate_strategy_variations(self) -> List[Strategy]:
        """Generate different strategy variations to test"""
        variations = []
        
        # Base strategies with different combinations
        base_configs = [
            # Conservative (high precision)
            {
                'name': 'conservative_multi_tf',
                'min_confidence': 0.60,
                'min_ai_confidence': 0.65,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'use_correlation_filter': False,  # Will be tested
                'use_btc_correlation': False,  # Will be tested
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.8
            },
            # Aggressive (more signals)
            {
                'name': 'aggressive_momentum',
                'min_confidence': 0.45,
                'min_ai_confidence': 0.50,
                'use_multi_timeframe': False,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': False,  # Will be tested
                'tp_multiplier': 1.5,
                'sl_multiplier': 1.0
            },
            # Balanced
            {
                'name': 'balanced_atr',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': True,  # Test BTC correlation
                'use_atr_based_tp_sl': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            # Scalping (quick in/out)
            {
                'name': 'scalping_fast',
                'min_confidence': 0.50,
                'min_ai_confidence': 0.55,
                'use_multi_timeframe': False,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': False,  # Scalping less affected by BTC
                'tp_multiplier': 0.8,
                'sl_multiplier': 0.6,
                'max_hold_time_minutes': 30
            },
            # Swing (longer hold)
            {
                'name': 'swing_trend',
                'min_confidence': 0.58,
                'min_ai_confidence': 0.63,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': True,  # Swing more affected by BTC
                'use_trailing_stop': True,
                'tp_multiplier': 2.0,
                'sl_multiplier': 1.2
            },
            # BTC Correlation Test Strategies
            {
                'name': 'btc_correlation_strict',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': True,
                'btc_correlation_threshold': -0.02,  # Block if BTC drops >2%
                'btc_correlation_required': True,  # Must have BTC alignment
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.9
            },
            {
                'name': 'btc_correlation_moderate',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'use_correlation_filter': False,
                'use_btc_correlation': True,
                'btc_correlation_threshold': -0.03,  # Block if BTC drops >3%
                'btc_correlation_required': False,  # Warning only
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.9
            },
            # Time-Based Strategies
            {
                'name': 'time_optimized_asia',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_hours': [0, 1, 2, 3, 4, 5, 6, 7, 8],  # Asia session
                'use_time_filter': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            {
                'name': 'time_optimized_europe',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_hours': [8, 9, 10, 11, 12, 13, 14, 15, 16],  # Europe session
                'use_time_filter': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            {
                'name': 'time_optimized_us',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_hours': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22],  # US session
                'use_time_filter': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            # Volatility-Based Strategies
            {
                'name': 'volatility_high_atr',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_atr_based_tp_sl': True,
                'atr_multiplier_tp': 2.5,  # Higher TP for volatile coins
                'atr_multiplier_sl': 1.5,
                'min_volatility_threshold': 0.03,  # Only high volatility
                'tp_multiplier': 1.5,
                'sl_multiplier': 1.2
            },
            {
                'name': 'volatility_low_atr',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_atr_based_tp_sl': True,
                'atr_multiplier_tp': 1.5,  # Lower TP for stable coins
                'atr_multiplier_sl': 1.0,
                'max_volatility_threshold': 0.02,  # Only low volatility
                'tp_multiplier': 0.8,
                'sl_multiplier': 0.7
            },
            # Market Regime Strategies
            {
                'name': 'regime_trending',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_regime': 'trending',
                'use_regime_filter': True,
                'use_trailing_stop': True,
                'tp_multiplier': 2.0,
                'sl_multiplier': 1.0
            },
            {
                'name': 'regime_ranging',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_regime': 'ranging',
                'use_regime_filter': True,
                'tp_multiplier': 0.6,
                'sl_multiplier': 0.5,
                'max_hold_time_minutes': 60
            },
            {
                'name': 'regime_volatile',
                'min_confidence': 0.60,
                'min_ai_confidence': 0.65,
                'preferred_regime': 'volatile',
                'use_regime_filter': True,
                'tp_multiplier': 1.2,
                'sl_multiplier': 1.0
            },
            # Optimal Hold Time Strategies
            {
                'name': 'hold_time_15min',
                'min_confidence': 0.50,
                'min_ai_confidence': 0.55,
                'max_hold_time_minutes': 15,
                'tp_multiplier': 0.6,
                'sl_multiplier': 0.5
            },
            {
                'name': 'hold_time_30min',
                'min_confidence': 0.52,
                'min_ai_confidence': 0.57,
                'max_hold_time_minutes': 30,
                'tp_multiplier': 0.8,
                'sl_multiplier': 0.7
            },
            {
                'name': 'hold_time_60min',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'max_hold_time_minutes': 60,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            {
                'name': 'hold_time_120min',
                'min_confidence': 0.57,
                'min_ai_confidence': 0.62,
                'max_hold_time_minutes': 120,
                'tp_multiplier': 1.3,
                'sl_multiplier': 1.1
            },
            {
                'name': 'hold_time_240min',
                'min_confidence': 0.58,
                'min_ai_confidence': 0.63,
                'max_hold_time_minutes': 240,
                'tp_multiplier': 1.8,
                'sl_multiplier': 1.3
            },
            # Volume-Based Strategies
            {
                'name': 'volume_high_threshold',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'min_volume_spike': 2.0,  # Only high volume spikes
                'use_volume_filter': True,
                'tp_multiplier': 1.2,
                'sl_multiplier': 1.0
            },
            {
                'name': 'volume_moderate_threshold',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'min_volume_spike': 1.4,  # Moderate volume
                'use_volume_filter': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            # Pattern-Based Strategies
            {
                'name': 'pattern_breakout',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_pattern': 'breakout',
                'use_pattern_filter': True,
                'tp_multiplier': 1.5,
                'sl_multiplier': 1.0
            },
            {
                'name': 'pattern_reversal',
                'min_confidence': 0.58,
                'min_ai_confidence': 0.63,
                'preferred_pattern': 'reversal',
                'use_pattern_filter': True,
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.9
            },
            # Multi-Correlation Strategies
            {
                'name': 'multi_correlation_btc_eth',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_btc_correlation': True,
                'use_eth_correlation': True,
                'use_multi_correlation': True,
                'btc_correlation_threshold': -0.02,
                'eth_correlation_threshold': -0.02,
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.9
            },
            # Ensemble Strategies (voting)
            {
                'name': 'ensemble_3of5',
                'min_confidence': 0.50,
                'min_ai_confidence': 0.55,
                'use_ensemble': True,
                'ensemble_votes_required': 3,
                'ensemble_total_strategies': 5,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            # Advanced BTC Correlation Strategies for Small Coins
            {
                'name': 'small_cap_btc_correlation',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'use_btc_correlation': True,
                'btc_correlation_threshold': -0.015,  # Block if BTC drops >1.5%
                'btc_correlation_required': True,  # Must align with BTC
                'use_volume_filter': True,
                'min_volume_spike': 1.3,
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.9,
                'preferred_market_cap': 'small'  # For small caps
            },
            {
                'name': 'btc_dependent_strict',
                'min_confidence': 0.58,
                'min_ai_confidence': 0.63,
                'use_btc_correlation': True,
                'btc_correlation_threshold': -0.01,  # Very strict - block if BTC drops >1%
                'btc_correlation_required': True,
                'use_multi_timeframe': True,
                'tp_multiplier': 1.3,
                'sl_multiplier': 0.85
            },
            # BTC/ETH Specific Strategies
            {
                'name': 'btc_eth_conservative',
                'min_confidence': 0.65,
                'min_ai_confidence': 0.70,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'min_volume_spike': 1.2,  # Lower for BTC/ETH
                'tp_multiplier': 0.8,  # Smaller TP for large caps
                'sl_multiplier': 0.7,
                'max_hold_time_minutes': 120
            },
            {
                'name': 'btc_eth_momentum',
                'min_confidence': 0.60,
                'min_ai_confidence': 0.65,
                'use_multi_timeframe': True,
                'use_trailing_stop': True,
                'tp_multiplier': 1.5,
                'sl_multiplier': 1.0,
                'max_hold_time_minutes': 180
            },
            # Market Cap Based Strategies
            {
                'name': 'large_cap_stable',
                'min_confidence': 0.62,
                'min_ai_confidence': 0.67,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'min_volume_spike': 1.15,
                'tp_multiplier': 0.7,
                'sl_multiplier': 0.6,
                'preferred_market_cap': 'large'
            },
            {
                'name': 'mid_cap_balanced',
                'min_confidence': 0.57,
                'min_ai_confidence': 0.62,
                'use_multi_timeframe': True,
                'use_volume_filter': True,
                'min_volume_spike': 1.4,
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.9,
                'preferred_market_cap': 'mid'
            },
            {
                'name': 'small_cap_aggressive',
                'min_confidence': 0.52,
                'min_ai_confidence': 0.57,
                'use_btc_correlation': True,
                'btc_correlation_threshold': -0.02,
                'use_volume_filter': True,
                'min_volume_spike': 1.6,
                'tp_multiplier': 1.4,
                'sl_multiplier': 1.1,
                'preferred_market_cap': 'small'
            },
            # Advanced Pattern Strategies
            {
                'name': 'pattern_momentum_breakout',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_pattern': 'breakout',
                'use_pattern_filter': True,
                'use_multi_timeframe': True,
                'tp_multiplier': 1.6,
                'sl_multiplier': 1.0
            },
            {
                'name': 'pattern_reversal_cautious',
                'min_confidence': 0.60,
                'min_ai_confidence': 0.65,
                'preferred_pattern': 'reversal',
                'use_pattern_filter': True,
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.8
            },
            # Volatility Adaptive Strategies
            {
                'name': 'volatility_adaptive_high',
                'min_confidence': 0.53,
                'min_ai_confidence': 0.58,
                'use_atr_based_tp_sl': True,
                'atr_multiplier_tp': 3.0,
                'atr_multiplier_sl': 2.0,
                'min_volatility_threshold': 0.04,
                'tp_multiplier': 1.8,
                'sl_multiplier': 1.3
            },
            {
                'name': 'volatility_adaptive_low',
                'min_confidence': 0.58,
                'min_ai_confidence': 0.63,
                'use_atr_based_tp_sl': True,
                'atr_multiplier_tp': 1.2,
                'atr_multiplier_sl': 0.8,
                'max_volatility_threshold': 0.015,
                'tp_multiplier': 0.6,
                'sl_multiplier': 0.5
            },
            # Time-Optimized Extended Strategies
            {
                'name': 'time_optimized_24h',
                'min_confidence': 0.55,
                'min_ai_confidence': 0.60,
                'preferred_hours': list(range(24)),  # All hours
                'use_time_filter': False,  # No time restriction
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.9
            },
            {
                'name': 'time_optimized_peak',
                'min_confidence': 0.52,
                'min_ai_confidence': 0.57,
                'preferred_hours': [8, 9, 10, 13, 14, 15, 16, 20, 21, 22],  # Peak trading hours
                'use_time_filter': True,
                'tp_multiplier': 1.1,
                'sl_multiplier': 0.9
            },
            # Multi-Correlation Advanced
            {
                'name': 'multi_correlation_btc_eth_strict',
                'min_confidence': 0.60,
                'min_ai_confidence': 0.65,
                'use_btc_correlation': True,
                'use_eth_correlation': True,
                'use_multi_correlation': True,
                'btc_correlation_threshold': -0.01,
                'eth_correlation_threshold': -0.01,
                'btc_correlation_required': True,
                'eth_correlation_required': False,
                'tp_multiplier': 1.2,
                'sl_multiplier': 0.9
            },
            # Regime-Specific Extended
            {
                'name': 'regime_sideways_scalping',
                'min_confidence': 0.50,
                'min_ai_confidence': 0.55,
                'preferred_regime': 'sideways',
                'use_regime_filter': True,
                'tp_multiplier': 0.5,
                'sl_multiplier': 0.4,
                'max_hold_time_minutes': 30
            },
            {
                'name': 'regime_volatile_cautious',
                'min_confidence': 0.65,
                'min_ai_confidence': 0.70,
                'preferred_regime': 'volatile',
                'use_regime_filter': True,
                'tp_multiplier': 1.0,
                'sl_multiplier': 0.8
            }
        ]
        
        # Generate variations with slight parameter tweaks
        for base in base_configs:
            variations.append(Strategy(base['name'], base.copy()))
            
            # Create more variations for better coverage (3 variations per base)
            for i in range(3):
                variant = base.copy()
                variant['name'] = f"{base['name']}_v{i+1}"
                # Slight parameter adjustments
                variant['min_confidence'] = base['min_confidence'] + (i * 0.015 - 0.015)
                variant['min_ai_confidence'] = base['min_ai_confidence'] + (i * 0.015 - 0.015)
                # Also vary TP/SL multipliers slightly
                if 'tp_multiplier' in base:
                    variant['tp_multiplier'] = base['tp_multiplier'] * (0.95 + i * 0.05)
                if 'sl_multiplier' in base:
                    variant['sl_multiplier'] = base['sl_multiplier'] * (0.95 + i * 0.05)
                variations.append(Strategy(variant['name'], variant))
        
        print(f"✅ Generated {len(variations)} strategy variations (including BTC/ETH support)")
        return variations
    
    def test_strategy(self, strategy: Strategy, historical_data: Dict) -> Dict:
        """Test a strategy on historical data"""
        # Temporarily apply strategy parameters
        old_min_conf = config.MIN_CONFIDENCE_SCORE
        old_min_ai_conf = getattr(config, 'MIN_AI_CONFIDENCE', 0.6)
        
        try:
            # Apply strategy parameters
            config.MIN_CONFIDENCE_SCORE = strategy.params.get('min_confidence', 0.55)
            setattr(config, 'MIN_AI_CONFIDENCE', strategy.params.get('min_ai_confidence', 0.6))
            
            # Run backtest with strategy-specific logic
            results = self._backtest_strategy(strategy, historical_data)
            
            return results
        finally:
            # Restore original config
            config.MIN_CONFIDENCE_SCORE = old_min_conf
            setattr(config, 'MIN_AI_CONFIDENCE', old_min_ai_conf)
    
    def _backtest_strategy(self, strategy: Strategy, historical_data: Dict) -> Dict:
        """Backtest a specific strategy"""
        total_signals = 0
        wins = 0
        losses = 0
        profits = []
        
        for symbol, df in historical_data.items():
            if len(df) < 300:
                continue  # Need enough data
            # BTC/ETH are now included in backtesting (not skipped)
            
            # Get timestamp for time-based filters
            timestamp = df.index[-1] if hasattr(df.index[-1], 'hour') else datetime.now()
            if not hasattr(timestamp, 'hour'):
                timestamp = datetime.now()
            
            # Apply strategy filters (with all advanced features)
            if not self._apply_strategy_filters(strategy, symbol, df, historical_data, 
                                               time_idx=None, timestamp=timestamp):
                continue
            
            # Generate signals with strategy logic
            signals = self._generate_strategy_signals(strategy, symbol, df, historical_data, timestamp)
            
            for signal in signals:
                total_signals += 1
                outcome = self._simulate_trade(strategy, signal, df)
                
                # Learn from errors
                if outcome['result'] == 'loss':
                    self._learn_from_errors(signal, outcome)
                
                # Track time-based performance
                if timestamp:
                    hour = timestamp.hour if hasattr(timestamp, 'hour') else datetime.now().hour
                    day = timestamp.weekday() if hasattr(timestamp, 'weekday') else datetime.now().weekday()
                    
                    hour_key = str(hour)
                    day_key = str(day)
                    
                    if hour_key not in self.time_performance:
                        self.time_performance[hour_key] = {'wins': 0, 'total': 0}
                    if day_key not in self.day_performance:
                        self.day_performance[day_key] = {'wins': 0, 'total': 0}
                    
                    self.time_performance[hour_key]['total'] += 1
                    self.day_performance[day_key]['total'] += 1
                    
                    if outcome['result'] == 'win':
                        self.time_performance[hour_key]['wins'] += 1
                        self.day_performance[day_key]['wins'] += 1
                
                if outcome['result'] == 'win':
                    wins += 1
                    profits.append(outcome['profit'])
                    # Learn coin-specific TP/SL
                    self._learn_coin_specific_params(symbol, signal, outcome)
                elif outcome['result'] == 'loss':
                    losses += 1
                    profits.append(outcome['profit'])
        
        if total_signals == 0:
            return {
                'win_rate': 0.0,
                'total_signals': 0,
                'profit': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        win_rate = wins / total_signals if total_signals > 0 else 0.0
        total_profit = sum(profits) if profits else 0.0
        
        # Calculate Sharpe Ratio
        if len(profits) > 1:
            sharpe = np.mean(profits) / (np.std(profits) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate max drawdown
        if profits:
            cumulative = np.cumsum(profits)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / (running_max + 1e-10)
            max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
        else:
            max_drawdown = 0.0
        
        return {
            'win_rate': win_rate,
            'total_signals': total_signals,
            'profit': total_profit,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown
        }
    
    def _get_btc_price_data(self, historical_data: Dict) -> Optional[pd.DataFrame]:
        """Get BTC price data from historical data or fetch if not available"""
        # Try to get from historical data first
        if 'BTC/USDT' in historical_data:
            return historical_data['BTC/USDT']
        return None
    
    def _get_eth_price_data(self, historical_data: Dict) -> Optional[pd.DataFrame]:
        """Get ETH price data for multi-correlation"""
        if 'ETH/USDT' in historical_data:
            return historical_data['ETH/USDT']
        return None
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime: trending, ranging, volatile, sideways"""
        if len(df) < 50:
            return 'unknown'
        
        # Calculate trend strength
        price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] if len(df) >= 20 else 0
        price_change_50 = (df['close'].iloc[-1] - df['close'].iloc[-50]) / df['close'].iloc[-50] if len(df) >= 50 else 0
        
        # Calculate volatility (ATR)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        volatility = atr / df['close'].iloc[-1]
        
        # Calculate range (high - low) / price
        recent_high = df['high'].iloc[-20:].max()
        recent_low = df['low'].iloc[-20:].min()
        range_pct = (recent_high - recent_low) / df['close'].iloc[-1]
        
        # Determine regime
        if abs(price_change_20) > 0.05 and abs(price_change_50) > 0.10:
            if abs(price_change_20 - price_change_50) < 0.03:  # Consistent direction
                return 'trending'
        
        if volatility > 0.04:  # High volatility
            return 'volatile'
        
        if range_pct < 0.03:  # Small range
            return 'sideways'
        
        if range_pct < 0.08 and abs(price_change_20) < 0.02:  # Small moves, small range
            return 'ranging'
        
        return 'trending'  # Default
    
    def _detect_price_pattern(self, df: pd.DataFrame) -> str:
        """Detect price pattern: breakout, reversal, consolidation, momentum"""
        if len(df) < 30:
            return 'unknown'
        
        # Recent price action
        price_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
        price_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
        price_20m = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] if len(df) >= 20 else 0
        
        # Volume
        volume_recent = df['volume'].iloc[-10:].mean()
        volume_previous = df['volume'].iloc[-20:-10].mean() if len(df) >= 20 else volume_recent
        volume_ratio = volume_recent / (volume_previous + 1e-10)
        
        # Breakout: strong move with high volume
        if abs(price_10m) > 0.03 and volume_ratio > 1.5:
            if price_10m > 0:
                return 'breakout_up'
            else:
                return 'breakout_down'
        
        # Reversal: change in direction
        if price_5m > 0.02 and price_20m < -0.02:
            return 'reversal_up'
        elif price_5m < -0.02 and price_20m > 0.02:
            return 'reversal_down'
        
        # Momentum: consistent direction
        if price_5m > 0.01 and price_10m > 0.01 and price_20m > 0.01:
            return 'momentum_up'
        elif price_5m < -0.01 and price_10m < -0.01 and price_20m < -0.01:
            return 'momentum_down'
        
        # Consolidation: small moves
        if abs(price_20m) < 0.01:
            return 'consolidation'
        
        return 'unknown'
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calculate volatility (ATR-based)"""
        if len(df) < 14:
            return 0.0
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        if df['close'].iloc[-1] > 0:
            return atr / df['close'].iloc[-1]
        return 0.0
    
    def _check_time_filter(self, strategy: Strategy, timestamp: datetime) -> bool:
        """Check if current time matches strategy's preferred hours"""
        params = strategy.params
        
        if not params.get('use_time_filter', False):
            return True
        
        preferred_hours = params.get('preferred_hours', [])
        if not preferred_hours:
            return True
        
        current_hour = timestamp.hour
        return current_hour in preferred_hours
    
    def _check_eth_correlation(self, strategy: Strategy, symbol: str, historical_data: Dict) -> Tuple[bool, str]:
        """Check ETH correlation (similar to BTC)"""
        params = strategy.params
        
        if not params.get('use_eth_correlation', False):
            return True, ""
        
        try:
            eth_df = self._get_eth_price_data(historical_data)
            if eth_df is None or len(eth_df) < 10:
                return True, "ETH data unavailable"
            
            eth_current = eth_df['close'].iloc[-1]
            eth_10m_ago = eth_df['close'].iloc[-10] if len(eth_df) >= 10 else eth_current
            eth_change = (eth_current - eth_10m_ago) / eth_10m_ago
            
            threshold = params.get('eth_correlation_threshold', -0.02)
            
            if eth_change < threshold:
                return False, f"ETH dropping {eth_change:.2%}"
            elif eth_change > 0.01:
                return True, f"✅ ETH rising {eth_change:.2%}"
            else:
                return True, f"ETH stable ({eth_change:.2%})"
        except:
            return True, "ETH check error"
    
    def _get_coin_specific_params(self, symbol: str) -> Optional[Dict]:
        """Get coin-specific TP/SL parameters if learned"""
        return self.coin_specific_params.get(symbol)
    
    def _learn_from_errors(self, signal: Dict, outcome: Dict):
        """Learn from failed signals - store error patterns"""
        if outcome.get('result') == 'loss':
            error_pattern = {
                'symbol': signal.get('symbol'),
                'signal_type': signal.get('signal_type'),
                'entry_price': signal.get('entry'),
                'price_change_10m': None,  # Would need to calculate
                'volume_change': None,
                'timestamp': signal.get('timestamp'),
                'outcome': 'loss',
                'loss_pct': outcome.get('profit', 0)
            }
            self.error_patterns.append(error_pattern)
            
            # Keep only recent errors (last 100)
            if len(self.error_patterns) > 100:
                self.error_patterns = self.error_patterns[-100:]
    
    def _check_error_patterns(self, symbol: str, analysis: Dict) -> bool:
        """Check if current signal matches known error patterns"""
        if not self.error_patterns:
            return True  # No errors learned yet
        
        # Check recent errors for this symbol
        recent_errors = [e for e in self.error_patterns[-20:] 
                         if e.get('symbol') == symbol and e.get('outcome') == 'loss']
        
        if len(recent_errors) >= 3:  # 3+ recent losses for this symbol
            # Check if current signal is similar to error patterns
            # Simple check: if too many recent losses, be more cautious
            return False  # Block signal if pattern matches errors
        
        return True
    
    def _check_btc_correlation(self, strategy: Strategy, symbol: str, historical_data: Dict, 
                               current_time_idx: int = None) -> Tuple[bool, str]:
        """Check BTC correlation and return (should_allow_signal, reason)"""
        params = strategy.params
        
        if not params.get('use_btc_correlation', False):
            return True, ""  # No BTC filter
        
        try:
            btc_df = self._get_btc_price_data(historical_data)
            if btc_df is None or len(btc_df) < 10:
                return True, "BTC data unavailable"  # Allow if can't check
            
            # For backtesting, use current time index if provided, otherwise use latest
            if current_time_idx is not None and current_time_idx < len(btc_df):
                btc_current = btc_df['close'].iloc[current_time_idx]
                btc_10m_ago = btc_df['close'].iloc[max(0, current_time_idx - 10)]
            else:
                btc_current = btc_df['close'].iloc[-1]
                btc_10m_ago = btc_df['close'].iloc[-10] if len(btc_df) >= 10 else btc_current
            
            btc_change = (btc_current - btc_10m_ago) / btc_10m_ago
            
            threshold = params.get('btc_correlation_threshold', -0.02)  # Default -2%
            required = params.get('btc_correlation_required', False)
            
            # For PUMP signals: BTC should not be dropping too much
            # For DUMP signals: BTC drop can be confirmation
            
            if btc_change < threshold:  # BTC dropping significantly
                if required:
                    return False, f"BTC dropping {btc_change:.2%} (threshold: {threshold:.2%})"
                else:
                    return True, f"⚠️ Warning: BTC dropping {btc_change:.2%}"
            elif btc_change > 0.01:  # BTC rising
                return True, f"✅ BTC rising {btc_change:.2%} - good for PUMP"
            else:
                return True, f"BTC stable ({btc_change:.2%})"
                
        except Exception as e:
            return True, f"BTC check error: {str(e)[:30]}"
    
    def _apply_strategy_filters(self, strategy: Strategy, symbol: str, df: pd.DataFrame, 
                               historical_data: Dict = None, time_idx: int = None,
                               timestamp: datetime = None) -> bool:
        """Apply strategy-specific filters with all advanced learning features"""
        params = strategy.params
        
        # Error pattern filter (learn from mistakes)
        if not self._check_error_patterns(symbol, {}):
            return False
        
        # Volume filter
        if params.get('use_volume_filter', True):
            volume_change = df['volume'].iloc[-10:].mean() / (df['volume'].iloc[-20:-10].mean() + 1e-10)
            min_volume_spike = params.get('min_volume_spike', 1.3)
            if volume_change < min_volume_spike:
                return False
        
        # Volatility filter
        if params.get('min_volatility_threshold') is not None:
            volatility = self._calculate_volatility(df)
            if volatility < params.get('min_volatility_threshold'):
                return False
        if params.get('max_volatility_threshold') is not None:
            volatility = self._calculate_volatility(df)
            if volatility > params.get('max_volatility_threshold'):
                return False
        
        # Market regime filter
        if params.get('use_regime_filter', False):
            current_regime = self._detect_market_regime(df)
            preferred_regime = params.get('preferred_regime')
            if preferred_regime and current_regime != preferred_regime:
                return False
        
        # Pattern filter
        if params.get('use_pattern_filter', False):
            current_pattern = self._detect_price_pattern(df)
            preferred_pattern = params.get('preferred_pattern')
            if preferred_pattern:
                if preferred_pattern == 'breakout' and 'breakout' not in current_pattern:
                    return False
                elif preferred_pattern == 'reversal' and 'reversal' not in current_pattern:
                    return False
        
        # Multi-timeframe filter
        if params.get('use_multi_timeframe', False):
            price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            price_change_15m = (df['close'].iloc[-1] - df['close'].iloc[-15]) / df['close'].iloc[-15] if len(df) >= 15 else 0
            if abs(price_change_5m - price_change_15m) > 0.05:
                return False
        
        # Time-based filter
        if timestamp and params.get('use_time_filter', False):
            if not self._check_time_filter(strategy, timestamp):
                return False
        
        # BTC Correlation filter
        if params.get('use_btc_correlation', False) and historical_data:
            allow, reason = self._check_btc_correlation(strategy, symbol, historical_data, time_idx)
            if not allow:
                return False
        
        # ETH Correlation filter
        if params.get('use_eth_correlation', False) and historical_data:
            allow, reason = self._check_eth_correlation(strategy, symbol, historical_data)
            if not allow:
                return False
        
        return True
    
    def _learn_coin_specific_params(self, symbol: str, signal: Dict, outcome: Dict):
        """Learn optimal TP/SL for each coin"""
        if symbol not in self.coin_specific_params:
            self.coin_specific_params[symbol] = {
                'wins': 0,
                'total': 0,
                'avg_profit': 0.0,
                'optimal_tp_multiplier': 1.0,
                'optimal_sl_multiplier': 0.9
            }
        
        coin_data = self.coin_specific_params[symbol]
        coin_data['total'] += 1
        
        if outcome.get('result') == 'win' and outcome.get('profit', 0) > 0:
            coin_data['wins'] += 1
            # Update average profit
            total_profit = coin_data['avg_profit'] * (coin_data['total'] - 1) + outcome.get('profit', 0)
            coin_data['avg_profit'] = total_profit / coin_data['total']
            
            # Adjust TP/SL based on performance
            if coin_data['wins'] >= 5:  # After 5 wins, optimize
                win_rate = coin_data['wins'] / coin_data['total']
                if win_rate > 0.7:  # High win rate - can increase TP
                    coin_data['optimal_tp_multiplier'] = min(1.5, coin_data['optimal_tp_multiplier'] * 1.05)
                elif win_rate < 0.5:  # Low win rate - reduce TP
                    coin_data['optimal_tp_multiplier'] = max(0.8, coin_data['optimal_tp_multiplier'] * 0.95)
    
    def _generate_strategy_signals(self, strategy: Strategy, symbol: str, df: pd.DataFrame, 
                                  historical_data: Dict = None, timestamp: datetime = None) -> List[Dict]:
        """Generate signals using strategy logic with all advanced features"""
        signals = []
        
        # Use AI analyzer with strategy parameters
        analysis = self.ai_analyzer.analyze_coin(symbol, df)
        
        if analysis is None:
            return signals
        
        # Check if signal meets strategy criteria
        signal_prob = analysis.get('signal_probability', 0)
        confidence = analysis.get('confidence', 0)
        
        min_conf = strategy.params.get('min_confidence', 0.55)
        min_ai_conf = strategy.params.get('min_ai_confidence', 0.6)
        
        if signal_prob >= min_conf and confidence >= min_ai_conf:
            # Additional BTC correlation check
            btc_check_passed = True
            btc_reason = ""
            if strategy.params.get('use_btc_correlation', False) and historical_data:
                btc_check_passed, btc_reason = self._check_btc_correlation(strategy, symbol, historical_data)
                if not btc_check_passed and strategy.params.get('btc_correlation_required', False):
                    return signals
            
            # Apply coin-specific TP/SL if learned
            coin_params = self._get_coin_specific_params(symbol)
            tp_mult = strategy.params.get('tp_multiplier', 1.0)
            sl_mult = strategy.params.get('sl_multiplier', 0.9)
            
            if coin_params and coin_params.get('total', 0) >= 5:
                tp_mult = coin_params.get('optimal_tp_multiplier', tp_mult)
                sl_mult = coin_params.get('optimal_sl_multiplier', sl_mult)
            
            # Adjust TP/SL based on coin-specific learning
            entry = analysis.get('entry', df['close'].iloc[-1])
            if analysis.get('signal_type') == 'PUMP':
                exit1 = entry * (1 + (0.10 * tp_mult))
                sl = entry * (1 - (0.07 * sl_mult))
            else:
                exit1 = entry * (1 - (0.10 * tp_mult))
                sl = entry * (1 + (0.07 * sl_mult))
            
            signal = {
                'symbol': symbol,
                'signal_type': analysis.get('signal_type'),
                'entry': entry,
                'exit1': exit1,
                'exit2': analysis.get('exit2', exit1 * 1.2 if analysis.get('signal_type') == 'PUMP' else exit1 * 0.8),
                'exit3': analysis.get('exit3', exit1 * 1.5 if analysis.get('signal_type') == 'PUMP' else exit1 * 0.5),
                'stop_loss': sl,
                'timestamp': timestamp or analysis.get('timestamp', datetime.now()),
                'strategy_params': strategy.params,
                'btc_correlation': btc_reason if strategy.params.get('use_btc_correlation', False) else None,
                'coin_specific_params': coin_params is not None
            }
            signals.append(signal)
        
        return signals
    
    def _simulate_trade(self, strategy: Strategy, signal: Dict, df: pd.DataFrame) -> Dict:
        """Simulate a trade with strategy-specific TP/SL"""
        entry_price = signal['entry']
        entry_idx = len(df) - 1
        
        # Adjust TP/SL based on strategy
        params = strategy.params
        
        if params.get('use_atr_based_tp_sl', False):
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]
            
            if signal['signal_type'] == 'PUMP':
                tp = entry_price + (atr * params.get('tp_multiplier', 1.0))
                sl = entry_price - (atr * params.get('sl_multiplier', 0.9))
            else:
                tp = entry_price - (atr * params.get('tp_multiplier', 1.0))
                sl = entry_price + (atr * params.get('sl_multiplier', 0.9))
        else:
            # Use fixed TP/SL with multiplier
            if signal['signal_type'] == 'PUMP':
                tp = entry_price * (1 + (0.10 * params.get('tp_multiplier', 1.0)))
                sl = entry_price * (1 - (0.07 * params.get('sl_multiplier', 0.9)))
            else:
                tp = entry_price * (1 - (0.10 * params.get('tp_multiplier', 1.0)))
                sl = entry_price * (1 + (0.07 * params.get('sl_multiplier', 0.9)))
        
        # Simulate forward price movement
        max_lookahead = params.get('max_hold_time_minutes', 240)  # Default 4 hours
        lookahead = min(max_lookahead, len(df) - entry_idx - 1)
        
        if lookahead <= 0:
            return {'result': 'timeout', 'profit': 0.0}
        
        future_prices = df['close'].iloc[entry_idx+1:entry_idx+1+lookahead]
        
        # Check if TP or SL hit
        if signal['signal_type'] == 'PUMP':
            tp_hit = (future_prices >= tp).any()
            sl_hit = (future_prices <= sl).any()
        else:
            tp_hit = (future_prices <= tp).any()
            sl_hit = (future_prices >= sl).any()
        
        if tp_hit and not sl_hit:
            # Win
            exit_price = tp
            profit_pct = (exit_price - entry_price) / entry_price if signal['signal_type'] == 'PUMP' else (entry_price - exit_price) / entry_price
            return {'result': 'win', 'profit': profit_pct * 100}
        elif sl_hit:
            # Loss
            exit_price = sl
            profit_pct = (exit_price - entry_price) / entry_price if signal['signal_type'] == 'PUMP' else (entry_price - exit_price) / entry_price
            return {'result': 'loss', 'profit': profit_pct * 100}
        else:
            # Timeout
            exit_price = future_prices.iloc[-1]
            profit_pct = (exit_price - entry_price) / entry_price if signal['signal_type'] == 'PUMP' else (entry_price - exit_price) / entry_price
            return {'result': 'timeout', 'profit': profit_pct * 100}
    
    def optimize_strategies(self, historical_data: Dict) -> Dict:
        """Test all strategies and find the best ones"""
        print("\n" + "="*60)
        print("🧠 SELF-LEARNING STRATEGY OPTIMIZER")
        print("="*60)
        
        # Generate strategy variations
        strategies_to_test = self.generate_strategy_variations()
        
        # Add existing strategies that haven't been tested recently
        for existing in self.strategies:
            if existing.performance.get('tests_count', 0) < 5:  # Re-test if not tested enough
                strategies_to_test.append(existing)
        
        print(f"📊 Testing {len(strategies_to_test)} strategy variations...")
        
        results = []
        for i, strategy in enumerate(strategies_to_test, 1):
            print(f"   Testing {i}/{len(strategies_to_test)}: {strategy.name}...")
            
            try:
                test_results = self.test_strategy(strategy, historical_data)
                
                # Update strategy performance
                strategy.performance.update(test_results)
                strategy.performance['last_tested'] = datetime.now().isoformat()
                strategy.performance['tests_count'] = strategy.performance.get('tests_count', 0) + 1
                
                # Calculate composite score
                score = self._calculate_strategy_score(test_results)
                strategy.performance['score'] = score
                
                results.append((strategy, score))
                
                print(f"      ✅ Win Rate: {test_results['win_rate']:.1%}, "
                      f"Signals: {test_results['total_signals']}, "
                      f"Profit: {test_results['profit']:.2f}, "
                      f"Score: {score:.2f}")
                
            except Exception as e:
                print(f"      ❌ Error testing {strategy.name}: {e}")
                continue
        
        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Update strategies list
        self.strategies = [s for s, _ in results]
        
        # Select best strategies (top 30% or minimum 3)
        num_best = max(3, int(len(results) * 0.3))
        self.best_strategies = [s for s, _ in results[:num_best]]
        
        # Save strategies
        self.save_strategies()
        
        # Apply best strategy to config
        if self.best_strategies:
            best = self.best_strategies[0]
            print(f"\n🏆 Best Strategy: {best.name}")
            print(f"   Applying to live trading...")
            self._apply_best_strategy(best)
        
        return {
            'total_tested': len(strategies_to_test),
            'best_strategies': [s.name for s in self.best_strategies],
            'best_score': results[0][1] if results else 0.0
        }
    
    def _calculate_strategy_score(self, results: Dict) -> float:
        """Calculate composite score for a strategy"""
        win_rate = results.get('win_rate', 0.0)
        total_signals = results.get('total_signals', 0)
        profit = results.get('profit', 0.0)
        sharpe = results.get('sharpe_ratio', 0.0)
        drawdown = results.get('max_drawdown', 0.0)
        
        # Weighted scoring
        score = (
            win_rate * 0.35 +           # Win rate is important
            min(profit / 100, 1.0) * 0.25 +  # Profit (normalized)
            min(sharpe / 2.0, 1.0) * 0.20 +  # Sharpe ratio
            min(total_signals / 50, 1.0) * 0.10 +  # Signal count (not too few)
            (1.0 - min(drawdown, 1.0)) * 0.10  # Low drawdown is good
        )
        
        return score * 100  # Scale to 0-100
    
    def _apply_best_strategy(self, strategy: Strategy):
        """Apply best strategy parameters to config"""
        params = strategy.params
        
        # Update config with best strategy parameters
        config.MIN_CONFIDENCE_SCORE = params.get('min_confidence', config.MIN_CONFIDENCE_SCORE)
        setattr(config, 'MIN_AI_CONFIDENCE', params.get('min_ai_confidence', getattr(config, 'MIN_AI_CONFIDENCE', 0.6)))
        
        # Apply BTC correlation settings if strategy uses it
        if params.get('use_btc_correlation', False):
            setattr(config, 'USE_BTC_CORRELATION_FILTER', True)
            setattr(config, 'BTC_CORRELATION_THRESHOLD', params.get('btc_correlation_threshold', -0.02))
            setattr(config, 'BTC_CORRELATION_REQUIRED', params.get('btc_correlation_required', False))
            print(f"   ✅ Applied BTC Correlation Filter: threshold={params.get('btc_correlation_threshold', -0.02):.2%}, "
                  f"required={params.get('btc_correlation_required', False)}")
        else:
            setattr(config, 'USE_BTC_CORRELATION_FILTER', False)
        
        print(f"   ✅ Applied: MIN_CONFIDENCE_SCORE={params.get('min_confidence', 0.55):.2f}, "
              f"MIN_AI_CONFIDENCE={params.get('min_ai_confidence', 0.6):.2f}")
    
    def get_best_strategy_for_market(self, market_conditions: Dict) -> Optional[Strategy]:
        """Select best strategy based on current market conditions"""
        if not self.best_strategies:
            return None
        
        # Simple selection: use top strategy for now
        # Could be enhanced to match market conditions
        return self.best_strategies[0]

