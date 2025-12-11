"""
Coin-Specific Learning System
Learns unique strategies, timing, and patterns for each individual coin
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os


class CoinSpecificLearner:
    """
    Learns coin-specific characteristics:
    - Best trading times for each coin
    - Optimal strategy parameters per coin
    - Price movement patterns unique to each coin
    - Volume patterns
    - Volatility characteristics
    """
    
    def __init__(self):
        self.coin_profiles = {}  # symbol -> coin profile
        self.profile_file = 'models/coin_profiles.json'
        self._load_profiles()
        
        # Learning data per coin
        self.coin_performance = defaultdict(lambda: {
            'wins': deque(maxlen=50),
            'losses': deque(maxlen=50),
            'timeouts': deque(maxlen=50),
            'best_times': defaultdict(int),  # hour -> win count
            'worst_times': defaultdict(int),  # hour -> loss count
            'best_days': defaultdict(int),    # day_of_week -> win count
            'optimal_params': {},             # Learned optimal parameters
            'movement_patterns': []           # Successful movement patterns
        })
    
    def _load_profiles(self):
        """Load saved coin profiles"""
        if os.path.exists(self.profile_file):
            try:
                with open(self.profile_file, 'r') as f:
                    data = json.load(f)
                    self.coin_profiles = data.get('coin_profiles', {})
                print(f"Loaded profiles for {len(self.coin_profiles)} coins")
            except Exception as e:
                print(f"Error loading coin profiles: {e}")
    
    def _save_profiles(self):
        """Save coin profiles to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            data = {
                'coin_profiles': self.coin_profiles,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.profile_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving coin profiles: {e}")
    
    def learn_from_signal(self, symbol: str, signal: Dict, outcome: str, 
                         profit_pct: float = 0.0, df: Optional[pd.DataFrame] = None):
        """
        Learn from a signal outcome for a specific coin
        Enhanced for BTC/ETH and all coins
        """
        if symbol not in self.coin_profiles:
            # Initialize profile - works for BTC/ETH and all coins
            self.coin_profiles[symbol] = {
                'total_signals': 0,
                'wins': 0,
                'losses': 0,
                'timeouts': 0,
                'win_rate': 0.0,
                'best_trading_hours': [],
                'worst_trading_hours': [],
                'optimal_entry_times': [],
                'avg_profit_per_win': 0.0,
                'avg_loss_per_loss': 0.0,
                'optimal_tp_pct': None,
                'optimal_sl_pct': None,
                'price_movement_profile': {},
                'volume_profile': {},
                'volatility_profile': {},
                'btc_correlation': None,  # BTC correlation for small coins
                'market_cap_category': self._detect_market_cap_category(symbol),  # large/mid/small
                'last_updated': datetime.now().isoformat()
            }
        
        profile = self.coin_profiles[symbol]
        profile['total_signals'] += 1
        
        # Record outcome
        if outcome == 'win':
            profile['wins'] += 1
            self.coin_performance[symbol]['wins'].append({
                'profit': profit_pct,
                'timestamp': signal.get('timestamp', datetime.now()),
                'confidence': signal.get('confidence', 0),
                'probability': signal.get('signal_probability', 0)
            })
        elif outcome == 'loss':
            profile['losses'] += 1
            self.coin_performance[symbol]['losses'].append({
                'loss': abs(profit_pct),
                'timestamp': signal.get('timestamp', datetime.now()),
                'confidence': signal.get('confidence', 0),
                'probability': signal.get('signal_probability', 0)
            })
        elif outcome == 'timeout':
            profile['timeouts'] += 1
            self.coin_performance[symbol]['timeouts'].append({
                'timestamp': signal.get('timestamp', datetime.now())
            })
        
        # Update win rate
        total = profile['wins'] + profile['losses'] + profile['timeouts']
        profile['win_rate'] = profile['wins'] / total if total > 0 else 0.0
        
        # Learn optimal trading times
        timestamp = signal.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if outcome == 'win':
            self.coin_performance[symbol]['best_times'][hour] += 1
            self.coin_performance[symbol]['best_days'][day_of_week] += 1
        elif outcome == 'loss':
            self.coin_performance[symbol]['worst_times'][hour] += 1
        
        # Learn optimal TP/SL based on outcomes
        if outcome == 'win' and profit_pct > 0:
            # If we hit TP, learn what TP worked
            if signal.get('hit_target'):
                tp_pct = abs(profit_pct)
                if profile['optimal_tp_pct'] is None:
                    profile['optimal_tp_pct'] = tp_pct
                else:
                    # Moving average
                    profile['optimal_tp_pct'] = profile['optimal_tp_pct'] * 0.7 + tp_pct * 0.3
        
        # Learn price movement patterns
        if df is not None and len(df) >= 20:
            self._learn_movement_pattern(symbol, signal, outcome, df)
        
        # Learn volume patterns
        volume_change = signal.get('volume_change', 1.0)
        if outcome == 'win':
            if 'volume_profile' not in profile:
                profile['volume_profile'] = {'winning_volumes': [], 'losing_volumes': []}
            profile['volume_profile']['winning_volumes'].append(volume_change)
            if len(profile['volume_profile']['winning_volumes']) > 100:
                profile['volume_profile']['winning_volumes'] = profile['volume_profile']['winning_volumes'][-100:]
        elif outcome == 'loss':
            if 'volume_profile' not in profile:
                profile['volume_profile'] = {'winning_volumes': [], 'losing_volumes': []}
            profile['volume_profile']['losing_volumes'].append(volume_change)
            if len(profile['volume_profile']['losing_volumes']) > 100:
                profile['volume_profile']['losing_volumes'] = profile['volume_profile']['losing_volumes'][-100:]
        
        # Update optimal parameters
        self._update_optimal_params(symbol, signal, outcome)
        
        # Save periodically
        if profile['total_signals'] % 10 == 0:
            self._save_profiles()
    
    def _learn_movement_pattern(self, symbol: str, signal: Dict, outcome: str, df: pd.DataFrame):
        """
        Learn price movement patterns specific to this coin
        """
        if symbol not in self.coin_profiles:
            return
        
        profile = self.coin_profiles[symbol]
        
        # Analyze price movement before signal
        if len(df) >= 20:
            price_changes = df['close'].pct_change().tail(20).dropna()
            volatility = price_changes.std()
            avg_move = price_changes.abs().mean()
            
            if 'price_movement_profile' not in profile:
                profile['price_movement_profile'] = {
                    'winning_patterns': [],
                    'losing_patterns': []
                }
            
            pattern = {
                'volatility': float(volatility),
                'avg_move': float(avg_move),
                'price_change_10m': signal.get('price_change_10m', 0),
                'outcome': outcome
            }
            
            if outcome == 'win':
                profile['price_movement_profile']['winning_patterns'].append(pattern)
                if len(profile['price_movement_profile']['winning_patterns']) > 50:
                    profile['price_movement_profile']['winning_patterns'] = \
                        profile['price_movement_profile']['winning_patterns'][-50:]
            elif outcome == 'loss':
                profile['price_movement_profile']['losing_patterns'].append(pattern)
                if len(profile['price_movement_profile']['losing_patterns']) > 50:
                    profile['price_movement_profile']['losing_patterns'] = \
                        profile['price_movement_profile']['losing_patterns'][-50:]
    
    def _update_optimal_params(self, symbol: str, signal: Dict, outcome: str):
        """
        Update optimal parameters for this coin based on outcomes
        """
        if symbol not in self.coin_profiles:
            return
        
        profile = self.coin_profiles[symbol]
        confidence = signal.get('confidence', 0)
        probability = signal.get('signal_probability', 0)
        
        if 'optimal_params' not in profile:
            profile['optimal_params'] = {
                'min_confidence': [],
                'min_probability': [],
                'min_price_move': [],
                'min_volume_spike': []
            }
        
        if outcome == 'win':
            # Learn what worked
            profile['optimal_params']['min_confidence'].append(confidence)
            profile['optimal_params']['min_probability'].append(probability)
            profile['optimal_params']['min_price_move'].append(abs(signal.get('price_change_10m', 0)))
            profile['optimal_params']['min_volume_spike'].append(signal.get('volume_change', 1.0))
            
            # Keep only recent (last 30)
            for key in profile['optimal_params']:
                if len(profile['optimal_params'][key]) > 30:
                    profile['optimal_params'][key] = profile['optimal_params'][key][-30:]
    
    def get_coin_strategy(self, symbol: str) -> Dict:
        """
        Get coin-specific strategy parameters
        """
        if symbol not in self.coin_profiles:
            return {}  # No profile yet, use defaults
        
        profile = self.coin_profiles[symbol]
        
        strategy = {
            'has_profile': True,
            'win_rate': profile.get('win_rate', 0.0),
            'total_signals': profile.get('total_signals', 0)
        }
        
        # Optimal trading times
        best_times = self.coin_performance[symbol]['best_times']
        worst_times = self.coin_performance[symbol]['worst_times']
        
        if best_times:
            # Find best hours (top 3)
            sorted_best = sorted(best_times.items(), key=lambda x: x[1], reverse=True)
            strategy['best_hours'] = [h for h, count in sorted_best[:3]]
            strategy['worst_hours'] = [h for h, count in sorted(worst_times.items(), 
                                                               key=lambda x: x[1], reverse=True)[:3]]
        
        # Optimal parameters
        optimal = profile.get('optimal_params', {})
        if optimal:
            if optimal.get('min_confidence'):
                strategy['optimal_min_confidence'] = np.percentile(optimal['min_confidence'], 25)  # 25th percentile
            if optimal.get('min_probability'):
                strategy['optimal_min_probability'] = np.percentile(optimal['min_probability'], 25)
            if optimal.get('min_price_move'):
                strategy['optimal_min_price_move'] = np.percentile(optimal['min_price_move'], 25)
            if optimal.get('min_volume_spike'):
                strategy['optimal_min_volume_spike'] = np.percentile(optimal['min_volume_spike'], 25)
        
        # Optimal TP/SL
        if profile.get('optimal_tp_pct'):
            strategy['optimal_tp_pct'] = profile['optimal_tp_pct']
        if profile.get('optimal_sl_pct'):
            strategy['optimal_sl_pct'] = profile['optimal_sl_pct']
        
        # Volume profile
        vol_profile = profile.get('volume_profile', {})
        if vol_profile.get('winning_volumes'):
            strategy['optimal_volume_spike'] = np.percentile(vol_profile['winning_volumes'], 25)
        
        # BTC correlation for small coins
        if profile.get('btc_correlation') is not None:
            strategy['btc_correlation'] = profile['btc_correlation']
            strategy['use_btc_correlation'] = profile.get('market_cap_category') == 'small'
        
        return strategy
    
    def _detect_market_cap_category(self, symbol: str) -> str:
        """Detect if coin is large cap (BTC/ETH), mid cap, or small cap"""
        symbol_upper = symbol.upper()
        # Large caps
        if symbol_upper in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 
                            'ADA/USDT', 'DOGE/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT']:
            return 'large'
        # Could add more detection logic here based on market cap if available
        # For now, default to 'small' for unknown coins
        return 'small'
    
    def learn_btc_correlation(self, symbol: str, btc_price_change: float, coin_price_change: float, 
                             outcome: str):
        """Learn BTC correlation patterns for small coins"""
        if symbol not in self.coin_profiles:
            return
        
        profile = self.coin_profiles[symbol]
        
        # Only learn for small caps
        if profile.get('market_cap_category') != 'small':
            return
        
        if 'btc_correlation' not in profile or profile['btc_correlation'] is None:
            profile['btc_correlation'] = {
                'correlation_coefficient': 0.0,
                'samples': [],
                'positive_correlation_count': 0,
                'negative_correlation_count': 0,
                'outcomes': []
            }
        
        corr_data = profile['btc_correlation']
        corr_data['samples'].append({
            'btc_change': btc_price_change,
            'coin_change': coin_price_change,
            'outcome': outcome,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep last 50 samples
        if len(corr_data['samples']) > 50:
            corr_data['samples'] = corr_data['samples'][-50:]
        
        # Calculate correlation
        if len(corr_data['samples']) >= 10:
            btc_changes = [s['btc_change'] for s in corr_data['samples']]
            coin_changes = [s['coin_change'] for s in corr_data['samples']]
            if len(btc_changes) > 1 and len(coin_changes) > 1:
                correlation = np.corrcoef(btc_changes, coin_changes)[0, 1]
                if not np.isnan(correlation):
                    corr_data['correlation_coefficient'] = correlation
        
        # Track outcomes
        if outcome == 'win':
            if btc_price_change > 0 and coin_price_change > 0:
                corr_data['positive_correlation_count'] += 1
            elif btc_price_change < 0 and coin_price_change < 0:
                corr_data['positive_correlation_count'] += 1
            else:
                corr_data['negative_correlation_count'] += 1
        
        self._save_profiles()
    
    def should_trade_coin_now(self, symbol: str, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        Check if current time is good for trading this coin
        """
        if symbol not in self.coin_profiles:
            return True, ""  # No data yet, allow trading
        
        if current_time is None:
            current_time = datetime.now()
        
        hour = current_time.hour
        day_of_week = current_time.weekday()
        
        # Check if current hour is in worst hours
        worst_hours = self.coin_performance[symbol]['worst_times']
        if worst_hours.get(hour, 0) >= 3:  # 3+ losses at this hour
            return False, f"Hour {hour} has {worst_hours[hour]} losses for {symbol}"
        
        # Check if current hour is in best hours
        best_times = self.coin_performance[symbol]['best_times']
        if best_times.get(hour, 0) >= 2:  # 2+ wins at this hour
            return True, f"Hour {hour} is good for {symbol} ({best_times[hour]} wins)"
        
        return True, ""  # Default: allow trading
    
    def get_coin_specific_filters(self, symbol: str, analysis: Dict) -> Dict:
        """
        Get coin-specific filter adjustments based on learned patterns
        """
        strategy = self.get_coin_strategy(symbol)
        
        if not strategy.get('has_profile'):
            return {}  # No adjustments yet
        
        adjustments = {}
        
        # Adjust confidence threshold based on coin's win rate
        if strategy['win_rate'] < 0.40 and strategy['total_signals'] >= 10:
            # Coin has low win rate - require higher confidence
            if strategy.get('optimal_min_confidence'):
                adjustments['min_confidence'] = max(
                    strategy['optimal_min_confidence'],
                    analysis.get('confidence', 0) + 0.05
                )
        
        # Adjust volume spike based on coin's volume profile
        if strategy.get('optimal_volume_spike'):
            current_vol = analysis.get('volume_change', 1.0)
            if current_vol < strategy['optimal_volume_spike']:
                adjustments['volume_too_low'] = True
        
        # Adjust price move based on coin's patterns
        if strategy.get('optimal_min_price_move'):
            current_move = abs(analysis.get('price_change_10m', 0))
            if current_move < strategy['optimal_min_price_move']:
                adjustments['move_too_small'] = True
        
        return adjustments
    
    def get_coin_optimal_tp_sl(self, symbol: str, signal_type: str, 
                               default_tp: float, default_sl: float) -> Tuple[float, float]:
        """
        Get optimal TP/SL for this specific coin
        """
        if symbol not in self.coin_profiles:
            return default_tp, default_sl
        
        profile = self.coin_profiles[symbol]
        
        # Use learned optimal TP if available
        if profile.get('optimal_tp_pct'):
            optimal_tp = profile['optimal_tp_pct'] / 100.0  # Convert to decimal
            # Use learned TP but don't go too far from default
            tp = default_tp * 0.8 + optimal_tp * 0.2  # Blend
        else:
            tp = default_tp
        
        # Adjust SL based on coin's volatility
        if profile.get('price_movement_profile'):
            winning_patterns = profile['price_movement_profile'].get('winning_patterns', [])
            if winning_patterns:
                avg_volatility = np.mean([p.get('volatility', 0.05) for p in winning_patterns[-10:]])
                # More volatile coins need wider SL
                if avg_volatility > 0.03:
                    sl = default_sl * 1.1  # 10% wider
                else:
                    sl = default_sl
            else:
                sl = default_sl
        else:
            sl = default_sl
        
        return tp, sl

