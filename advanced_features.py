"""
Advanced Features for Better Profitability and Accuracy
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import asyncio


class AdvancedFeatures:
    """
    Advanced features to improve profitability:
    1. Multi-timeframe analysis
    2. Market regime detection
    3. BTC correlation analysis
    4. Dynamic position sizing
    5. Trailing stop loss
    6. Partial profit taking
    7. Order book analysis
    8. Real-time momentum detection
    """
    
    def __init__(self):
        self.btc_data_cache = None
        self.market_regime = 'neutral'  # bull, bear, neutral
        self.last_regime_update = None
    
    async def analyze_multi_timeframe(self, symbol: str, df_1m: pd.DataFrame) -> Dict:
        """
        Analyze multiple timeframes (1m, 5m, 15m, 1h) for better signals
        """
        analysis = {
            'timeframe_alignment': False,
            'trend_strength': 0.0,
            'support_resistance': {},
            'multi_tf_signal': None
        }
        
        try:
            # Check if df has timestamp index or column
            if df_1m.index.name == 'timestamp' or isinstance(df_1m.index, pd.DatetimeIndex):
                df_1m_copy = df_1m.copy()
                df_1m_copy['timestamp'] = df_1m_copy.index
            elif 'timestamp' in df_1m.columns:
                df_1m_copy = df_1m.copy()
            else:
                # Create timestamp from index if it's numeric
                df_1m_copy = df_1m.copy()
                df_1m_copy['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_1m_copy), freq='1T')
            
            # Resample to different timeframes
            df_5m = df_1m_copy.resample('5T', on='timestamp').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna() if len(df_1m_copy) >= 5 else None
            
            df_15m = df_1m_copy.resample('15T', on='timestamp').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna() if len(df_1m_copy) >= 15 else None
            
            
            # Calculate trends on different timeframes
            if df_5m is not None and len(df_5m) >= 20:
                trend_5m = (df_5m['close'].iloc[-1] - df_5m['close'].iloc[-20]) / df_5m['close'].iloc[-20]
            else:
                trend_5m = (df_1m['close'].iloc[-1] - df_1m['close'].iloc[-20]) / df_1m['close'].iloc[-20] if len(df_1m) >= 20 else 0
            
            if df_15m is not None and len(df_15m) >= 10:
                trend_15m = (df_15m['close'].iloc[-1] - df_15m['close'].iloc[-10]) / df_15m['close'].iloc[-10]
            else:
                trend_15m = 0
            
            trend_1m = (df_1m['close'].iloc[-1] - df_1m['close'].iloc[-10]) / df_1m['close'].iloc[-10] if len(df_1m) >= 10 else 0
            
            # Check alignment
            if trend_1m > 0 and trend_5m > 0 and trend_15m > 0:
                analysis['timeframe_alignment'] = True
                analysis['multi_tf_signal'] = 'PUMP'
                analysis['trend_strength'] = (abs(trend_1m) + abs(trend_5m) + abs(trend_15m)) / 3
            elif trend_1m < 0 and trend_5m < 0 and trend_15m < 0:
                analysis['timeframe_alignment'] = True
                analysis['multi_tf_signal'] = 'DUMP'
                analysis['trend_strength'] = (abs(trend_1m) + abs(trend_5m) + abs(trend_15m)) / 3
            
            # Support/Resistance levels
            if len(df_1m) >= 100:
                recent_highs = df_1m['high'].tail(100).rolling(20).max()
                recent_lows = df_1m['low'].tail(100).rolling(20).min()
                analysis['support_resistance'] = {
                    'resistance': float(recent_highs.max()),
                    'support': float(recent_lows.min()),
                    'current': float(df_1m['close'].iloc[-1])
                }
        
        except Exception as e:
            print(f"Error in multi-timeframe analysis: {e}")
        
        return analysis
    
    async def detect_market_regime(self, btc_df: Optional[pd.DataFrame] = None) -> str:
        """
        Detect current market regime: bull, bear, or neutral
        """
        try:
            if btc_df is None or len(btc_df) < 50:
                return 'neutral'
            
            # Calculate BTC trend
            price_20 = btc_df['close'].iloc[-20] if len(btc_df) >= 20 else btc_df['close'].iloc[0]
            price_current = btc_df['close'].iloc[-1]
            btc_change = (price_current - price_20) / price_20
            
            # Calculate volatility
            returns = btc_df['close'].pct_change().tail(50).dropna()
            volatility = returns.std()
            
            # Determine regime
            if btc_change > 0.05 and volatility < 0.03:  # 5%+ up, low volatility
                self.market_regime = 'bull'
            elif btc_change < -0.05 and volatility < 0.03:  # 5%+ down, low volatility
                self.market_regime = 'bear'
            else:
                self.market_regime = 'neutral'
            
            self.last_regime_update = datetime.now()
            return self.market_regime
        
        except Exception:
            return 'neutral'
    
    def analyze_btc_correlation(self, symbol_df: pd.DataFrame, btc_df: pd.DataFrame) -> Dict:
        """
        Analyze correlation with BTC to avoid false signals
        """
        if len(symbol_df) < 50 or len(btc_df) < 50:
            return {'correlation': 0.0, 'btc_dominance': False}
        
        try:
            # Calculate returns
            symbol_returns = symbol_df['close'].pct_change().tail(50).dropna()
            btc_returns = btc_df['close'].pct_change().tail(50).dropna()
            
            # Align lengths
            min_len = min(len(symbol_returns), len(btc_returns))
            symbol_returns = symbol_returns.tail(min_len)
            btc_returns = btc_returns.tail(min_len)
            
            # Calculate correlation
            correlation = symbol_returns.corr(btc_returns) if len(symbol_returns) > 1 else 0.0
            
            # Check if BTC is dominating
            btc_volatility = btc_returns.std()
            symbol_volatility = symbol_returns.std()
            
            # If correlation is high and BTC is moving, coin might just be following BTC
            btc_recent_move = abs(btc_returns.tail(10).sum())
            symbol_recent_move = abs(symbol_returns.tail(10).sum())
            
            btc_dominance = (
                correlation > 0.7 and 
                btc_recent_move > 0.02 and  # BTC moved 2%+
                symbol_recent_move < btc_recent_move * 1.5  # Coin not outperforming much
            )
            
            return {
                'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
                'btc_dominance': btc_dominance,
                'btc_volatility': float(btc_volatility),
                'symbol_volatility': float(symbol_volatility)
            }
        
        except Exception as e:
            print(f"Error in BTC correlation: {e}")
            return {'correlation': 0.0, 'btc_dominance': False}
    
    def calculate_dynamic_position_size(self, signal: Dict, account_balance: float = 100.0) -> float:
        """
        Calculate optimal position size based on:
        - Signal confidence
        - Risk/reward ratio
        - Market volatility
        - Coin-specific win rate
        """
        base_position = account_balance * 0.1  # 10% base
        
        confidence = signal.get('confidence', 0.5)
        probability = signal.get('signal_probability', 0.5)
        risk_reward = signal.get('risk_reward_ratio', 1.2)
        
        # Adjust based on confidence
        confidence_multiplier = 0.5 + (confidence * 1.0)  # 0.5x to 1.5x
        
        # Adjust based on probability
        probability_multiplier = 0.7 + (probability * 0.6)  # 0.7x to 1.3x
        
        # Adjust based on risk/reward
        rr_multiplier = min(1.5, risk_reward / 1.5)  # Better R/R = larger position
        
        # Adjust based on market regime
        regime_multiplier = 1.0
        if self.market_regime == 'bull':
            regime_multiplier = 1.2  # More aggressive in bull market
        elif self.market_regime == 'bear':
            regime_multiplier = 0.7  # More conservative in bear market
        
        position_size = base_position * confidence_multiplier * probability_multiplier * rr_multiplier * regime_multiplier
        
        # Cap at 20% of balance
        max_position = account_balance * 0.20
        position_size = min(position_size, max_position)
        
        # Minimum position
        min_position = account_balance * 0.05
        position_size = max(position_size, min_position)
        
        return position_size
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float, 
                                signal_type: str, initial_sl: float) -> float:
        """
        Calculate trailing stop loss
        """
        if signal_type == 'PUMP':
            profit_pct = (current_price - entry_price) / entry_price
            
            # Start trailing after 3% profit
            if profit_pct > 0.03:
                # Trail at 2% below highest price
                trailing_sl = current_price * 0.98
                # Don't let it go below initial SL
                return max(trailing_sl, initial_sl)
        
        elif signal_type == 'DUMP':
            profit_pct = (entry_price - current_price) / entry_price
            
            # Start trailing after 3% profit
            if profit_pct > 0.03:
                # Trail at 2% above lowest price
                trailing_sl = current_price * 1.02
                # Don't let it go above initial SL
                return min(trailing_sl, initial_sl)
        
        return initial_sl
    
    def calculate_partial_profit_levels(self, entry_price: float, signal_type: str,
                                       confidence: float) -> List[Dict]:
        """
        Calculate partial profit taking levels
        """
        levels = []
        
        if signal_type == 'PUMP':
            # Take 30% profit at first target
            levels.append({
                'price': entry_price * 1.05,  # 5% profit
                'percentage': 0.30,  # Close 30% of position
                'reason': 'First target - secure some profit'
            })
            
            # Take 40% profit at second target
            levels.append({
                'price': entry_price * 1.10,  # 10% profit
                'percentage': 0.40,  # Close 40% of position
                'reason': 'Second target - lock in more profit'
            })
            
            # Let remaining 30% run to final target
            if confidence > 0.7:
                levels.append({
                    'price': entry_price * 1.20,  # 20% profit
                    'percentage': 0.30,  # Close remaining 30%
                    'reason': 'Final target - high confidence'
                })
        
        elif signal_type == 'DUMP':
            # Similar for shorts
            levels.append({
                'price': entry_price * 0.95,  # 5% profit
                'percentage': 0.30,
                'reason': 'First target'
            })
            
            levels.append({
                'price': entry_price * 0.90,  # 10% profit
                'percentage': 0.40,
                'reason': 'Second target'
            })
            
            if confidence > 0.7:
                levels.append({
                    'price': entry_price * 0.80,  # 20% profit
                    'percentage': 0.30,
                    'reason': 'Final target'
                })
        
        return levels
    
    def detect_real_time_momentum(self, df: pd.DataFrame) -> Dict:
        """
        Detect real-time momentum using recent price action
        """
        if len(df) < 20:
            return {'momentum': 0.0, 'strength': 'weak'}
        
        try:
            # Calculate momentum using multiple methods
            price_change_1m = (df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] if len(df) >= 2 else 0
            price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            price_change_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
            
            # Weighted momentum (recent moves weighted more)
            momentum = (price_change_1m * 0.5 + price_change_5m * 0.3 + price_change_10m * 0.2)
            
            # Volume confirmation
            volume_avg = df['volume'].tail(20).mean()
            volume_recent = df['volume'].tail(5).mean()
            volume_ratio = volume_recent / volume_avg if volume_avg > 0 else 1.0
            
            # Determine strength
            if abs(momentum) > 0.02 and volume_ratio > 1.3:
                strength = 'strong'
            elif abs(momentum) > 0.01 and volume_ratio > 1.2:
                strength = 'medium'
            else:
                strength = 'weak'
            
            return {
                'momentum': float(momentum),
                'strength': strength,
                'volume_confirmation': volume_ratio > 1.2,
                'direction': 'up' if momentum > 0 else 'down'
            }
        
        except Exception as e:
            print(f"Error detecting momentum: {e}")
            return {'momentum': 0.0, 'strength': 'weak'}
    
    def calculate_support_resistance(self, df: pd.DataFrame) -> Dict:
        """
        Calculate key support and resistance levels
        """
        if len(df) < 50:
            return {}
        
        try:
            # Use recent data for dynamic levels
            recent_highs = df['high'].tail(100)
            recent_lows = df['low'].tail(100)
            
            # Find clusters (support/resistance zones)
            resistance = float(recent_highs.rolling(10).max().max())
            support = float(recent_lows.rolling(10).min().min())
            
            current_price = float(df['close'].iloc[-1])
            
            # Calculate distance to levels
            dist_to_resistance = (resistance - current_price) / current_price
            dist_to_support = (current_price - support) / current_price
            
            return {
                'resistance': resistance,
                'support': support,
                'current': current_price,
                'dist_to_resistance': float(dist_to_resistance),
                'dist_to_support': float(dist_to_support),
                'near_resistance': dist_to_resistance < 0.02,  # Within 2%
                'near_support': dist_to_support < 0.02
            }
        
        except Exception:
            return {}

