"""
Scam Coin Detection Module
Detects potential scam coins and low-quality projects
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta


class ScamDetector:
    """
    Detects potential scam coins based on various indicators:
    - Extreme price volatility (pump and dump patterns)
    - Low liquidity
    - Suspicious volume patterns
    - Price manipulation signs
    - Low market cap with high volume (wash trading)
    """
    
    def __init__(self):
        self.scam_patterns = {}
        self.blacklisted_symbols = set()
        
    def analyze_coin_quality(self, symbol: str, df: pd.DataFrame, 
                            market_data: Optional[Dict] = None) -> Dict:
        """
        Analyze coin quality and detect scam indicators
        Returns quality score (0-1, higher is better) and scam indicators
        """
        if len(df) < 50:
            return {
                'quality_score': 0.0,
                'is_scam': True,
                'is_low_quality': True,
                'reasons': ['Insufficient data'],
                'risk_level': 'high'
            }
        
        indicators = {
            'extreme_volatility': False,
            'low_liquidity': False,
            'suspicious_volume': False,
            'price_manipulation': False,
            'wash_trading': False,
            'low_market_cap_high_volume': False
        }
        
        reasons = []
        quality_score = 1.0
        
        # 1. Check extreme volatility (pump and dump pattern)
        price_changes = df['close'].pct_change().dropna()
        extreme_moves = (price_changes.abs() > 0.20).sum()  # >20% moves
        if extreme_moves > len(price_changes) * 0.1:  # More than 10% of candles
            indicators['extreme_volatility'] = True
            reasons.append(f"Extreme volatility: {extreme_moves} large moves detected")
            quality_score -= 0.3
        
        # 2. Check for pump and dump pattern (sudden spike then crash)
        recent_prices = df['close'].tail(100).values
        if len(recent_prices) >= 50:
            max_price = np.max(recent_prices)
            min_price = np.min(recent_prices)
            current_price = recent_prices[-1]
            
            # If price dropped >50% from recent high, might be dump
            if max_price > 0 and (max_price - current_price) / max_price > 0.5:
                indicators['price_manipulation'] = True
                reasons.append("Possible pump and dump: >50% drop from recent high")
                quality_score -= 0.4
        
        # 3. Check volume patterns (suspicious spikes)
        volumes = df['volume'].tail(200).values
        if len(volumes) > 20:
            avg_volume = np.mean(volumes)
            max_volume = np.max(volumes)
            
            # If max volume is >10x average, might be wash trading
            if avg_volume > 0 and max_volume / avg_volume > 10:
                indicators['suspicious_volume'] = True
                reasons.append(f"Suspicious volume spike: {max_volume/avg_volume:.1f}x average")
                quality_score -= 0.2
            
            # Check for very low average volume (low liquidity)
            if avg_volume < 1000:  # Very low volume
                indicators['low_liquidity'] = True
                reasons.append("Low liquidity: average volume < $1000")
                quality_score -= 0.3
        
        # 4. Check price consistency (many zero or near-zero changes = low activity)
        zero_changes = (price_changes.abs() < 0.001).sum()
        if zero_changes > len(price_changes) * 0.3:  # >30% of candles unchanged
            indicators['low_liquidity'] = True
            reasons.append("Low trading activity: many zero-change periods")
            quality_score -= 0.2
        
        # 5. Market data checks (if available)
        if market_data:
            market_cap = market_data.get('market_cap', 0)
            volume_24h = market_data.get('volume_24h', 0)
            
            # Low market cap but very high volume = possible wash trading
            if market_cap > 0 and volume_24h > 0:
                volume_to_mcap_ratio = volume_24h / market_cap
                if volume_to_mcap_ratio > 0.5:  # Volume >50% of market cap
                    indicators['wash_trading'] = True
                    reasons.append(f"Suspicious volume/mcap ratio: {volume_to_mcap_ratio:.2f}")
                    quality_score -= 0.4
                
                # Very low market cap with high volume
                if market_cap < 100000 and volume_24h > 100000:  # <$100k mcap, >$100k volume
                    indicators['low_market_cap_high_volume'] = True
                    reasons.append("Very low market cap with high volume (possible scam)")
                    quality_score -= 0.5
        
        # 6. Check for consistent downward trend (dead coin)
        if len(df) >= 100:
            recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-100]) / df['close'].iloc[-100]
            if recent_trend < -0.8:  # >80% drop
                reasons.append("Severe downtrend: >80% drop in recent period")
                quality_score -= 0.3
        
        # Determine risk level
        scam_count = sum(indicators.values())
        if scam_count >= 3 or quality_score < 0.3:
            risk_level = 'high'
            is_scam = True
        elif scam_count >= 2 or quality_score < 0.5:
            risk_level = 'medium'
            is_scam = False
        else:
            risk_level = 'low'
            is_scam = False
        
        is_low_quality = quality_score < 0.6
        
        return {
            'quality_score': max(0.0, min(1.0, quality_score)),
            'is_scam': is_scam,
            'is_low_quality': is_low_quality,
            'indicators': indicators,
            'reasons': reasons,
            'risk_level': risk_level,
            'scam_count': scam_count
        }
    
    def should_skip_coin(self, symbol: str, df: pd.DataFrame, 
                        market_data: Optional[Dict] = None,
                        min_quality_score: float = 0.5) -> bool:
        """
        Determine if coin should be skipped based on quality analysis
        """
        if symbol in self.blacklisted_symbols:
            return True
        
        analysis = self.analyze_coin_quality(symbol, df, market_data)
        
        # Skip if scam or low quality
        if analysis['is_scam']:
            self.blacklisted_symbols.add(symbol)
            return True
        
        if analysis['quality_score'] < min_quality_score:
            return True
        
        return False
    
    def get_risk_reward_ratio(self, entry: float, exit1: float, 
                             stop_loss: float, signal_type: str) -> float:
        """
        Calculate risk/reward ratio for a signal
        Returns ratio (higher is better)
        """
        if signal_type == 'PUMP':
            reward = abs(exit1 - entry)
            risk = abs(entry - stop_loss)
        else:  # DUMP
            reward = abs(entry - exit1)
            risk = abs(stop_loss - entry)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    def is_low_margin_signal(self, entry: float, exit1: float, 
                            stop_loss: float, signal_type: str,
                            min_rr_ratio: float = 1.2) -> bool:
        """
        Check if signal has low profit margin (low risk/reward ratio)
        """
        rr_ratio = self.get_risk_reward_ratio(entry, exit1, stop_loss, signal_type)
        return rr_ratio < min_rr_ratio

