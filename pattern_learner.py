"""
Pattern Learning Module - Learns from successful signals to improve detection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os


class PatternLearner:
    """
    Learns patterns from successful signals and uses them to improve detection.
    Tracks winning patterns and applies them to filter/boost signals.
    """
    
    def __init__(self):
        self.winning_patterns = []  # List of successful signal patterns
        self.losing_patterns = []   # List of failed signal patterns
        self.pattern_file = 'models/winning_patterns.json'
        self.max_patterns = 500  # Keep last 500 patterns
        self._load_patterns()
        
        # Pattern statistics
        self.pattern_stats = {
            'total_wins': 0,
            'total_losses': 0,
            'pattern_matches': defaultdict(int),
            'pattern_wins': defaultdict(int)
        }
    
    def _load_patterns(self):
        """Load saved patterns from disk"""
        if os.path.exists(self.pattern_file):
            try:
                with open(self.pattern_file, 'r') as f:
                    data = json.load(f)
                    self.winning_patterns = data.get('winning_patterns', [])
                    self.losing_patterns = data.get('losing_patterns', [])
                    self.pattern_stats = data.get('pattern_stats', self.pattern_stats)
                print(f"Loaded {len(self.winning_patterns)} winning patterns")
            except Exception as e:
                print(f"Error loading patterns: {e}")
    
    def _save_patterns(self):
        """Save patterns to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            data = {
                'winning_patterns': self.winning_patterns[-self.max_patterns:],
                'losing_patterns': self.losing_patterns[-self.max_patterns:],
                'pattern_stats': self.pattern_stats,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.pattern_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving patterns: {e}")
    
    def extract_pattern_features(self, analysis: Dict, df: pd.DataFrame) -> Dict:
        """
        Extract key features from a signal that form a pattern
        """
        if df is None or len(df) < 20:
            return {}
        
        price_change_10m = analysis.get('price_change_10m', 0)
        price_change_5m = analysis.get('price_change_5m', 0)
        volume_change = analysis.get('volume_change', 1.0)
        
        # Calculate additional pattern features
        recent_prices = df['close'].tail(20).values
        recent_volumes = df['volume'].tail(20).values
        
        # Price momentum
        price_momentum = (recent_prices[-1] - recent_prices[-5]) / recent_prices[-5] if len(recent_prices) >= 5 else 0
        
        # Volume trend
        volume_trend = np.mean(recent_volumes[-5:]) / (np.mean(recent_volumes[-10:-5]) + 1e-10) if len(recent_volumes) >= 10 else 1.0
        
        # Volatility
        volatility = np.std(recent_prices[-20:]) / (np.mean(recent_prices[-20:]) + 1e-10) if len(recent_prices) >= 20 else 0
        
        # RSI-like indicator
        price_changes = df['close'].pct_change().tail(14).dropna()
        if len(price_changes) >= 14:
            gains = price_changes[price_changes > 0].sum()
            losses = abs(price_changes[price_changes < 0].sum())
            rsi_like = gains / (gains + losses + 1e-10) if (gains + losses) > 0 else 0.5
        else:
            rsi_like = 0.5
        
        pattern = {
            'signal_type': analysis.get('signal_type'),
            'signal_probability': round(analysis.get('signal_probability', 0), 2),
            'confidence': round(analysis.get('confidence', 0), 2),
            'price_change_10m': round(price_change_10m, 4),
            'price_change_5m': round(price_change_5m, 4),
            'volume_change': round(volume_change, 2),
            'price_momentum': round(price_momentum, 4),
            'volume_trend': round(volume_trend, 2),
            'volatility': round(volatility, 4),
            'rsi_like': round(rsi_like, 2),
            'risk_reward_ratio': round(analysis.get('risk_reward_ratio', 0), 2),
        }
        
        return pattern
    
    def learn_from_signal(self, signal: Dict, outcome: str, profit_pct: float = 0.0):
        """
        Learn from a signal outcome (win/loss/timeout)
        """
        if not signal.get('features'):
            return
        
        pattern = self.extract_pattern_features(signal, None)  # We'll use features from signal
        
        if outcome == 'win':
            self.winning_patterns.append({
                'pattern': pattern,
                'profit_pct': profit_pct,
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.get('symbol')
            })
            self.pattern_stats['total_wins'] += 1
            
            # Keep only recent patterns
            if len(self.winning_patterns) > self.max_patterns:
                self.winning_patterns = self.winning_patterns[-self.max_patterns:]
        
        elif outcome == 'loss':
            self.losing_patterns.append({
                'pattern': pattern,
                'loss_pct': abs(profit_pct),
                'timestamp': datetime.now().isoformat(),
                'symbol': signal.get('symbol')
            })
            self.pattern_stats['total_losses'] += 1
            
            if len(self.losing_patterns) > self.max_patterns:
                self.losing_patterns = self.losing_patterns[-self.max_patterns:]
        
        # Save periodically
        if len(self.winning_patterns) % 10 == 0:
            self._save_patterns()
    
    def match_pattern(self, analysis: Dict, df: pd.DataFrame) -> Dict:
        """
        Check if current signal matches known winning patterns
        Returns match score and boost factor
        """
        if len(self.winning_patterns) < 5:
            return {
                'match_score': 0.0,
                'boost_factor': 1.0,
                'matches': 0,
                'should_boost': False
            }
        
        current_pattern = self.extract_pattern_features(analysis, df)
        matches = 0
        total_score = 0.0
        
        # Compare with recent winning patterns (last 100)
        recent_winners = self.winning_patterns[-100:]
        
        for winner in recent_winners:
            winner_pattern = winner['pattern']
            similarity = self._calculate_similarity(current_pattern, winner_pattern)
            
            if similarity > 0.7:  # 70% similarity threshold
                matches += 1
                # Weight by profit and recency
                profit_weight = min(1.0, winner.get('profit_pct', 0) / 20.0)  # Normalize profit
                total_score += similarity * (1.0 + profit_weight)
        
        if matches > 0:
            avg_score = total_score / matches
            # Boost factor: 1.0 (no boost) to 1.3 (30% boost)
            boost_factor = 1.0 + (avg_score - 0.7) * 0.5  # Scale boost
            boost_factor = min(1.3, max(1.0, boost_factor))
            
            return {
                'match_score': avg_score,
                'boost_factor': boost_factor,
                'matches': matches,
                'should_boost': matches >= 3  # Need at least 3 matches
            }
        
        return {
            'match_score': 0.0,
            'boost_factor': 1.0,
            'matches': 0,
            'should_boost': False
        }
    
    def _calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """
        Calculate similarity between two patterns (0-1)
        """
        if not pattern1 or not pattern2:
            return 0.0
        
        # Key features to compare
        features = [
            'signal_type', 'price_change_10m', 'volume_change',
            'price_momentum', 'volume_trend', 'volatility', 'rsi_like'
        ]
        
        similarities = []
        for feature in features:
            val1 = pattern1.get(feature, 0)
            val2 = pattern2.get(feature, 0)
            
            if feature == 'signal_type':
                # Exact match for signal type
                similarity = 1.0 if val1 == val2 else 0.0
            else:
                # Numerical similarity (tolerance-based)
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # Use relative difference
                    max_val = max(abs(val1), abs(val2), 1e-10)
                    diff = abs(val1 - val2) / max_val
                    similarity = max(0.0, 1.0 - diff)
                else:
                    similarity = 1.0 if val1 == val2 else 0.0
            
            similarities.append(similarity)
        
        # Weighted average (signal_type and volume_change are more important)
        weights = [0.2, 0.15, 0.2, 0.1, 0.15, 0.1, 0.1]
        weighted_similarity = sum(s * w for s, w in zip(similarities, weights))
        
        return weighted_similarity
    
    def should_skip_signal(self, analysis: Dict, df: pd.DataFrame) -> bool:
        """
        Check if signal should be skipped based on losing patterns
        """
        if len(self.losing_patterns) < 5:
            return False
        
        current_pattern = self.extract_pattern_features(analysis, df)
        matches = 0
        
        # Check against recent losing patterns
        recent_losers = self.losing_patterns[-50:]
        
        for loser in recent_losers:
            loser_pattern = loser['pattern']
            similarity = self._calculate_similarity(current_pattern, loser_pattern)
            
            if similarity > 0.75:  # High similarity to losing pattern
                matches += 1
        
        # Skip if matches too many losing patterns
        return matches >= 3
    
    def get_pattern_insights(self) -> Dict:
        """
        Get insights about learned patterns
        """
        if len(self.winning_patterns) < 10:
            return {
                'total_patterns': len(self.winning_patterns),
                'insights': ['Not enough data yet']
            }
        
        # Analyze common features in winning patterns
        winning_features = {
            'avg_price_change_10m': [],
            'avg_volume_change': [],
            'avg_confidence': [],
            'common_signal_type': defaultdict(int)
        }
        
        for winner in self.winning_patterns[-100:]:
            pattern = winner['pattern']
            winning_features['avg_price_change_10m'].append(abs(pattern.get('price_change_10m', 0)))
            winning_features['avg_volume_change'].append(pattern.get('volume_change', 1.0))
            winning_features['avg_confidence'].append(pattern.get('confidence', 0))
            winning_features['common_signal_type'][pattern.get('signal_type', 'UNKNOWN')] += 1
        
        insights = [
            f"Average 10m price change in winners: {np.mean(winning_features['avg_price_change_10m']):.2%}",
            f"Average volume spike in winners: {np.mean(winning_features['avg_volume_change']):.2f}x",
            f"Average confidence in winners: {np.mean(winning_features['avg_confidence']):.2%}",
            f"Most common signal type: {max(winning_features['common_signal_type'], key=winning_features['common_signal_type'].get)}"
        ]
        
        return {
            'total_patterns': len(self.winning_patterns),
            'win_rate': self.pattern_stats['total_wins'] / (self.pattern_stats['total_wins'] + self.pattern_stats['total_losses'] + 1e-10),
            'insights': insights
        }

