"""
Adaptive Filter System - Automatically adjusts filters based on performance
"""
from typing import Dict, Optional
from datetime import datetime, timedelta
from collections import deque
import config


class AdaptiveFilter:
    """
    Adaptively adjusts signal filters based on recent performance.
    Learns what works and adjusts thresholds automatically.
    """
    
    def __init__(self):
        self.performance_history = deque(maxlen=100)  # Last 100 signals
        self.recent_wins = deque(maxlen=20)
        self.recent_losses = deque(maxlen=20)
        self.adaptive_thresholds = {
            'min_confidence': config.MIN_CONFIDENCE_SCORE,
            'min_ai_confidence': getattr(config, 'MIN_AI_CONFIDENCE', 0.6),
            'min_price_move': config.MIN_PRICE_MOVE_FOR_SIGNAL,
            'min_volume_spike': config.MIN_VOLUME_SPIKE_FOR_SIGNAL,
            'min_rr_ratio': getattr(config, 'MIN_RISK_REWARD_RATIO', 1.2)
        }
        self.last_adjustment = datetime.now()
        self.adjustment_interval = timedelta(hours=6)  # Adjust every 6 hours
    
    def record_signal(self, signal: Dict, outcome: str, profit_pct: float = 0.0):
        """
        Record a signal and its outcome for learning
        """
        self.performance_history.append({
            'signal': signal,
            'outcome': outcome,
            'profit_pct': profit_pct,
            'timestamp': datetime.now()
        })
        
        if outcome == 'win':
            self.recent_wins.append({
                'confidence': signal.get('confidence', 0),
                'probability': signal.get('signal_probability', 0),
                'price_change_10m': signal.get('price_change_10m', 0),
                'volume_change': signal.get('volume_change', 1.0),
                'profit_pct': profit_pct
            })
        elif outcome == 'loss':
            self.recent_losses.append({
                'confidence': signal.get('confidence', 0),
                'probability': signal.get('signal_probability', 0),
                'price_change_10m': signal.get('price_change_10m', 0),
                'volume_change': signal.get('volume_change', 1.0),
                'loss_pct': abs(profit_pct)
            })
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate current performance metrics
        """
        if len(self.performance_history) < 10:
            return {
                'win_rate': 0.0,
                'total_signals': len(self.performance_history),
                'avg_profit': 0.0,
                'needs_adjustment': False
            }
        
        wins = sum(1 for p in self.performance_history if p['outcome'] == 'win')
        losses = sum(1 for p in self.performance_history if p['outcome'] == 'loss')
        total = len(self.performance_history)
        
        win_rate = wins / total if total > 0 else 0.0
        avg_profit = sum(p['profit_pct'] for p in self.performance_history) / total if total > 0 else 0.0
        
        # Determine if adjustment is needed
        target_win_rate = getattr(config, 'TARGET_WIN_RATE', 0.55)
        needs_adjustment = (
            (win_rate < target_win_rate - 0.1 and total >= 20) or  # 10% below target
            (win_rate > target_win_rate + 0.15 and total >= 15)     # 15% above target (can be more aggressive)
        )
        
        return {
            'win_rate': win_rate,
            'total_signals': total,
            'wins': wins,
            'losses': losses,
            'avg_profit': avg_profit,
            'needs_adjustment': needs_adjustment
        }
    
    def analyze_winning_patterns(self) -> Dict:
        """
        Analyze what makes signals win
        """
        if len(self.recent_wins) < 5:
            return {}
        
        avg_confidence = sum(w['confidence'] for w in self.recent_wins) / len(self.recent_wins)
        avg_probability = sum(w['probability'] for w in self.recent_wins) / len(self.recent_wins)
        avg_price_move = sum(abs(w['price_change_10m']) for w in self.recent_wins) / len(self.recent_wins)
        avg_volume = sum(w['volume_change'] for w in self.recent_wins) / len(self.recent_wins)
        
        return {
            'avg_confidence': avg_confidence,
            'avg_probability': avg_probability,
            'avg_price_move': avg_price_move,
            'avg_volume_spike': avg_volume
        }
    
    def analyze_losing_patterns(self) -> Dict:
        """
        Analyze what makes signals lose
        """
        if len(self.recent_losses) < 5:
            return {}
        
        avg_confidence = sum(l['confidence'] for l in self.recent_losses) / len(self.recent_losses)
        avg_probability = sum(l['probability'] for l in self.recent_losses) / len(self.recent_losses)
        avg_price_move = sum(abs(l['price_change_10m']) for l in self.recent_losses) / len(self.recent_losses)
        avg_volume = sum(l['volume_change'] for l in self.recent_losses) / len(self.recent_losses)
        
        return {
            'avg_confidence': avg_confidence,
            'avg_probability': avg_probability,
            'avg_price_move': avg_price_move,
            'avg_volume_spike': avg_volume
        }
    
    def adjust_filters(self) -> Dict:
        """
        Automatically adjust filters based on performance
        Returns dict of adjustments made
        """
        if datetime.now() - self.last_adjustment < self.adjustment_interval:
            return {}  # Too soon to adjust
        
        metrics = self.calculate_performance_metrics()
        if not metrics['needs_adjustment']:
            return {}
        
        adjustments = {}
        win_rate = metrics['win_rate']
        target_win_rate = getattr(config, 'TARGET_WIN_RATE', 0.55)
        
        # Analyze patterns
        winning_patterns = self.analyze_winning_patterns()
        losing_patterns = self.analyze_losing_patterns()
        
        # Adjust confidence threshold
        if win_rate < target_win_rate - 0.1:
            # Win rate too low - increase thresholds
            if winning_patterns:
                # Use average confidence of winners as new minimum
                new_min_conf = min(0.90, max(
                    self.adaptive_thresholds['min_confidence'] + 0.03,
                    winning_patterns.get('avg_probability', 0.55)
                ))
                adjustments['min_confidence'] = {
                    'old': self.adaptive_thresholds['min_confidence'],
                    'new': new_min_conf,
                    'reason': f'Win rate {win_rate:.1%} below target. Increasing threshold.'
                }
                self.adaptive_thresholds['min_confidence'] = new_min_conf
                
                # Also adjust AI confidence
                if winning_patterns.get('avg_confidence', 0) > self.adaptive_thresholds['min_ai_confidence']:
                    new_ai_conf = min(0.95, winning_patterns['avg_confidence'] + 0.02)
                    adjustments['min_ai_confidence'] = {
                        'old': self.adaptive_thresholds['min_ai_confidence'],
                        'new': new_ai_conf,
                        'reason': 'Matching winning pattern confidence levels.'
                    }
                    self.adaptive_thresholds['min_ai_confidence'] = new_ai_conf
        
        elif win_rate > target_win_rate + 0.15:
            # Win rate very high - can be slightly more aggressive
            if winning_patterns and metrics['total_signals'] < 30:
                # Only if we have few signals
                new_min_conf = max(0.40, self.adaptive_thresholds['min_confidence'] - 0.02)
                adjustments['min_confidence'] = {
                    'old': self.adaptive_thresholds['min_confidence'],
                    'new': new_min_conf,
                    'reason': f'Excellent win rate {win_rate:.1%} but few signals. Slightly reducing threshold.'
                }
                self.adaptive_thresholds['min_confidence'] = new_min_conf
        
        # Adjust price move threshold based on winning patterns
        if winning_patterns and losing_patterns:
            win_price_move = winning_patterns.get('avg_price_move', 0.015)
            lose_price_move = losing_patterns.get('avg_price_move', 0.015)
            
            # If winners have higher price moves, increase threshold
            if win_price_move > lose_price_move * 1.2:
                new_price_move = min(0.05, win_price_move * 0.8)  # 80% of winning move
                adjustments['min_price_move'] = {
                    'old': self.adaptive_thresholds['min_price_move'],
                    'new': new_price_move,
                    'reason': f'Winners have {win_price_move:.2%} avg move vs {lose_price_move:.2%} for losers.'
                }
                self.adaptive_thresholds['min_price_move'] = new_price_move
        
        # Adjust volume spike threshold
        if winning_patterns and losing_patterns:
            win_volume = winning_patterns.get('avg_volume_spike', 1.4)
            lose_volume = losing_patterns.get('avg_volume_spike', 1.4)
            
            if win_volume > lose_volume * 1.1:
                new_volume = max(1.2, win_volume * 0.9)  # 90% of winning volume
                adjustments['min_volume_spike'] = {
                    'old': self.adaptive_thresholds['min_volume_spike'],
                    'new': new_volume,
                    'reason': f'Winners have {win_volume:.2f}x volume vs {lose_volume:.2f}x for losers.'
                }
                self.adaptive_thresholds['min_volume_spike'] = new_volume
        
        if adjustments:
            self.last_adjustment = datetime.now()
            # Apply adjustments to config
            for key, adj in adjustments.items():
                if key == 'min_confidence':
                    config.MIN_CONFIDENCE_SCORE = adj['new']
                elif key == 'min_ai_confidence':
                    setattr(config, 'MIN_AI_CONFIDENCE', adj['new'])
                elif key == 'min_price_move':
                    config.MIN_PRICE_MOVE_FOR_SIGNAL = adj['new']
                elif key == 'min_volume_spike':
                    config.MIN_VOLUME_SPIKE_FOR_SIGNAL = adj['new']
        
        return adjustments
    
    def get_adaptive_thresholds(self) -> Dict:
        """
        Get current adaptive thresholds
        """
        return self.adaptive_thresholds.copy()
    
    def should_accept_signal(self, analysis: Dict) -> bool:
        """
        Check if signal should be accepted based on adaptive thresholds
        """
        confidence = analysis.get('confidence', 0)
        probability = analysis.get('signal_probability', 0)
        
        min_conf = self.adaptive_thresholds['min_confidence']
        min_ai_conf = self.adaptive_thresholds['min_ai_confidence']
        
        return probability >= min_conf and confidence >= min_ai_conf

