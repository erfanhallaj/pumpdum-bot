"""
Loss Analyzer - Deep analysis of losing signals to prevent repetition
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
import os


class LossAnalyzer:
    """
    Analyzes losing signals in depth to:
    1. Identify failure patterns
    2. Extract root causes
    3. Create filters to prevent repetition
    4. Learn from mistakes
    """
    
    def __init__(self):
        self.loss_patterns = []  # Detailed loss records
        self.failure_reasons = defaultdict(int)  # Count of failure reasons
        self.symbol_loss_history = defaultdict(list)  # Loss history per symbol
        self.loss_file = 'models/loss_patterns.json'
        self.max_losses = 1000  # Keep last 1000 losses
        self._load_loss_data()
        
        # Failure categories
        self.failure_categories = {
            'stop_loss_hit': 'Stop loss was hit (price moved against signal)',
            'timeout': 'Signal timed out (no significant move)',
            'false_breakout': 'False breakout pattern detected',
            'low_volume': 'Insufficient volume to sustain move',
            'market_reversal': 'Market reversed direction',
            'scam_coin': 'Coin showed scam characteristics',
            'low_liquidity': 'Low liquidity caused slippage',
            'wrong_timing': 'Signal timing was incorrect',
            'weak_momentum': 'Momentum was too weak',
            'correlation_failure': 'Correlation with market failed'
        }
    
    def _load_loss_data(self):
        """Load saved loss data from disk"""
        if os.path.exists(self.loss_file):
            try:
                with open(self.loss_file, 'r') as f:
                    data = json.load(f)
                    self.loss_patterns = data.get('loss_patterns', [])[-self.max_losses:]
                    self.failure_reasons = defaultdict(int, data.get('failure_reasons', {}))
                    # Convert symbol_loss_history back to defaultdict
                    symbol_data = data.get('symbol_loss_history', {})
                    for symbol, losses in symbol_data.items():
                        self.symbol_loss_history[symbol] = losses[-20:]  # Keep last 20 per symbol
                print(f"Loaded {len(self.loss_patterns)} loss patterns")
            except Exception as e:
                print(f"Error loading loss data: {e}")
    
    def _save_loss_data(self):
        """Save loss data to disk"""
        try:
            os.makedirs('models', exist_ok=True)
            data = {
                'loss_patterns': self.loss_patterns[-self.max_losses:],
                'failure_reasons': dict(self.failure_reasons),
                'symbol_loss_history': {k: v[-20:] for k, v in self.symbol_loss_history.items()},
                'last_updated': datetime.now().isoformat()
            }
            with open(self.loss_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving loss data: {e}")
    
    def analyze_loss(self, signal: Dict, outcome: str, 
                    final_price: float, entry_price: float,
                    stop_loss: float, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Deep analysis of a losing signal to identify root cause
        """
        if outcome not in ['loss', 'timeout']:
            return {}
        
        loss_analysis = {
            'symbol': signal.get('symbol'),
            'signal_type': signal.get('signal_type'),
            'timestamp': signal.get('timestamp', datetime.now()),
            'entry_price': entry_price,
            'final_price': final_price,
            'stop_loss': stop_loss,
            'outcome': outcome,
            'failure_reason': None,
            'failure_category': None,
            'lessons_learned': [],
            'should_blacklist': False
        }
        
        # Calculate loss details
        if signal.get('signal_type') == 'PUMP':
            loss_pct = (final_price - entry_price) / entry_price * 100
            hit_sl = final_price <= stop_loss
        else:  # DUMP
            loss_pct = (entry_price - final_price) / entry_price * 100
            hit_sl = final_price >= stop_loss
        
        loss_analysis['loss_pct'] = loss_pct
        loss_analysis['hit_stop_loss'] = hit_sl
        
        # Analyze failure reason
        failure_reason = self._identify_failure_reason(
            signal, outcome, final_price, entry_price, stop_loss, df
        )
        loss_analysis['failure_reason'] = failure_reason['reason']
        loss_analysis['failure_category'] = failure_reason['category']
        loss_analysis['confidence'] = failure_reason['confidence']
        loss_analysis['lessons_learned'] = failure_reason['lessons']
        
        # Check if should blacklist symbol
        symbol = signal.get('symbol')
        if symbol:
            self.symbol_loss_history[symbol].append({
                'timestamp': datetime.now().isoformat(),
                'loss_pct': loss_pct,
                'failure_reason': failure_reason['reason']
            })
            
            # Blacklist if too many recent losses
            recent_losses = [l for l in self.symbol_loss_history[symbol] 
                           if (datetime.now() - datetime.fromisoformat(l['timestamp'])).days < 7]
            if len(recent_losses) >= 5:  # 5+ losses in last 7 days
                loss_analysis['should_blacklist'] = True
                loss_analysis['lessons_learned'].append(
                    f"Symbol {symbol} has {len(recent_losses)} losses in 7 days - consider blacklisting"
                )
        
        # Store loss pattern
        self.loss_patterns.append(loss_analysis)
        self.failure_reasons[failure_reason['category']] += 1
        
        # Keep only recent losses
        if len(self.loss_patterns) > self.max_losses:
            self.loss_patterns = self.loss_patterns[-self.max_losses:]
        
        # Save periodically
        if len(self.loss_patterns) % 10 == 0:
            self._save_loss_data()
        
        return loss_analysis
    
    def _identify_failure_reason(self, signal: Dict, outcome: str,
                                final_price: float, entry_price: float,
                                stop_loss: float, df: Optional[pd.DataFrame]) -> Dict:
        """
        Identify the root cause of failure
        """
        signal_type = signal.get('signal_type')
        confidence = signal.get('confidence', 0)
        probability = signal.get('signal_probability', 0)
        price_change_10m = signal.get('price_change_10m', 0)
        volume_change = signal.get('volume_change', 1.0)
        
        reasons = []
        lessons = []
        category = 'unknown'
        confidence_score = 0.5
        
        # 1. Stop Loss Hit Analysis
        if outcome == 'loss':
            if signal_type == 'PUMP' and final_price <= stop_loss:
                category = 'stop_loss_hit'
                confidence_score = 0.9
                reasons.append("Stop loss was hit - price moved against signal")
                
                # Analyze why SL was hit
                if df is not None and len(df) >= 20:
                    # Check if there was a sudden reversal
                    recent_prices = df['close'].tail(20).values
                    if len(recent_prices) >= 10:
                        price_trend = (recent_prices[-1] - recent_prices[-10]) / recent_prices[-10]
                        if price_trend < -0.05:  # >5% drop
                            reasons.append("Sudden price reversal detected")
                            lessons.append("Avoid signals during potential reversal points")
                            category = 'market_reversal'
                
                # Check volume
                if volume_change < 1.2:
                    reasons.append("Volume spike was insufficient")
                    lessons.append("Require stronger volume confirmation (>1.2x)")
                    category = 'low_volume'
            
            elif signal_type == 'DUMP' and final_price >= stop_loss:
                category = 'stop_loss_hit'
                confidence_score = 0.9
                reasons.append("Stop loss was hit - price moved up instead of down")
        
        # 2. Timeout Analysis
        elif outcome == 'timeout':
            category = 'timeout'
            confidence_score = 0.8
            reasons.append("Signal timed out - no significant price movement")
            
            # Analyze why timeout occurred
            if abs(price_change_10m) < 0.01:  # Less than 1% move
                reasons.append("Initial price move was too weak")
                lessons.append("Require stronger initial momentum (>1.5%)")
                category = 'weak_momentum'
            
            if volume_change < 1.3:
                reasons.append("Volume spike was not sustained")
                lessons.append("Volume must remain elevated, not just spike")
                category = 'low_volume'
        
        # 3. Confidence Analysis
        if confidence < 0.50:
            reasons.append(f"Low AI confidence ({confidence:.1%})")
            lessons.append("Increase minimum confidence threshold")
            if category == 'unknown':
                category = 'wrong_timing'
        
        if probability < 0.55:
            reasons.append(f"Low signal probability ({probability:.1%})")
            lessons.append("Require higher probability threshold")
            if category == 'unknown':
                category = 'wrong_timing'
        
        # 4. Risk/Reward Analysis
        risk_reward = signal.get('risk_reward_ratio', 0)
        if risk_reward < 1.2:
            reasons.append(f"Low risk/reward ratio ({risk_reward:.2f})")
            lessons.append("Require minimum 1.2 risk/reward ratio")
        
        # 5. Pattern Analysis (if we have historical data)
        if df is not None and len(df) >= 50:
            # Check for false breakout
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            current_price = df['close'].iloc[-1]
            
            if signal_type == 'PUMP':
                # If price was near high but didn't break through
                if current_price < recent_high * 0.98:  # 2% below high
                    reasons.append("False breakout - price failed to break resistance")
                    lessons.append("Wait for confirmed breakout above resistance")
                    category = 'false_breakout'
        
        # Default reason if none found
        if category == 'unknown':
            category = 'wrong_timing'
            reasons.append("Unable to identify specific cause - likely timing issue")
        
        return {
            'reason': ' | '.join(reasons) if reasons else 'Unknown failure',
            'category': category,
            'confidence': confidence_score,
            'lessons': lessons
        }
    
    def should_skip_signal(self, analysis: Dict, df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Check if signal should be skipped based on learned loss patterns
        Returns (should_skip, reason)
        """
        symbol = analysis.get('symbol')
        signal_type = analysis.get('signal_type')
        confidence = analysis.get('confidence', 0)
        probability = analysis.get('signal_probability', 0)
        price_change_10m = analysis.get('price_change_10m', 0)
        volume_change = analysis.get('volume_change', 1.0)
        
        # 1. Check symbol loss history
        if symbol and symbol in self.symbol_loss_history:
            recent_losses = [l for l in self.symbol_loss_history[symbol]
                           if (datetime.now() - datetime.fromisoformat(l['timestamp'])).days < 3]
            if len(recent_losses) >= 3:
                return True, f"Symbol {symbol} has {len(recent_losses)} recent losses"
        
        # 2. Check for known failure patterns
        recent_losses = [l for l in self.loss_patterns[-50:]]
        
        for loss in recent_losses:
            # Check if current signal matches a recent loss pattern
            if loss['signal_type'] == signal_type:
                # Similar confidence/probability that led to loss
                if abs(loss.get('confidence', 0) - confidence) < 0.1 and confidence < 0.60:
                    if loss.get('failure_category') in ['wrong_timing', 'weak_momentum']:
                        return True, f"Matches recent loss pattern: {loss.get('failure_category')}"
        
        # 3. Check for common failure indicators
        # Low confidence + low probability = high risk
        if confidence < 0.55 and probability < 0.60:
            return True, "Low confidence and probability - matches failure pattern"
        
        # Weak momentum + low volume = likely timeout
        if abs(price_change_10m) < 0.01 and volume_change < 1.2:
            return True, "Weak momentum and volume - likely timeout pattern"
        
        # 4. Check failure reason frequency
        top_failure = max(self.failure_reasons.items(), key=lambda x: x[1], default=(None, 0))
        if top_failure[0] == 'low_volume' and volume_change < 1.3:
            return True, "Volume pattern matches common failure cause"
        
        return False, ""
    
    def get_failure_statistics(self) -> Dict:
        """
        Get statistics about failures for learning
        """
        if len(self.loss_patterns) < 5:
            return {
                'total_losses': len(self.loss_patterns),
                'insights': ['Not enough loss data yet']
            }
        
        # Analyze failure categories
        category_counts = defaultdict(int)
        avg_loss_pct = defaultdict(list)
        
        for loss in self.loss_patterns[-100:]:  # Last 100 losses
            category = loss.get('failure_category', 'unknown')
            category_counts[category] += 1
            if 'loss_pct' in loss:
                avg_loss_pct[category].append(abs(loss['loss_pct']))
        
        # Calculate insights
        insights = []
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            avg = np.mean(avg_loss_pct[category]) if category in avg_loss_pct else 0
            insights.append(
                f"{category}: {count} losses, avg loss: {avg:.2f}%"
            )
        
        # Most common failure
        most_common = max(category_counts.items(), key=lambda x: x[1], default=(None, 0))
        
        return {
            'total_losses': len(self.loss_patterns),
            'failure_categories': dict(category_counts),
            'most_common_failure': most_common[0] if most_common[0] else 'unknown',
            'insights': insights,
            'top_lessons': self._extract_top_lessons()
        }
    
    def _extract_top_lessons(self) -> List[str]:
        """
        Extract most common lessons from losses
        """
        all_lessons = []
        for loss in self.loss_patterns[-100:]:
            all_lessons.extend(loss.get('lessons_learned', []))
        
        # Count lesson frequency
        lesson_counts = defaultdict(int)
        for lesson in all_lessons:
            lesson_counts[lesson] += 1
        
        # Return top 5 lessons
        top_lessons = sorted(lesson_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [lesson for lesson, count in top_lessons]
    
    def generate_prevention_filters(self) -> Dict:
        """
        Generate filters based on learned loss patterns
        """
        if len(self.loss_patterns) < 10:
            return {}
        
        filters = {}
        
        # Analyze recent losses
        recent_losses = self.loss_patterns[-50:]
        
        # Filter 1: Minimum confidence based on loss history
        low_conf_losses = [l for l in recent_losses 
                          if l.get('confidence', 1.0) < 0.60]
        if len(low_conf_losses) > len(recent_losses) * 0.4:  # >40% of losses
            filters['min_confidence'] = {
                'current': 0.55,
                'suggested': 0.60,
                'reason': f"{len(low_conf_losses)} losses had confidence <60%"
            }
        
        # Filter 2: Minimum volume spike
        low_vol_losses = [l for l in recent_losses 
                         if l.get('volume_change', 2.0) < 1.3]
        if len(low_vol_losses) > len(recent_losses) * 0.3:  # >30% of losses
            filters['min_volume_spike'] = {
                'current': 1.4,
                'suggested': 1.5,
                'reason': f"{len(low_vol_losses)} losses had volume <1.3x"
            }
        
        # Filter 3: Minimum price move
        weak_momentum_losses = [l for l in recent_losses 
                               if abs(l.get('price_change_10m', 0)) < 0.015]
        if len(weak_momentum_losses) > len(recent_losses) * 0.35:  # >35% of losses
            filters['min_price_move'] = {
                'current': 0.015,
                'suggested': 0.020,
                'reason': f"{len(weak_momentum_losses)} losses had weak momentum"
            }
        
        return filters

