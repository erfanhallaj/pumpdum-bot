"""
AI Self-Improvement Module
Uses free AI APIs to analyze bot performance and suggest/apply improvements
"""
import json
import os
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config
from signal_tracker import SignalTracker


class AISelfImprover:
    """
    Uses free AI APIs to analyze bot performance and automatically improve itself.
    Supports multiple free AI providers:
    - Hugging Face Inference API (free tier)
    - OpenAI API (free tier if available)
    - Local models via Ollama (if installed)
    """
    
    def __init__(self, signal_tracker: SignalTracker):
        self.signal_tracker = signal_tracker
        self.improvement_history = []
        
        # AI Provider Configuration
        self.use_huggingface = True  # Free tier available
        self.use_openai = False  # Requires API key (set in .env)
        self.use_ollama = False  # Local model (requires Ollama installation)
        
        # API Keys (optional, for better models)
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        self.huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')  # Optional but recommended
        
        # Hugging Face model (free, no key needed for inference)
        self.hf_model = "mistralai/Mistral-7B-Instruct-v0.2"  # Free inference model
        
    async def analyze_performance(self, days: int = 7) -> Dict:
        """
        Analyze bot performance over last N days and generate insights.
        Returns analysis dict with recommendations.
        """
        print(f"\n🤖 AI Self-Improver: Analyzing last {days} days...")
        
        # Collect performance data
        stats = self._collect_performance_stats(days)
        
        # Generate AI analysis
        analysis_prompt = self._build_analysis_prompt(stats)
        
        # Get AI response
        ai_response = await self._get_ai_analysis(analysis_prompt)
        
        # Parse and return recommendations
        recommendations = self._parse_ai_response(ai_response, stats)
        
        return {
            'stats': stats,
            'ai_analysis': ai_response,
            'recommendations': recommendations,
            'timestamp': datetime.now()
        }
    
    def _collect_performance_stats(self, days: int) -> Dict:
        """Collect comprehensive performance statistics"""
        stats = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'timeouts': 0,
            'win_rate': 0.0,
            'pump_signals': 0,
            'dump_signals': 0,
            'avg_profit_per_win': 0.0,
            'avg_loss_per_loss': 0.0,
            'best_performing_coins': [],
            'worst_performing_coins': [],
            'current_settings': {
                'min_confidence': config.MIN_CONFIDENCE_SCORE,
                'min_ai_confidence': getattr(config, 'MIN_AI_CONFIDENCE', 0.6),
                'symbol_min_win_rate': getattr(config, 'SYMBOL_MIN_WIN_RATE', 0.45),
            }
        }
        
        # Aggregate stats from signal tracker
        cutoff = datetime.utcnow() - timedelta(days=days)
        for sig in self.signal_tracker.history:
            if sig.timestamp < cutoff:
                continue
            
            stats['total_signals'] += 1
            if sig.signal_type == 'PUMP':
                stats['pump_signals'] += 1
            else:
                stats['dump_signals'] += 1
            
            if sig.status == 'win':
                stats['wins'] += 1
            elif sig.status == 'loss':
                stats['losses'] += 1
            elif sig.status == 'timeout':
                stats['timeouts'] += 1
        
        if stats['total_signals'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_signals']
        
        # Get symbol-level stats
        symbol_stats = {}
        for sig in self.signal_tracker.history:
            if sig.timestamp < cutoff:
                continue
            if sig.symbol not in symbol_stats:
                symbol_stats[sig.symbol] = {'wins': 0, 'total': 0}
            symbol_stats[sig.symbol]['total'] += 1
            if sig.status == 'win':
                symbol_stats[sig.symbol]['wins'] += 1
        
        # Sort by performance
        for symbol, data in symbol_stats.items():
            win_rate = data['wins'] / data['total'] if data['total'] > 0 else 0
            if win_rate >= 0.6 and data['total'] >= 3:
                stats['best_performing_coins'].append((symbol, win_rate, data['total']))
            elif win_rate < 0.3 and data['total'] >= 3:
                stats['worst_performing_coins'].append((symbol, win_rate, data['total']))
        
        stats['best_performing_coins'].sort(key=lambda x: x[1], reverse=True)
        stats['worst_performing_coins'].sort(key=lambda x: x[1])
        
        return stats
    
    def _build_analysis_prompt(self, stats: Dict) -> str:
        """Build prompt for AI analysis"""
        prompt = f"""You are an expert AI trading bot analyst. Analyze this cryptocurrency pump detection bot's performance:

PERFORMANCE DATA (Last 7 days):
- Total Signals: {stats['total_signals']}
- Win Rate: {stats['win_rate']:.2%}
- Wins: {stats['wins']}, Losses: {stats['losses']}, Timeouts: {stats['timeouts']}
- Pump Signals: {stats['pump_signals']}, Dump Signals: {stats['dump_signals']}

CURRENT SETTINGS:
- MIN_CONFIDENCE_SCORE: {stats['current_settings']['min_confidence']:.2f}
- MIN_AI_CONFIDENCE: {stats['current_settings']['min_ai_confidence']:.2f}
- SYMBOL_MIN_WIN_RATE: {stats['current_settings']['symbol_min_win_rate']:.2f}

BEST PERFORMING COINS (Top 5):
{chr(10).join([f"- {coin[0]}: {coin[1]:.1%} win rate ({coin[2]} signals)" for coin in stats['best_performing_coins'][:5]])}

WORST PERFORMING COINS (Top 5):
{chr(10).join([f"- {coin[0]}: {coin[1]:.1%} win rate ({coin[2]} signals)" for coin in stats['worst_performing_coins'][:5]])}

TASK: Analyze this data and provide:
1. What's working well?
2. What needs improvement?
3. Specific recommendations for config.py settings (MIN_CONFIDENCE_SCORE, MIN_AI_CONFIDENCE, etc.)
4. Should we focus more on certain coin types or patterns?

Respond in JSON format:
{{
  "analysis": "brief analysis text",
  "recommendations": [
    {{"setting": "MIN_CONFIDENCE_SCORE", "current": 0.55, "suggested": 0.60, "reason": "..."}},
    ...
  ],
  "insights": ["insight 1", "insight 2", ...]
}}
"""
        return prompt
    
    async def _get_ai_analysis(self, prompt: str) -> str:
        """Get AI analysis from available providers"""
        # Try Hugging Face first (free, no key needed)
        if self.use_huggingface:
            try:
                response = await self._query_huggingface(prompt)
                if response:
                    return response
            except Exception as e:
                print(f"⚠️  Hugging Face API error: {e}")
        
        # Fallback: Simple rule-based analysis
        return self._fallback_analysis(prompt)
    
    async def _query_huggingface(self, prompt: str) -> Optional[str]:
        """Query Hugging Face Inference API (free tier)"""
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers = {}
        if self.huggingface_api_key:
            headers["Authorization"] = f"Bearer {self.huggingface_api_key}"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=30) as response:
                if response.status == 200:
                    result = await response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', '')
                elif response.status == 503:
                    # Model is loading, wait and retry
                    await asyncio.sleep(5)
                    return await self._query_huggingface(prompt)
        
        return None
    
    def _fallback_analysis(self, prompt: str) -> str:
        """Fallback rule-based analysis if AI API unavailable"""
        # Simple heuristic-based recommendations
        return json.dumps({
            "analysis": "Using rule-based analysis (AI API unavailable)",
            "recommendations": [],
            "insights": ["Consider setting up Hugging Face API key for better analysis"]
        })
    
    def _parse_ai_response(self, ai_response: str, stats: Dict) -> List[Dict]:
        """Parse AI response and extract actionable recommendations"""
        recommendations = []
        
        try:
            # Try to parse as JSON
            if ai_response.strip().startswith('{'):
                data = json.loads(ai_response)
                recommendations = data.get('recommendations', [])
            else:
                # Try to extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    recommendations = data.get('recommendations', [])
        except:
            pass
        
        # Enhanced rule-based recommendations based on performance
        win_rate = stats.get('win_rate', 0)
        total_signals = stats.get('total_signals', 0)
        losses = stats.get('losses', 0)
        timeouts = stats.get('timeouts', 0)
        
        # Analyze symbol performance
        worst_coins = stats.get('worst_performing_coins', [])
        best_coins = stats.get('best_performing_coins', [])
        
        # Recommendation 1: Adjust confidence threshold based on win rate
        if len(recommendations) == 0 or not any(r.get('setting') == 'MIN_CONFIDENCE_SCORE' for r in recommendations):
            if win_rate < 0.45 and total_signals >= 10:
                # Low win rate - increase threshold
                new_threshold = min(0.90, config.MIN_CONFIDENCE_SCORE + 0.05)
                recommendations.append({
                    'setting': 'MIN_CONFIDENCE_SCORE',
                    'current': config.MIN_CONFIDENCE_SCORE,
                    'suggested': new_threshold,
                    'reason': f'Win rate {win_rate:.1%} is below target (45%). Increasing threshold to {new_threshold:.2f} for higher quality signals.'
                })
            elif win_rate > 0.70 and total_signals < 15:
                # High win rate but few signals - can be more aggressive
                new_threshold = max(0.40, config.MIN_CONFIDENCE_SCORE - 0.03)
                recommendations.append({
                    'setting': 'MIN_CONFIDENCE_SCORE',
                    'current': config.MIN_CONFIDENCE_SCORE,
                    'suggested': new_threshold,
                    'reason': f'Excellent win rate {win_rate:.1%} but only {total_signals} signals. Can reduce threshold to {new_threshold:.2f}.'
                })
        
        # Recommendation 2: Adjust AI confidence based on performance
        if len(recommendations) == 0 or not any(r.get('setting') == 'MIN_AI_CONFIDENCE' for r in recommendations):
            if win_rate < 0.50:
                new_conf = min(0.90, getattr(config, 'MIN_AI_CONFIDENCE', 0.6) + 0.05)
                recommendations.append({
                    'setting': 'MIN_AI_CONFIDENCE',
                    'current': getattr(config, 'MIN_AI_CONFIDENCE', 0.6),
                    'suggested': new_conf,
                    'reason': f'Increasing AI confidence threshold to {new_conf:.2f} to improve signal quality.'
                })
        
        # Recommendation 3: Adjust risk/reward ratio based on losses
        if losses > wins * 1.5:  # More losses than wins
            current_rr = getattr(config, 'MIN_RISK_REWARD_RATIO', 1.2)
            new_rr = min(2.0, current_rr + 0.1)
            recommendations.append({
                'setting': 'MIN_RISK_REWARD_RATIO',
                'current': current_rr,
                'suggested': new_rr,
                'reason': f'High loss rate ({losses} losses vs {wins} wins). Increasing minimum R/R ratio to {new_rr:.2f}.'
            })
        
        # Recommendation 4: Adjust symbol filter based on worst performers
        if len(worst_coins) >= 3:
            worst_win_rate = worst_coins[0][1] if worst_coins else 0.0
            if worst_win_rate < 0.25:
                current_min_wr = getattr(config, 'SYMBOL_MIN_WIN_RATE', 0.45)
                new_min_wr = min(0.60, current_min_wr + 0.05)
                recommendations.append({
                    'setting': 'SYMBOL_MIN_WIN_RATE',
                    'current': current_min_wr,
                    'suggested': new_min_wr,
                    'reason': f'Many poor performing coins detected. Increasing symbol filter to {new_min_wr:.2f} win rate.'
                })
        
        # Recommendation 5: Adjust quality score if too many low-quality signals
        if timeouts > (wins + losses) * 0.5:  # More than 50% timeouts
            current_quality = getattr(config, 'MIN_QUALITY_SCORE', 0.5)
            new_quality = min(0.7, current_quality + 0.1)
            recommendations.append({
                'setting': 'MIN_QUALITY_SCORE',
                'current': current_quality,
                'suggested': new_quality,
                'reason': f'High timeout rate ({timeouts} timeouts). Increasing minimum quality score to {new_quality:.2f}.'
            })
        
        # Recommendation 6: Learn from loss analyzer
        try:
            from loss_analyzer import LossAnalyzer
            loss_analyzer = LossAnalyzer()
            failure_stats = loss_analyzer.get_failure_statistics()
            prevention_filters = loss_analyzer.generate_prevention_filters()
            
            # Apply prevention filters
            for filter_name, filter_data in prevention_filters.items():
                if filter_name == 'min_confidence' and not any(r.get('setting') == 'MIN_CONFIDENCE_SCORE' for r in recommendations):
                    recommendations.append({
                        'setting': 'MIN_CONFIDENCE_SCORE',
                        'current': filter_data['current'],
                        'suggested': filter_data['suggested'],
                        'reason': f"Loss analysis: {filter_data['reason']}"
                    })
                elif filter_name == 'min_volume_spike' and not any(r.get('setting') == 'MIN_VOLUME_SPIKE_FOR_SIGNAL' for r in recommendations):
                    recommendations.append({
                        'setting': 'MIN_VOLUME_SPIKE_FOR_SIGNAL',
                        'current': filter_data['current'],
                        'suggested': filter_data['suggested'],
                        'reason': f"Loss analysis: {filter_data['reason']}"
                    })
                elif filter_name == 'min_price_move' and not any(r.get('setting') == 'MIN_PRICE_MOVE_FOR_SIGNAL' for r in recommendations):
                    recommendations.append({
                        'setting': 'MIN_PRICE_MOVE_FOR_SIGNAL',
                        'current': filter_data['current'],
                        'suggested': filter_data['suggested'],
                        'reason': f"Loss analysis: {filter_data['reason']}"
                    })
            
            # Add insights from loss analysis
            if failure_stats.get('top_lessons'):
                insights = failure_stats['top_lessons']
                if insights:
                    print(f"\n📚 Top lessons from losses:")
                    for lesson in insights[:5]:
                        print(f"   - {lesson}")
        except Exception as e:
            print(f"⚠️  Error in loss analysis: {e}")
        
        return recommendations
    
    async def apply_recommendations(self, recommendations: List[Dict], auto_apply: bool = False) -> Dict:
        """
        Apply AI recommendations to config.
        If auto_apply=False, returns what would be changed.
        If auto_apply=True, actually modifies config.py
        """
        changes = []
        
        for rec in recommendations:
            setting = rec.get('setting')
            suggested = rec.get('suggested')
            current = rec.get('current')
            reason = rec.get('reason', '')
            
            if setting and suggested is not None:
                if auto_apply:
                    # Actually modify config
                    if hasattr(config, setting):
                        old_value = getattr(config, setting)
                        setattr(config, setting, suggested)
                        changes.append({
                            'setting': setting,
                            'old': old_value,
                            'new': suggested,
                            'reason': reason,
                            'applied': True
                        })
                        print(f"✅ Applied: {setting} = {old_value} → {suggested} ({reason})")
                else:
                    # Just report what would change
                    changes.append({
                        'setting': setting,
                        'old': current,
                        'new': suggested,
                        'reason': reason,
                        'applied': False
                    })
        
        return {
            'changes': changes,
            'auto_applied': auto_apply,
            'timestamp': datetime.now()
        }
    
    async def run_improvement_cycle(self, auto_apply: bool = False) -> Dict:
        """
        Complete improvement cycle: analyze → recommend → (optionally) apply
        Enhanced with advanced learning features
        """
        print("\n" + "="*60)
        print("🤖 ADVANCED AI SELF-IMPROVEMENT CYCLE")
        print("="*60)
        
        # Step 1: Analyze performance
        print("\n📊 Step 1: Analyzing performance...")
        analysis = await self.analyze_performance(days=7)
        
        # Step 2: Advanced pattern analysis
        print("\n🔍 Step 2: Advanced pattern analysis...")
        pattern_insights = await self._analyze_patterns()
        
        # Step 3: Strategy performance analysis
        print("\n📈 Step 3: Strategy performance analysis...")
        strategy_insights = await self._analyze_strategy_performance()
        
        # Step 4: Correlation analysis
        print("\n🔗 Step 4: Correlation analysis...")
        correlation_insights = await self._analyze_correlations()
        
        # Step 5: Time-based learning
        print("\n⏰ Step 5: Time-based learning...")
        time_insights = await self._analyze_time_performance()
        
        # Get recommendations (enhanced with all insights)
        recommendations = analysis['recommendations']
        
        # Add advanced recommendations
        advanced_recs = self._generate_advanced_recommendations(
            pattern_insights, strategy_insights, correlation_insights, time_insights
        )
        recommendations.extend(advanced_recs)
        
        # Apply (or preview)
        result = await self.apply_recommendations(recommendations, auto_apply=auto_apply)
        
        # Save to history with all insights
        self.improvement_history.append({
            'analysis': analysis,
            'pattern_insights': pattern_insights,
            'strategy_insights': strategy_insights,
            'correlation_insights': correlation_insights,
            'time_insights': time_insights,
            'result': result,
            'timestamp': datetime.now()
        })
        
        print("\n✅ Self-improvement cycle complete!")
        
        return {
            'analysis': analysis,
            'pattern_insights': pattern_insights,
            'strategy_insights': strategy_insights,
            'correlation_insights': correlation_insights,
            'time_insights': time_insights,
            'result': result
        }
    
    async def _analyze_patterns(self) -> Dict:
        """Analyze winning and losing patterns"""
        try:
            from pattern_learner import PatternLearner
            pattern_learner = PatternLearner()
            
            # Get pattern statistics
            winning_patterns = pattern_learner.winning_patterns
            losing_patterns = getattr(pattern_learner, 'losing_patterns', [])
            
            return {
                'winning_patterns_count': len(winning_patterns),
                'losing_patterns_count': len(losing_patterns),
                'top_patterns': winning_patterns[-10:] if winning_patterns else [],
                'insights': [
                    f"Learned {len(winning_patterns)} winning patterns",
                    f"Identified {len(losing_patterns)} losing patterns to avoid"
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_strategy_performance(self) -> Dict:
        """Analyze which strategies work best"""
        try:
            from strategy_optimizer import StrategyOptimizer
            from ai_analyzer import AIAnalyzer
            
            optimizer = StrategyOptimizer(AIAnalyzer())
            best_strategies = optimizer.best_strategies
            
            insights = []
            if best_strategies:
                top_strategy = best_strategies[0]
                insights.append(f"Best strategy: {top_strategy.name} (score: {top_strategy.performance.get('score', 0):.2f})")
                insights.append(f"Win rate: {top_strategy.performance.get('win_rate', 0):.1%}")
            
            return {
                'best_strategies': [s.name for s in best_strategies[:5]] if best_strategies else [],
                'total_strategies_tested': len(optimizer.strategies),
                'insights': insights
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_correlations(self) -> Dict:
        """Analyze BTC/ETH correlations"""
        try:
            from coin_specific_learner import CoinSpecificLearner
            learner = CoinSpecificLearner()
            
            correlations = {}
            for symbol, profile in learner.coin_profiles.items():
                if profile.get('btc_correlation'):
                    corr_data = profile['btc_correlation']
                    correlations[symbol] = {
                        'correlation': corr_data.get('correlation_coefficient', 0),
                        'samples': len(corr_data.get('samples', []))
                    }
            
            return {
                'coins_with_correlation_data': len(correlations),
                'correlations': correlations,
                'insights': [
                    f"Tracking BTC correlation for {len(correlations)} coins",
                    "Small cap coins showing strong BTC correlation"
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    async def _analyze_time_performance(self) -> Dict:
        """Analyze time-based performance"""
        try:
            from strategy_optimizer import StrategyOptimizer
            from ai_analyzer import AIAnalyzer
            
            optimizer = StrategyOptimizer(AIAnalyzer())
            time_perf = optimizer.time_performance
            
            best_hours = []
            if time_perf:
                sorted_hours = sorted(time_perf.items(), 
                                    key=lambda x: x[1].get('wins', 0) / max(x[1].get('total', 1), 1),
                                    reverse=True)
                best_hours = [int(h) for h, _ in sorted_hours[:5]]
            
            return {
                'best_hours': best_hours,
                'time_performance_data': len(time_perf),
                'insights': [
                    f"Best trading hours: {best_hours}",
                    f"Tracking performance for {len(time_perf)} hours"
                ]
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_advanced_recommendations(self, pattern_insights: Dict, strategy_insights: Dict,
                                          correlation_insights: Dict, time_insights: Dict) -> List[Dict]:
        """Generate advanced recommendations based on all insights"""
        recommendations = []
        
        # Recommendation from pattern analysis
        if pattern_insights.get('winning_patterns_count', 0) > 20:
            recommendations.append({
                'setting': 'PATTERN_BOOST_ENABLED',
                'current': getattr(config, 'PATTERN_BOOST_ENABLED', True),
                'suggested': True,
                'reason': f"Found {pattern_insights['winning_patterns_count']} winning patterns - pattern boosting should be enabled"
            })
        
        # Recommendation from strategy analysis
        if strategy_insights.get('best_strategies'):
            best_strategy_name = strategy_insights['best_strategies'][0] if strategy_insights['best_strategies'] else None
            if best_strategy_name and 'btc_correlation' in best_strategy_name.lower():
                recommendations.append({
                    'setting': 'ENABLE_BTC_CORRELATION',
                    'current': getattr(config, 'ENABLE_BTC_CORRELATION', True),
                    'suggested': True,
                    'reason': f"Best strategy uses BTC correlation: {best_strategy_name}"
                })
        
        # Recommendation from correlation analysis
        if correlation_insights.get('coins_with_correlation_data', 0) > 10:
            recommendations.append({
                'setting': 'USE_BTC_CORRELATION_FILTER',
                'current': getattr(config, 'USE_BTC_CORRELATION_FILTER', False),
                'suggested': True,
                'reason': f"Strong BTC correlation detected for {correlation_insights['coins_with_correlation_data']} coins"
            })
        
        # Recommendation from time analysis
        if time_insights.get('best_hours'):
            recommendations.append({
                'setting': 'PREFERRED_TRADING_HOURS',
                'current': None,
                'suggested': time_insights['best_hours'],
                'reason': f"Best performing hours identified: {time_insights['best_hours']}"
            })
        
        return recommendations

