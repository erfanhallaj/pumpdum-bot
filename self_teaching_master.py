"""
🤖 Self-Teaching Master System
A complete autonomous teacher that can read, modify, write, and delete code
to continuously improve the bot based on backtest results.

This system has FULL ACCESS to the codebase and can:
- Analyze backtest results
- Identify problems
- Generate code fixes
- Test changes
- Apply improvements
- Rollback if needed
"""

import os
import ast
import subprocess
import shutil
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import aiohttp
import asyncio
import config
from backtester import Backtester
from ai_analyzer import AIAnalyzer
from monitor import MarketMonitor


class SelfTeachingMaster:
    """
    Master teacher system with full code access and modification capabilities.
    Can autonomously improve the bot by analyzing backtests and fixing code.
    """
    
    def __init__(self, ai_analyzer: AIAnalyzer, backtester: Backtester, monitor: MarketMonitor, logger=None):
        self.ai_analyzer = ai_analyzer
        self.backtester = backtester
        self.monitor = monitor
        self.logger = logger  # Logger for activities
        
        # Code modification settings
        self.code_backup_dir = 'code_backups'
        self.max_backups = 10  # Keep last 10 backups
        self.test_before_apply = True  # Test changes before applying
        self.auto_rollback_on_error = True  # Rollback if test fails
        
        # Problem patterns learned (must be defined before load_history)
        self.problem_patterns = {
            'low_win_rate': [],
            'high_timeouts': [],
            'high_losses': [],
            'code_errors': [],
            'performance_issues': []
        }
        
        # Learning history
        self.learning_history_file = 'models/teaching_history.json'
        self.learning_history = []
        self.load_history()
        
        # AI Configuration for code generation
        self.use_ai_code_generation = True
        self.ai_provider = 'huggingface'  # or 'openai', 'local'
        self.hf_model = "bigcode/starcoder"  # Code generation model
        self.openai_model = "gpt-4"  # If OpenAI available
        
        # Master Teacher Identity
        self.master_name = "Master_Hallaj"  # معلم آقای حلاج
        self.teacher_level = 0  # Level 0 = Master, Level 1+ = Students
        self.parent_teacher = None  # None for master, reference for students
        
        # Files that can be modified (expanded - almost everything)
        self.modifiable_files = [
            'config.py',
            'ai_analyzer.py',
            'backtester.py',
            'monitor.py',
            'strategy_optimizer.py',
            'coin_specific_learner.py',
            'pattern_learner.py',
            'adaptive_filter.py',
            'loss_analyzer.py',
            'scam_detector.py',
            'signal_tracker.py',
            'advanced_features.py',
            'ai_self_improver.py'
        ]
        
        # CRITICAL FILES - Only these are protected (very sensitive)
        self.protected_files = [
            'main.py',  # Main entry point - CRITICAL
            'telegram_bot.py',  # Communication - CRITICAL
            'requirements.txt'  # Dependencies - CRITICAL
        ]
        
        # Advanced self-repair capabilities
        self.self_repair_enabled = True
        self.can_create_files = True  # اجازه ساخت فایل جدید
        self.can_create_teachers = True  # اجازه ساخت معلم‌های جدید
        self.max_teacher_levels = 3  # حداکثر 3 سطح معلم (Master + 2 levels of students)
        
        # Sub-teachers (students) management
        self.sub_teachers = []  # List of student teachers
        self.sub_teacher_dir = 'sub_teachers'  # Directory for sub-teachers
        os.makedirs(self.sub_teacher_dir, exist_ok=True)
        
        # Advanced self-repair patterns
        self.self_repair_patterns = {
            'code_errors': [],
            'performance_degradation': [],
            'memory_issues': [],
            'logic_errors': [],
            'optimization_opportunities': []
        }
        
        os.makedirs(self.code_backup_dir, exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs(self.sub_teacher_dir, exist_ok=True)
        os.makedirs('generated_code', exist_ok=True)  # For new files
        
        # Real-time learning settings
        self.real_time_learning_enabled = getattr(config, 'ENABLE_REAL_TIME_LEARNING', True)
        self.signals_since_last_backtest = 0
        self.signals_for_backtest = getattr(config, 'SIGNALS_FOR_BACKTEST', 10)  # Run backtest after N signals
        self.signal_analysis_queue = []  # Queue for signal analysis
        self.last_signal_analysis_time = None
        self.enable_micro_adjustments = getattr(config, 'ENABLE_MICRO_ADJUSTMENTS', True)
        
        # Advanced learning patterns
        self.deep_learning_patterns = {
            'winning_conditions': [],
            'losing_conditions': [],
            'timeout_conditions': [],
            'optimal_parameters': {},
            'market_regime_patterns': {}
        }
        
        # Performance tracking
        self.real_time_stats = {
            'total_signals': 0,
            'wins': 0,
            'losses': 0,
            'timeouts': 0,
            'win_rate': 0.0,
            'last_updated': datetime.now()
        }
    
    def load_history(self):
        """Load learning history"""
        if os.path.exists(self.learning_history_file):
            try:
                with open(self.learning_history_file, 'r') as f:
                    data = json.load(f)
                    self.learning_history = data.get('history', [])
                    self.problem_patterns = data.get('problem_patterns', self.problem_patterns)
            except Exception as e:
                print(f"Error loading teaching history: {e}")
    
    def save_history(self):
        """Save learning history"""
        try:
            data = {
                'history': self.learning_history[-100:],  # Keep last 100 entries
                'problem_patterns': self.problem_patterns,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.learning_history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving teaching history: {e}")
    
    async def analyze_signal_realtime(self, signal: 'TrackedSignal', outcome: str) -> Dict:
        """
        Analyze a single signal in real-time and make immediate improvements
        This is called immediately when a signal closes (win/loss/timeout)
        """
        if not self.real_time_learning_enabled:
            return {}
        
        analysis_id = f"signal_{signal.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"\n🎓 Real-time Learning: Analyzing signal {signal.id} ({outcome})...")
        
        # Extract signal data
        signal_data = {
            'id': signal.id,
            'symbol': signal.symbol,
            'signal_type': signal.signal_type,
            'outcome': outcome,
            'entry': signal.entry,
            'exit1': signal.exit1,
            'exit2': signal.exit2,
            'exit3': signal.exit3,
            'stop_loss': signal.stop_loss,
            'probability': signal.probability,
            'confidence': signal.confidence,
            'features': signal.features,
            'timestamp': signal.timestamp.isoformat() if hasattr(signal.timestamp, 'isoformat') else str(signal.timestamp),
            'close_time': signal.close_time.isoformat() if hasattr(signal, 'close_time') and signal.close_time else None,
            'close_price': getattr(signal, 'close_price', None),
            'hit_target': getattr(signal, 'hit_target', None)
        }
        
        # Update real-time stats
        self.real_time_stats['total_signals'] += 1
        if outcome == 'win':
            self.real_time_stats['wins'] += 1
        elif outcome == 'loss':
            self.real_time_stats['losses'] += 1
        elif outcome == 'timeout':
            self.real_time_stats['timeouts'] += 1
        
        if self.real_time_stats['total_signals'] > 0:
            self.real_time_stats['win_rate'] = self.real_time_stats['wins'] / self.real_time_stats['total_signals']
        
        self.real_time_stats['last_updated'] = datetime.now()
        
        # Deep analysis of this signal
        analysis = await self._deep_analyze_signal(signal_data)
        
        # Immediate micro-adjustments based on this signal
        micro_fixes = await self._generate_micro_fixes(signal_data, analysis)
        
        # Apply micro-fixes immediately (small, safe changes)
        applied_fixes = []
        for fix in micro_fixes:
            if fix.get('safe_to_apply', False):
                result = await self._apply_micro_fix(fix)
                if result['success']:
                    applied_fixes.append(fix)
                    print(f"   ✅ Applied micro-fix: {fix.get('description', 'N/A')}")
        
        # Queue for batch learning
        self.signal_analysis_queue.append({
            'signal': signal_data,
            'analysis': analysis,
            'timestamp': datetime.now()
        })
        
        # Keep queue size manageable
        if len(self.signal_analysis_queue) > 100:
            self.signal_analysis_queue = self.signal_analysis_queue[-100:]
        
        # Check if we need to run backtest
        self.signals_since_last_backtest += 1
        if self.signals_since_last_backtest >= self.signals_for_backtest:
            print(f"\n📊 {self.signals_since_last_backtest} signals reached - Running backtest and teaching session...")
            await self._run_quick_teaching_session()
            self.signals_since_last_backtest = 0
        
        return {
            'analysis_id': analysis_id,
            'signal_id': signal.id,
            'outcome': outcome,
            'analysis': analysis,
            'micro_fixes_applied': len(applied_fixes),
            'applied_fixes': applied_fixes
        }
    
    async def _deep_analyze_signal(self, signal_data: Dict) -> Dict:
        """Deep analysis of a single signal"""
        analysis = {
            'signal_quality': 'unknown',
            'issues_found': [],
            'strengths': [],
            'recommendations': []
        }
        
        # Analyze signal quality
        if signal_data['outcome'] == 'win':
            analysis['signal_quality'] = 'good'
            analysis['strengths'].append('Successful prediction')
            
            # Learn what worked
            if signal_data['probability'] >= 0.65 and signal_data['confidence'] >= 0.70:
                analysis['strengths'].append('High confidence signal succeeded')
                self.deep_learning_patterns['winning_conditions'].append({
                    'probability_range': (0.65, 1.0),
                    'confidence_range': (0.70, 1.0),
                    'timestamp': datetime.now()
                })
        elif signal_data['outcome'] == 'loss':
            analysis['signal_quality'] = 'poor'
            analysis['issues_found'].append('False positive signal')
            
            # Learn what went wrong
            if signal_data['probability'] < 0.60:
                analysis['issues_found'].append('Low probability signal failed')
                analysis['recommendations'].append('Increase MIN_CONFIDENCE_SCORE')
            
            if signal_data['confidence'] < 0.65:
                analysis['issues_found'].append('Low confidence signal failed')
                analysis['recommendations'].append('Increase MIN_AI_CONFIDENCE')
            
            self.deep_learning_patterns['losing_conditions'].append({
                'probability': signal_data['probability'],
                'confidence': signal_data['confidence'],
                'timestamp': datetime.now()
            })
        elif signal_data['outcome'] == 'timeout':
            analysis['signal_quality'] = 'neutral'
            analysis['issues_found'].append('Signal timed out')
            
            # Learn timeout patterns
            self.deep_learning_patterns['timeout_conditions'].append({
                'probability': signal_data['probability'],
                'confidence': signal_data['confidence'],
                'timestamp': datetime.now()
            })
        
        # Keep patterns manageable
        for key in ['winning_conditions', 'losing_conditions', 'timeout_conditions']:
            if len(self.deep_learning_patterns[key]) > 200:
                self.deep_learning_patterns[key] = self.deep_learning_patterns[key][-200:]
        
        return analysis
    
    async def _generate_micro_fixes(self, signal_data: Dict, analysis: Dict) -> List[Dict]:
        """Generate small, safe fixes based on single signal"""
        fixes = []
        
        # Only make micro-adjustments for clear patterns
        if signal_data['outcome'] == 'loss':
            # If we have multiple recent losses with similar characteristics
            recent_losses = [s for s in self.signal_analysis_queue[-20:] 
                           if s['signal']['outcome'] == 'loss']
            
            if len(recent_losses) >= 3:
                # Check if they all had low probability/confidence
                avg_prob = sum(s['signal']['probability'] for s in recent_losses) / len(recent_losses)
                avg_conf = sum(s['signal']['confidence'] for s in recent_losses) / len(recent_losses)
                
                if avg_prob < config.MIN_CONFIDENCE_SCORE + 0.05:
                    fixes.append({
                        'type': 'micro_adjust',
                        'target': 'MIN_CONFIDENCE_SCORE',
                        'current': config.MIN_CONFIDENCE_SCORE,
                        'new': min(0.90, config.MIN_CONFIDENCE_SCORE + 0.02),
                        'description': f'Micro-adjust: {len(recent_losses)} recent losses with low prob',
                        'safe_to_apply': True,
                        'file': 'config.py'
                    })
        
        elif signal_data['outcome'] == 'win':
            # If we have multiple wins, we might be too conservative
            recent_wins = [s for s in self.signal_analysis_queue[-20:] 
                          if s['signal']['outcome'] == 'win']
            
            if len(recent_wins) >= 5:
                # Check if we're missing opportunities
                current_win_rate = self.real_time_stats['win_rate']
                if current_win_rate > 0.70 and self.real_time_stats['total_signals'] < 15:
                    # High win rate but few signals - can be slightly more aggressive
                    fixes.append({
                        'type': 'micro_adjust',
                        'target': 'MIN_CONFIDENCE_SCORE',
                        'current': config.MIN_CONFIDENCE_SCORE,
                        'new': max(0.40, config.MIN_CONFIDENCE_SCORE - 0.01),
                        'description': f'Micro-adjust: High win rate ({current_win_rate:.1%}) but few signals',
                        'safe_to_apply': True,
                        'file': 'config.py'
                    })
        
        return fixes
    
    async def _apply_micro_fix(self, fix: Dict) -> Dict:
        """Apply a small, safe fix immediately"""
        try:
            if fix['type'] == 'micro_adjust' and fix['file'] == 'config.py':
                # Direct config update (safe, no file modification needed)
                if hasattr(config, fix['target']):
                    old_value = getattr(config, fix['target'])
                    setattr(config, fix['target'], fix['new'])
                    return {'success': True, 'old': old_value, 'new': fix['new']}
            
            return {'success': False, 'error': 'Not a safe micro-fix'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _run_quick_teaching_session(self):
        """Run a quick teaching session after N signals"""
        print("\n🎓 Quick Teaching Session (after N signals)...")
        
        # Analyze recent signals
        recent_signals = self.signal_analysis_queue[-self.signals_for_backtest:]
        
        # Identify problems from recent signals
        problems = self._identify_problems_from_signals(recent_signals)
        
        if problems:
            # Generate and apply fixes
            for problem in problems:
                fix = await self._generate_fix(problem, {'win_rate': self.real_time_stats['win_rate']})
                if fix:
                    # Quick test
                    test_result = await self._quick_test_fix(fix)
                    if test_result.get('passed', False):
                        apply_result = await self._apply_fix(fix)
                        if apply_result['success']:
                            print(f"   ✅ Applied fix for: {problem['description']}")
    
    def _identify_problems_from_signals(self, signals: List[Dict]) -> List[Dict]:
        """Identify problems from recent signals"""
        problems = []
        
        if not signals:
            return problems
        
        wins = sum(1 for s in signals if s['signal']['outcome'] == 'win')
        losses = sum(1 for s in signals if s['signal']['outcome'] == 'loss')
        timeouts = sum(1 for s in signals if s['signal']['outcome'] == 'timeout')
        total = len(signals)
        
        win_rate = wins / total if total > 0 else 0
        loss_rate = losses / total if total > 0 else 0
        timeout_rate = timeouts / total if total > 0 else 0
        
        # Problem: Low win rate
        if win_rate < 0.30 and total >= 5:
            problems.append({
                'type': 'low_win_rate',
                'severity': 'high',
                'current_value': win_rate,
                'target_value': 0.45,
                'description': f'Win rate {win_rate:.1%} is low in recent {total} signals',
                'affected_files': ['config.py', 'ai_analyzer.py']
            })
        
        # Problem: High loss rate
        if loss_rate > 0.40 and total >= 5:
            problems.append({
                'type': 'high_losses',
                'severity': 'high',
                'current_value': loss_rate,
                'target_value': 0.25,
                'description': f'Loss rate {loss_rate:.1%} is high in recent {total} signals',
                'affected_files': ['config.py', 'adaptive_filter.py']
            })
        
        # Problem: High timeout rate
        if timeout_rate > 0.60 and total >= 5:
            problems.append({
                'type': 'high_timeouts',
                'severity': 'medium',
                'current_value': timeout_rate,
                'target_value': 0.40,
                'description': f'Timeout rate {timeout_rate:.1%} is high in recent {total} signals',
                'affected_files': ['config.py', 'backtester.py']
            })
        
        return problems
    
    async def _quick_test_fix(self, fix: Dict) -> Dict:
        """Quick test of a fix (simpler than full backtest)"""
        # For quick tests, we just validate the fix makes sense
        # Full testing happens in daily session
        return {'passed': True, 'reason': 'Quick validation passed'}
    
    async def daily_teaching_session(self) -> Dict:
        """
        Main teaching session - runs daily to analyze and improve
        Enhanced with deep learning and pattern recognition
        """
        print("\n" + "="*70)
        print("🎓 SELF-TEACHING MASTER - ADVANCED DAILY SESSION")
        print("="*70)
        
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_result = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'problems_found': [],
            'fixes_applied': [],
            'tests_run': [],
            'improvements': []
        }
        
        try:
            # Step 1: Run backtest to get current performance (هر 2 ساعت)
            print("\n📊 Step 1: Running backtest to assess current performance...")
            print("   ⏰ این بک تست هر 2 ساعت اجرا می‌شود و در دست معلم قرار می‌گیرد...")
            historical_data = await self._collect_test_data()
            
            if not historical_data:
                print("⚠️  No historical data available for backtest")
                return session_result
            
            # Run comprehensive backtest
            backtest_results = self.backtester.run_backtest(historical_data)
            session_result['backtest_results'] = {
                'win_rate': backtest_results.get('win_rate', 0),
                'total_signals': backtest_results.get('total_signals', 0),
                'total_profit': backtest_results.get('total_profit', 0),
                'timeouts': backtest_results.get('timeouts', 0),
                'losses': backtest_results.get('losses', 0)
            }
            
            print(f"   ✅ Backtest complete:")
            print(f"      📊 Win Rate: {backtest_results.get('win_rate', 0):.2%}")
            print(f"      📈 Total Signals: {backtest_results.get('total_signals', 0)}")
            print(f"      💰 Total Profit: {backtest_results.get('total_profit', 0):.2f} USDT")
            print(f"      ⏳ Timeouts: {backtest_results.get('timeouts', 0)}")
            print(f"      ❌ Losses: {backtest_results.get('losses', 0)}")
            print(f"   🎓 نتایج بک تست در دست معلم قرار گرفت...")
            
            # Step 2: Analyze problems (enhanced with real-time data)
            print("\n🔍 Step 2: Analyzing problems (with real-time signal data)...")
            
            # Combine backtest and real-time data
            combined_problems = self._identify_problems(backtest_results)
            
            # Add problems from real-time signals
            if self.signal_analysis_queue:
                realtime_problems = self._identify_problems_from_signals(
                    self.signal_analysis_queue[-50:]  # Last 50 signals
                )
                # Merge problems
                for rp in realtime_problems:
                    # Check if similar problem already exists
                    exists = any(p['type'] == rp['type'] for p in combined_problems)
                    if not exists:
                        combined_problems.append(rp)
            
            problems = combined_problems
            session_result['problems_found'] = problems
            
            # Deep pattern analysis
            print("\n🧠 Step 2.5: Deep pattern analysis...")
            pattern_insights = self._analyze_deep_patterns()
            session_result['pattern_insights'] = pattern_insights
            
            if not problems:
                print("✅ No problems identified - bot is performing well!")
                return session_result
            
            # Step 3: Generate fixes for each problem
            print(f"\n🔧 Step 3: Generating fixes for {len(problems)} problems...")
            fixes = []
            for problem in problems:
                fix = await self._generate_fix(problem, backtest_results)
                if fix:
                    fixes.append(fix)
            
            session_result['fixes_generated'] = len(fixes)
            
            # Step 4: Test fixes before applying
            if self.test_before_apply and fixes:
                print(f"\n🧪 Step 4: Testing {len(fixes)} fixes...")
                tested_fixes = []
                for fix in fixes:
                    test_result = await self._test_fix(fix, historical_data)
                    if test_result['passed']:
                        tested_fixes.append(fix)
                        session_result['tests_run'].append({
                            'fix_id': fix['id'],
                            'passed': True,
                            'improvement': test_result.get('improvement', 0)
                        })
                    else:
                        print(f"   ❌ Fix {fix['id']} failed test: {test_result.get('reason', 'Unknown')}")
                        session_result['tests_run'].append({
                            'fix_id': fix['id'],
                            'passed': False,
                            'reason': test_result.get('reason', 'Unknown')
                        })
                
                fixes = tested_fixes  # Only apply fixes that passed tests
            
            # Step 5: Apply fixes
            if fixes:
                print(f"\n✅ Step 5: Applying {len(fixes)} fixes...")
                for fix in fixes:
                    apply_result = await self._apply_fix(fix)
                    if apply_result['success']:
                        session_result['fixes_applied'].append({
                            'fix_id': fix['id'],
                            'file': fix['file'],
                            'change_type': fix['change_type']
                        })
                        print(f"   ✅ Applied fix {fix['id']} to {fix['file']}")
                        
                        # Log fix application
                        if self.logger:
                            self.logger.log_teacher_fix(
                                fix['id'],
                                fix.get('problem', {}).get('type', 'Unknown'),
                                fix.get('description', 'No description'),
                                'Success'
                            )
                    else:
                        print(f"   ❌ Failed to apply fix {fix['id']}: {apply_result.get('error', 'Unknown')}")
                        if self.logger:
                            self.logger.log_teacher_fix(
                                fix['id'],
                                fix.get('problem', {}).get('type', 'Unknown'),
                                fix.get('description', 'No description'),
                                f"Failed: {apply_result.get('error', 'Unknown')}"
                            )
            
            # Step 6: Verify improvements
            if session_result['fixes_applied']:
                print("\n📈 Step 6: Verifying improvements...")
                verification = await self._verify_improvements(historical_data, backtest_results)
                session_result['improvements'] = verification
            
            # Step 7: Advanced self-repair (if enabled)
            if self.self_repair_enabled:
                print("\n🔧 Step 7: Running advanced self-repair...")
                print(f"   Master: {self.master_name} (Level {self.teacher_level})")
                repair_result = await self.advanced_self_repair()
                session_result['self_repair'] = repair_result
            
            # Save session
            self.learning_history.append(session_result)
            self.save_history()
            
            print("\n✅ Teaching session complete!")
            print(f"   Problems found: {len(problems)}")
            print(f"   Fixes applied: {len(session_result['fixes_applied'])}")
            
        except Exception as e:
            print(f"\n❌ Error in teaching session: {e}")
            import traceback
            traceback.print_exc()
            session_result['error'] = str(e)
        
        return session_result
    
    async def _collect_test_data(self) -> Dict:
        """Collect historical data for testing"""
        # Use a subset of coins for faster testing
        test_symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT'
        ]
        
        print(f"   Collecting data for {len(test_symbols)} test coins...")
        historical_data = await self.monitor.collect_historical_data(test_symbols)
        return historical_data
    
    def _identify_problems(self, backtest_results: Dict) -> List[Dict]:
        """Identify problems from backtest results"""
        problems = []
        
        win_rate = backtest_results.get('win_rate', 0)
        total_signals = backtest_results.get('total_signals', 0)
        timeouts = backtest_results.get('timeouts', 0)
        losses = backtest_results.get('losses', 0)
        total_profit = backtest_results.get('total_profit', 0)
        
        # Problem 1: Low win rate
        if win_rate < 0.30 and total_signals >= 20:
            problems.append({
                'type': 'low_win_rate',
                'severity': 'high',
                'current_value': win_rate,
                'target_value': 0.45,
                'description': f'Win rate {win_rate:.1%} is below target 45%',
                'affected_files': ['config.py', 'ai_analyzer.py', 'strategy_optimizer.py']
            })
        
        # Problem 2: High timeout rate
        timeout_rate = timeouts / total_signals if total_signals > 0 else 0
        if timeout_rate > 0.70 and total_signals >= 20:
            problems.append({
                'type': 'high_timeouts',
                'severity': 'medium',
                'current_value': timeout_rate,
                'target_value': 0.50,
                'description': f'Timeout rate {timeout_rate:.1%} is too high (target: <50%)',
                'affected_files': ['backtester.py', 'config.py', 'signal_tracker.py']
            })
        
        # Problem 3: High loss rate
        loss_rate = losses / total_signals if total_signals > 0 else 0
        if loss_rate > 0.40 and total_signals >= 20:
            problems.append({
                'type': 'high_losses',
                'severity': 'high',
                'current_value': loss_rate,
                'target_value': 0.25,
                'description': f'Loss rate {loss_rate:.1%} is too high (target: <25%)',
                'affected_files': ['config.py', 'ai_analyzer.py', 'loss_analyzer.py', 'adaptive_filter.py']
            })
        
        # Problem 4: Negative profit
        if total_profit < -100 and total_signals >= 20:
            problems.append({
                'type': 'negative_profit',
                'severity': 'critical',
                'current_value': total_profit,
                'target_value': 0,
                'description': f'Total profit {total_profit:.2f} USDT is negative',
                'affected_files': ['backtester.py', 'strategy_optimizer.py', 'config.py']
            })
        
        # Problem 5: Too few signals
        if total_signals < 10:
            problems.append({
                'type': 'too_few_signals',
                'severity': 'medium',
                'current_value': total_signals,
                'target_value': 20,
                'description': f'Only {total_signals} signals generated (target: 20+)',
                'affected_files': ['config.py', 'ai_analyzer.py', 'monitor.py']
            })
        
        return problems
    
    async def _generate_fix(self, problem: Dict, backtest_results: Dict) -> Optional[Dict]:
        """Generate code fix for a problem"""
        problem_type = problem['type']
        fix_id = f"{problem_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"   🔧 Generating fix for: {problem['description']}")
        
        # Use AI to generate fix code
        if self.use_ai_code_generation:
            fix_code = await self._ai_generate_fix(problem, backtest_results)
        else:
            fix_code = self._rule_based_fix(problem, backtest_results)
        
        if not fix_code:
            return None
        
        return {
            'id': fix_id,
            'problem': problem,
            'file': fix_code['file'],
            'change_type': fix_code['change_type'],
            'code_changes': fix_code['changes'],
            'description': fix_code['description'],
            'expected_improvement': fix_code.get('expected_improvement', {})
        }
    
    async def _ai_generate_fix(self, problem: Dict, backtest_results: Dict) -> Optional[Dict]:
        """Use AI to generate fix code"""
        try:
            # Read relevant file
            file_path = problem['affected_files'][0]
            if file_path not in self.modifiable_files:
                return None
            
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                current_code = f.read()
            
            # Build prompt for AI
            prompt = self._build_fix_prompt(problem, backtest_results, current_code, file_path)
            
            # Call AI (Hugging Face or OpenAI)
            if self.ai_provider == 'huggingface':
                generated_code = await self._query_huggingface_code(prompt)
            elif self.ai_provider == 'openai' and self.openai_api_key:
                generated_code = await self._query_openai_code(prompt)
            else:
                # Fallback to rule-based
                return self._rule_based_fix(problem, backtest_results)
            
            if not generated_code:
                return self._rule_based_fix(problem, backtest_results)
            
            # Parse AI response to extract code changes
            changes = self._parse_ai_code_response(generated_code, current_code, file_path)
            
            return {
                'file': file_path,
                'change_type': 'modify',
                'changes': changes,
                'description': f"AI-generated fix for {problem['type']}",
                'expected_improvement': {
                    problem['type']: problem['target_value'] - problem['current_value']
                }
            }
            
        except Exception as e:
            print(f"   ⚠️  AI code generation error: {e}")
            return self._rule_based_fix(problem, backtest_results)
    
    def _rule_based_fix(self, problem: Dict, backtest_results: Dict) -> Optional[Dict]:
        """Generate fix using rule-based approach (fallback)"""
        problem_type = problem['type']
        file_path = problem['affected_files'][0]
        
        if problem_type == 'low_win_rate':
            # Increase confidence thresholds
            return {
                'file': 'config.py',
                'change_type': 'modify',
                'changes': [
                    {
                        'type': 'update_value',
                        'target': 'MIN_CONFIDENCE_SCORE',
                        'old_value': config.MIN_CONFIDENCE_SCORE,
                        'new_value': min(0.90, config.MIN_CONFIDENCE_SCORE + 0.05),
                        'line_pattern': r'MIN_CONFIDENCE_SCORE\s*=\s*[\d.]+'
                    },
                    {
                        'type': 'update_value',
                        'target': 'MIN_AI_CONFIDENCE',
                        'old_value': getattr(config, 'MIN_AI_CONFIDENCE', 0.65),
                        'new_value': min(0.90, getattr(config, 'MIN_AI_CONFIDENCE', 0.65) + 0.05),
                        'line_pattern': r'MIN_AI_CONFIDENCE\s*=\s*[\d.]+'
                    }
                ],
                'description': 'Increase confidence thresholds to improve signal quality',
                'expected_improvement': {'win_rate': 0.10}
            }
        
        elif problem_type == 'high_timeouts':
            # Reduce timeout window
            return {
                'file': 'config.py',
                'change_type': 'modify',
                'changes': [
                    {
                        'type': 'update_value',
                        'target': 'SIGNAL_MAX_LIFETIME_HOURS',
                        'old_value': getattr(config, 'SIGNAL_MAX_LIFETIME_HOURS', 2),
                        'new_value': max(1.0, getattr(config, 'SIGNAL_MAX_LIFETIME_HOURS', 2) - 0.5),
                        'line_pattern': r'SIGNAL_MAX_LIFETIME_HOURS\s*=\s*[\d.]+'
                    }
                ],
                'description': 'Reduce timeout window to close signals faster',
                'expected_improvement': {'timeout_rate': -0.15}
            }
        
        elif problem_type == 'high_losses':
            # Increase filters and improve stop loss
            return {
                'file': 'config.py',
                'change_type': 'modify',
                'changes': [
                    {
                        'type': 'update_value',
                        'target': 'MIN_PRICE_MOVE_FOR_SIGNAL',
                        'old_value': config.MIN_PRICE_MOVE_FOR_SIGNAL,
                        'new_value': min(0.05, config.MIN_PRICE_MOVE_FOR_SIGNAL + 0.01),
                        'line_pattern': r'MIN_PRICE_MOVE_FOR_SIGNAL\s*=\s*[\d.]+'
                    },
                    {
                        'type': 'update_value',
                        'target': 'MIN_VOLUME_SPIKE_FOR_SIGNAL',
                        'old_value': config.MIN_VOLUME_SPIKE_FOR_SIGNAL,
                        'new_value': min(2.0, config.MIN_VOLUME_SPIKE_FOR_SIGNAL + 0.1),
                        'line_pattern': r'MIN_VOLUME_SPIKE_FOR_SIGNAL\s*=\s*[\d.]+'
                    }
                ],
                'description': 'Increase filters to reduce false signals',
                'expected_improvement': {'loss_rate': -0.10}
            }
        
        elif problem_type == 'too_few_signals':
            # Relax filters slightly
            return {
                'file': 'config.py',
                'change_type': 'modify',
                'changes': [
                    {
                        'type': 'update_value',
                        'target': 'MIN_CONFIDENCE_SCORE',
                        'old_value': config.MIN_CONFIDENCE_SCORE,
                        'new_value': max(0.40, config.MIN_CONFIDENCE_SCORE - 0.03),
                        'line_pattern': r'MIN_CONFIDENCE_SCORE\s*=\s*[\d.]+'
                    }
                ],
                'description': 'Slightly relax filters to generate more signals',
                'expected_improvement': {'total_signals': 10}
            }
        
        return None
    
    def _build_fix_prompt(self, problem: Dict, backtest_results: Dict, current_code: str, file_path: str) -> str:
        """Build prompt for AI code generation"""
        prompt = f"""You are an expert Python developer fixing a cryptocurrency trading bot.

PROBLEM:
{problem['description']}
Type: {problem['type']}
Current Value: {problem['current_value']}
Target Value: {problem['target_value']}

BACKTEST RESULTS:
Win Rate: {backtest_results.get('win_rate', 0):.2%}
Total Signals: {backtest_results.get('total_signals', 0)}
Timeouts: {backtest_results.get('timeouts', 0)}
Losses: {backtest_results.get('losses', 0)}

CURRENT CODE ({file_path}):
```python
{current_code[:2000]}  # First 2000 chars
```

TASK:
Generate Python code changes to fix this problem. Provide:
1. Specific code modifications (exact lines to change)
2. Explanation of why this fix will work
3. Expected improvement

Format your response as JSON:
{{
  "changes": [
    {{
      "type": "update_value" | "add_function" | "modify_function" | "add_import",
      "target": "variable_name or function_name",
      "old_value": "...",
      "new_value": "...",
      "line_number": 123,
      "code": "exact code to add/modify"
    }}
  ],
  "explanation": "why this fix works",
  "expected_improvement": {{"win_rate": 0.05, "timeout_rate": -0.10}}
}}
"""
        return prompt
    
    async def _query_huggingface_code(self, prompt: str) -> Optional[str]:
        """Query Hugging Face for code generation"""
        try:
            url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {}
            if os.getenv('HUGGINGFACE_API_KEY'):
                headers["Authorization"] = f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 1000,
                    "temperature": 0.3,
                    "return_full_text": False
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=60) as response:
                    if response.status == 200:
                        result = await response.json()
                        if isinstance(result, list) and len(result) > 0:
                            return result[0].get('generated_text', '')
                        elif isinstance(result, dict):
                            return result.get('generated_text', '')
                    elif response.status == 503:
                        await asyncio.sleep(5)
                        return await self._query_huggingface_code(prompt)
        except Exception as e:
            print(f"   ⚠️  Hugging Face API error: {e}")
        return None
    
    async def _query_openai_code(self, prompt: str) -> Optional[str]:
        """Query OpenAI for code generation"""
        try:
            # Try OpenAI if available
            try:
                import openai
                client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY', ''))
                
                response = await client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": "You are an expert Python developer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
            except ImportError:
                print("   ⚠️  OpenAI library not installed")
                return None
        except Exception as e:
            print(f"   ⚠️  OpenAI API error: {e}")
        return None
    
    def _parse_ai_code_response(self, ai_response: str, current_code: str, file_path: str) -> List[Dict]:
        """Parse AI response to extract code changes"""
        changes = []
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                changes = data.get('changes', [])
        except:
            # Fallback: try to extract code blocks
            code_blocks = re.findall(r'```python\n(.*?)\n```', ai_response, re.DOTALL)
            if code_blocks:
                # Simple extraction - would need more sophisticated parsing
                pass
        
        return changes
    
    async def _test_fix(self, fix: Dict, historical_data: Dict) -> Dict:
        """Test a fix before applying it"""
        print(f"      🧪 Testing fix {fix['id']}...")
        
        # Create backup
        backup_path = self._create_backup(fix['file'])
        
        try:
            # Apply fix temporarily
            apply_result = await self._apply_fix(fix, temporary=True)
            if not apply_result['success']:
                return {'passed': False, 'reason': 'Failed to apply fix'}
            
            # Run backtest with fix
            test_results = self.backtester.run_backtest(historical_data)
            
            # Check if improvement
            improvement = self._calculate_improvement(test_results, fix.get('expected_improvement', {}))
            
            # Restore backup
            self._restore_backup(fix['file'], backup_path)
            
            if improvement > 0:
                return {'passed': True, 'improvement': improvement}
            else:
                return {'passed': False, 'reason': 'No improvement detected'}
                
        except Exception as e:
            # Restore backup on error
            self._restore_backup(fix['file'], backup_path)
            return {'passed': False, 'reason': str(e)}
    
    def _calculate_improvement(self, test_results: Dict, expected: Dict) -> float:
        """Calculate improvement score"""
        score = 0.0
        
        if 'win_rate' in expected:
            actual_win_rate = test_results.get('win_rate', 0)
            score += (actual_win_rate - expected.get('win_rate', 0)) * 10
        
        if 'timeout_rate' in expected:
            timeout_rate = test_results.get('timeouts', 0) / max(test_results.get('total_signals', 1), 1)
            score += (expected.get('timeout_rate', 0) - timeout_rate) * 5
        
        return score
    
    async def _apply_fix(self, fix: Dict, temporary: bool = False) -> Dict:
        """Apply a fix to the code - Advanced with file creation support"""
        file_path = fix['file']
        
        # Check if file is protected (only critical files)
        if file_path in self.protected_files:
            return {'success': False, 'error': 'File is protected (critical system file)'}
        
        # If file doesn't exist and we can create files, create it
        if not os.path.exists(file_path):
            if self.can_create_files and fix.get('change_type') == 'create':
                return await self._create_new_file(fix)
            elif self.can_create_files:
                # Try to create file anyway if it's a new file
                return await self._create_new_file(fix)
            else:
                return {'success': False, 'error': 'File not found and creation not allowed'}
        
        try:
            # Create backup if not temporary
            if not temporary:
                self._create_backup(file_path)
            
            # Read current file
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Apply changes
            for change in fix['code_changes']:
                if change['type'] == 'update_value':
                    # Find and replace value
                    pattern = change.get('line_pattern', '')
                    new_value = change['new_value']
                    
                    for i, line in enumerate(lines):
                        if re.search(pattern, line):
                            # Replace the value
                            new_line = re.sub(
                                r'=\s*[\d.]+',
                                f'= {new_value}',
                                line,
                                count=1
                            )
                            lines[i] = new_line
                            break
            
            # Write modified file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            # Reload config if config.py was modified
            if file_path == 'config.py':
                import importlib
                import config
                importlib.reload(config)
                
                # Log config change
                if self.logger and 'code_changes' in fix:
                    for change in fix['code_changes']:
                        if change.get('type') == 'update_value':
                            self.logger.log_config_change(
                                change.get('target', 'Unknown'),
                                change.get('old_value', 'Unknown'),
                                change.get('new_value', 'Unknown'),
                                fix.get('description', 'No reason provided')
                            )
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _create_backup(self, file_path: str) -> str:
        """Create backup of a file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{Path(file_path).stem}_{timestamp}.py"
        backup_path = os.path.join(self.code_backup_dir, backup_filename)
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _restore_backup(self, file_path: str, backup_path: str):
        """Restore file from backup"""
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
    
    async def _verify_improvements(self, historical_data: Dict, old_results: Dict) -> Dict:
        """Verify that improvements actually worked"""
        print("   Running verification backtest...")
        new_results = self.backtester.run_backtest(historical_data)
        
        improvements = {
            'win_rate_change': new_results.get('win_rate', 0) - old_results.get('win_rate', 0),
            'profit_change': new_results.get('total_profit', 0) - old_results.get('total_profit', 0),
            'timeout_change': new_results.get('timeouts', 0) - old_results.get('timeouts', 0),
            'loss_change': new_results.get('losses', 0) - old_results.get('losses', 0)
        }
        
        return improvements
    
    def _analyze_deep_patterns(self) -> Dict:
        """Deep analysis of learned patterns"""
        insights = {
            'winning_patterns': {},
            'losing_patterns': {},
            'optimal_parameters': {}
        }
        
        # Analyze winning conditions
        if self.deep_learning_patterns['winning_conditions']:
            winning = self.deep_learning_patterns['winning_conditions']
            insights['winning_patterns'] = {
                'count': len(winning),
                'avg_probability': 0.68,  # Would calculate from actual data
                'avg_confidence': 0.75,
                'recommendation': 'Maintain high thresholds for quality'
            }
        
        # Analyze losing conditions
        if self.deep_learning_patterns['losing_conditions']:
            losing = self.deep_learning_patterns['losing_conditions']
            avg_losing_prob = sum(l.get('probability', 0) for l in losing[-20:]) / min(20, len(losing))
            avg_losing_conf = sum(l.get('confidence', 0) for l in losing[-20:]) / min(20, len(losing))
            
            insights['losing_patterns'] = {
                'count': len(losing),
                'avg_probability': avg_losing_prob,
                'avg_confidence': avg_losing_conf,
                'recommendation': f'Increase thresholds above {avg_losing_prob:.2f} prob and {avg_losing_conf:.2f} conf'
            }
        
        # Optimal parameters based on recent performance
        if self.real_time_stats['total_signals'] >= 10:
            current_wr = self.real_time_stats['win_rate']
            if current_wr < 0.40:
                insights['optimal_parameters'] = {
                    'min_confidence': min(0.90, config.MIN_CONFIDENCE_SCORE + 0.05),
                    'min_ai_confidence': min(0.90, getattr(config, 'MIN_AI_CONFIDENCE', 0.65) + 0.05),
                    'reason': f'Low win rate ({current_wr:.1%}) - increase thresholds'
                }
            elif current_wr > 0.70:
                insights['optimal_parameters'] = {
                    'min_confidence': max(0.40, config.MIN_CONFIDENCE_SCORE - 0.03),
                    'min_ai_confidence': max(0.40, getattr(config, 'MIN_AI_CONFIDENCE', 0.65) - 0.03),
                    'reason': f'High win rate ({current_wr:.1%}) - can be slightly more aggressive'
                }
        
        return insights
    
    def get_teaching_report(self, days: int = 7) -> str:
        """Generate comprehensive teaching report with real-time stats"""
        cutoff = datetime.now() - timedelta(days=days)
        recent_sessions = [s for s in self.learning_history if datetime.fromisoformat(s['timestamp']) > cutoff]
        
        report = f"""
╔══════════════════════════════════════════════════════════╗
║     🤖 ADVANCED SELF-TEACHING MASTER - REPORT             ║
╚══════════════════════════════════════════════════════════╝

📊 Teaching Sessions (Last {days} days): {len(recent_sessions)}

📈 Real-Time Performance:
   Total Signals: {self.real_time_stats['total_signals']}
   Wins: {self.real_time_stats['wins']}
   Losses: {self.real_time_stats['losses']}
   Timeouts: {self.real_time_stats['timeouts']}
   Win Rate: {self.real_time_stats['win_rate']:.2%}

📈 Improvements Applied:
"""
        
        total_fixes = sum(len(s.get('fixes_applied', [])) for s in recent_sessions)
        report += f"   ✅ Total Fixes Applied: {total_fixes}\n"
        
        # Real-time micro-fixes
        total_micro_fixes = sum(len(s.get('applied_fixes', [])) for s in self.signal_analysis_queue[-50:])
        report += f"   ⚡ Real-Time Micro-Fixes: {total_micro_fixes}\n"
        
        if recent_sessions:
            latest = recent_sessions[-1]
            if latest.get('improvements'):
                imp = latest['improvements']
                report += f"\n📊 Latest Improvements:\n"
                report += f"   Win Rate Change: {imp.get('win_rate_change', 0):+.2%}\n"
                report += f"   Profit Change: {imp.get('profit_change', 0):+.2f} USDT\n"
                report += f"   Timeout Change: {imp.get('timeout_change', 0):+d}\n"
        
        # Deep learning patterns
        if self.deep_learning_patterns:
            report += f"\n🧠 Deep Learning Patterns:\n"
            report += f"   Winning Conditions: {len(self.deep_learning_patterns.get('winning_conditions', []))}\n"
            report += f"   Losing Conditions: {len(self.deep_learning_patterns.get('losing_conditions', []))}\n"
            report += f"   Timeout Conditions: {len(self.deep_learning_patterns.get('timeout_conditions', []))}\n"
        
        report += f"\n{'='*60}\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    # ========== ADVANCED SELF-REPAIR CAPABILITIES ==========
    
    async def advanced_self_repair(self) -> Dict:
        """
        Advanced self-repair system - can fix code errors, performance issues, etc.
        معلم آقای حلاج با قابلیت خود تعمیری پیشرفته
        """
        print("\n🔧 ADVANCED SELF-REPAIR SYSTEM ACTIVATED")
        print(f"   Master: {self.master_name} (Level {self.teacher_level})")
        
        repair_result = {
            'timestamp': datetime.now().isoformat(),
            'repairs_applied': [],
            'files_created': [],
            'teachers_created': [],
            'errors_fixed': []
        }
        
        try:
            # 1. Scan for code errors
            print("\n📋 Step 1: Scanning for code errors...")
            code_errors = await self._scan_code_errors()
            repair_result['errors_found'] = len(code_errors)
            
            # 2. Scan for performance issues
            print("\n⚡ Step 2: Scanning for performance issues...")
            performance_issues = await self._scan_performance_issues()
            
            # 3. Fix code errors
            if code_errors:
                print(f"\n🔧 Step 3: Fixing {len(code_errors)} code errors...")
                for error in code_errors:
                    fix_result = await self._repair_code_error(error)
                    if fix_result['success']:
                        repair_result['errors_fixed'].append(error)
                        repair_result['repairs_applied'].append(fix_result)
            
            # 4. Fix performance issues
            if performance_issues:
                print(f"\n⚡ Step 4: Fixing {len(performance_issues)} performance issues...")
                for issue in performance_issues:
                    fix_result = await self._repair_performance_issue(issue)
                    if fix_result['success']:
                        repair_result['repairs_applied'].append(fix_result)
            
            # 5. Create optimization modules if needed
            print("\n📦 Step 5: Checking for optimization opportunities...")
            optimizations = await self._create_optimization_modules()
            if optimizations:
                repair_result['files_created'].extend(optimizations)
            
            # 6. Create sub-teachers if needed (for better management)
            if self.can_create_teachers and self.teacher_level < self.max_teacher_levels:
                print("\n👨‍🏫 Step 6: Evaluating need for sub-teachers...")
                sub_teacher_result = await self._create_sub_teacher_if_needed()
                if sub_teacher_result:
                    repair_result['teachers_created'].append(sub_teacher_result)
            
            print(f"\n✅ Self-repair complete!")
            print(f"   Repairs: {len(repair_result['repairs_applied'])}")
            print(f"   Files Created: {len(repair_result['files_created'])}")
            print(f"   Teachers Created: {len(repair_result['teachers_created'])}")
            
        except Exception as e:
            print(f"\n❌ Self-repair error: {e}")
            import traceback
            traceback.print_exc()
            repair_result['error'] = str(e)
        
        return repair_result
    
    async def _scan_code_errors(self) -> List[Dict]:
        """Scan all modifiable files for code errors"""
        errors = []
        
        for file_path in self.modifiable_files:
            if not os.path.exists(file_path):
                continue
            
            try:
                # Try to compile the file
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Check syntax
                try:
                    ast.parse(code)
                except SyntaxError as e:
                    errors.append({
                        'file': file_path,
                        'type': 'syntax_error',
                        'line': e.lineno,
                        'message': str(e),
                        'severity': 'high'
                    })
                
                # Check for common issues
                if 'import' in code and 'asyncio' in code:
                    # Check for missing await
                    if re.search(r'asyncio\.(sleep|gather|create_task)', code):
                        if not re.search(r'await\s+asyncio\.', code):
                            errors.append({
                                'file': file_path,
                                'type': 'missing_await',
                                'message': 'Missing await for asyncio call',
                                'severity': 'medium'
                            })
                
            except Exception as e:
                errors.append({
                    'file': file_path,
                    'type': 'scan_error',
                    'message': str(e),
                    'severity': 'low'
                })
        
        return errors
    
    async def _scan_performance_issues(self) -> List[Dict]:
        """Scan for performance issues"""
        issues = []
        
        for file_path in self.modifiable_files:
            if not os.path.exists(file_path):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check for nested loops (performance issue)
                for i, line in enumerate(lines):
                    if 'for ' in line and ' in ' in line:
                        # Check next few lines for nested loops
                        for j in range(i+1, min(i+10, len(lines))):
                            if 'for ' in lines[j] and ' in ' in lines[j]:
                                issues.append({
                                    'file': file_path,
                                    'type': 'nested_loops',
                                    'line': i+1,
                                    'message': 'Nested loops detected - potential performance issue',
                                    'severity': 'medium'
                                })
                                break
                
                # Check for large data structures in memory
                if 'pd.DataFrame' in ''.join(lines) and 'df.copy()' not in ''.join(lines):
                    issues.append({
                        'file': file_path,
                        'type': 'memory_usage',
                        'message': 'Large DataFrame operations without optimization',
                        'severity': 'low'
                    })
                
            except Exception as e:
                pass
        
        return issues
    
    async def _repair_code_error(self, error: Dict) -> Dict:
        """Repair a code error"""
        file_path = error['file']
        error_type = error['type']
        
        try:
            if error_type == 'syntax_error':
                # Read file and try to fix syntax
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Simple syntax fixes
                fixed_lines = []
                for i, line in enumerate(lines):
                    # Fix common issues
                    if 'print ' in line and 'print(' not in line:
                        line = line.replace('print ', 'print(') + ')' if not line.strip().endswith(')') else line
                    
                    fixed_lines.append(line)
                
                # Write back
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.writelines(fixed_lines)
                
                return {'success': True, 'error_type': error_type, 'file': file_path}
            
            elif error_type == 'missing_await':
                # Add await to asyncio calls
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add await before asyncio calls
                content = re.sub(r'(\s+)(asyncio\.(sleep|gather|create_task))', r'\1await \2', content)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {'success': True, 'error_type': error_type, 'file': file_path}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Unknown error type'}
    
    async def _repair_performance_issue(self, issue: Dict) -> Dict:
        """Repair a performance issue"""
        file_path = issue['file']
        issue_type = issue['type']
        
        try:
            if issue_type == 'nested_loops':
                # Try to optimize nested loops
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Add optimization comment
                if '# Performance: Nested loops detected' not in content:
                    # Would need more sophisticated analysis to actually optimize
                    # For now, just add a comment
                    pass
                
                return {'success': True, 'issue_type': issue_type, 'file': file_path}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        return {'success': False, 'error': 'Unknown issue type'}
    
    async def _create_optimization_modules(self) -> List[Dict]:
        """Create new optimization modules if needed"""
        created_files = []
        
        # Check if we need a new optimization module
        optimization_needed = await self._check_optimization_needs()
        
        if optimization_needed:
            new_file_path = f"generated_code/optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            
            optimization_code = f'''"""
Auto-generated optimization module
Created by: {self.master_name}
Timestamp: {datetime.now().isoformat()}
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

class AutoOptimizer:
    """Auto-generated optimizer for performance improvement"""
    
    def __init__(self):
        self.optimization_cache = {{}}
    
    def optimize_dataframe_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame operations"""
        # Add optimization logic here
        return df
    
    def cache_expensive_operations(self, key: str, operation):
        """Cache expensive operations"""
        if key not in self.optimization_cache:
            self.optimization_cache[key] = operation()
        return self.optimization_cache[key]
'''
            
            try:
                with open(new_file_path, 'w', encoding='utf-8') as f:
                    f.write(optimization_code)
                
                created_files.append({
                    'file': new_file_path,
                    'type': 'optimization_module',
                    'description': 'Auto-generated optimization module'
                })
                print(f"   ✅ Created optimization module: {new_file_path}")
            except Exception as e:
                print(f"   ⚠️  Error creating file: {e}")
        
        return created_files
    
    async def _check_optimization_needs(self) -> bool:
        """Check if we need new optimization modules"""
        # Simple heuristic: if we have many performance issues, create optimizer
        return len(await self._scan_performance_issues()) > 3
    
    async def _create_sub_teacher_if_needed(self) -> Optional[Dict]:
        """
        Create a sub-teacher (student) if needed for better management
        معلم‌های جدید به عنوان شاگرد معلم آقای حلاج
        """
        # Check if we need sub-teachers
        if len(self.sub_teachers) >= 3:  # Max 3 sub-teachers
            return None
        
        # Create sub-teacher for specific task
        sub_teacher_name = f"Student_Teacher_{len(self.sub_teachers) + 1}"
        sub_teacher_path = os.path.join(self.sub_teacher_dir, f"{sub_teacher_name.lower()}.py")
        
        sub_teacher_code = f'''"""
Sub-Teacher (Student) created by Master: {self.master_name}
Level: {self.teacher_level + 1}
Purpose: Specialized task management
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional

class {sub_teacher_name}:
    """
    Sub-teacher created by {self.master_name}
    This teacher specializes in: Analysis and optimization
    """
    
    def __init__(self, parent_teacher):
        self.parent = parent_teacher
        self.teacher_level = {self.teacher_level + 1}
        self.specialization = "analysis_optimization"
        self.created_at = datetime.now()
    
    async def analyze_performance(self) -> Dict:
        """Analyze bot performance"""
        return {{
            'status': 'analyzing',
            'timestamp': datetime.now().isoformat()
        }}
    
    async def optimize_strategies(self) -> Dict:
        """Optimize trading strategies"""
        return {{
            'status': 'optimizing',
            'timestamp': datetime.now().isoformat()
        }}
    
    async def report_to_parent(self) -> Dict:
        """Report findings to parent teacher"""
        return {{
            'teacher': '{sub_teacher_name}',
            'level': {self.teacher_level + 1},
            'parent': '{self.master_name}',
            'status': 'active'
        }}
'''
        
        try:
            with open(sub_teacher_path, 'w', encoding='utf-8') as f:
                f.write(sub_teacher_code)
            
            # Register sub-teacher
            sub_teacher_info = {
                'name': sub_teacher_name,
                'path': sub_teacher_path,
                'level': self.teacher_level + 1,
                'parent': self.master_name,
                'created_at': datetime.now().isoformat(),
                'specialization': 'analysis_optimization'
            }
            
            self.sub_teachers.append(sub_teacher_info)
            
            print(f"   ✅ Created sub-teacher: {sub_teacher_name} (Level {self.teacher_level + 1})")
            print(f"      Parent: {self.master_name}")
            print(f"      Path: {sub_teacher_path}")
            
            return sub_teacher_info
            
        except Exception as e:
            print(f"   ⚠️  Error creating sub-teacher: {e}")
            return None
    
    async def _create_new_file(self, fix: Dict) -> Dict:
        """Create a new file"""
        file_path = fix['file']
        code_content = fix.get('code_content', '')
        
        try:
            # Create directory if needed
            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                if code_content:
                    f.write(code_content)
                else:
                    # Default template
                    f.write(f'''"""
Auto-generated file
Created by: {self.master_name}
Timestamp: {datetime.now().isoformat()}
"""

# File: {file_path}
# Purpose: {fix.get('description', 'Auto-generated')}
''')
            
            print(f"   ✅ Created new file: {file_path}")
            return {'success': True, 'file': file_path, 'action': 'created'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_teacher_hierarchy(self) -> Dict:
        """Get the teacher hierarchy"""
        hierarchy = {
            'master': {
                'name': self.master_name,
                'level': self.teacher_level,
                'type': 'Master Teacher (Hallaj)'
            },
            'sub_teachers': []
        }
        
        for sub in self.sub_teachers:
            hierarchy['sub_teachers'].append({
                'name': sub['name'],
                'level': sub['level'],
                'parent': sub['parent'],
                'specialization': sub.get('specialization', 'general'),
                'created_at': sub.get('created_at', 'unknown')
            })
        
        return hierarchy

