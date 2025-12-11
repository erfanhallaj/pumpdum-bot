"""
Advanced AI-Powered Cryptocurrency Pump Detection Bot
Fully automated with self-optimization capabilities
"""
import asyncio
import signal
import sys
from datetime import datetime, timedelta
import config
from ai_analyzer import AIAnalyzer
from monitor import MarketMonitor
from backtester import Backtester
from telegram_bot import TelegramNotifier
from signal_tracker import SignalTracker
from ai_self_improver import AISelfImprover
from strategy_optimizer import StrategyOptimizer
from self_teaching_master import SelfTeachingMaster
from logger import BotLogger

class PumpDetectionBot:
    def __init__(self):
        print("🤖 Initializing AI Pump Detection Bot...")
        
        # Initialize logger first
        self.logger = BotLogger()
        self.logger.log_bot_activity("Bot initialization started")
        
        # Initialize components
        self.ai_analyzer = AIAnalyzer()
        # Signal tracker must be created before monitor so it can be injected
        # Use configurable signal lifetime from config for more flexible timeouts
        self.signal_tracker = SignalTracker(
            max_lifetime_hours=getattr(config, "SIGNAL_MAX_LIFETIME_HOURS", 4)
        )
        self.monitor = MarketMonitor(self.ai_analyzer, signal_tracker=self.signal_tracker)
        self.backtester = Backtester(self.ai_analyzer)
        self.telegram = TelegramNotifier()
        self.ai_self_improver = AISelfImprover(self.signal_tracker)
        self.strategy_optimizer = StrategyOptimizer(self.ai_analyzer)
        # Self-Teaching Master - Full code access for autonomous improvement
        self.self_teaching_master = SelfTeachingMaster(
            self.ai_analyzer,
            self.backtester,
            self.monitor,
            logger=self.logger  # Pass logger to teacher
        )
        
        # State management
        self.running = True
        self.last_training_time = None
        self.last_optimization_time = None
        self.last_daily_report_date = None
        self.last_ai_improvement_time = None
        self.last_teaching_session_time = None
        self.monitored_symbols = []
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        print("✅ Bot initialized successfully!")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutting down bot...")
        self.running = False
    
    async def initial_setup(self):
        """Initial setup: collect data and train models"""
        print("\n📊 Starting initial setup...")
        
        # Send status to Telegram
        await self.telegram.send_status_update("🚀 Bot is starting up...\n\nCollecting market data and training AI models...")
        
        # Get all trading pairs
        print("📈 Fetching trading pairs...")
        all_pairs = await self.monitor.get_all_trading_pairs()
        print(f"Found {len(all_pairs)} trading pairs")
        
        if len(all_pairs) == 0:
            print("⚠️  No trading pairs found, using fallback list of small cap coins")
            all_pairs = [
                'DYM/USDT', 'BANANA/USDT', 'PIXEL/USDT', 'PORTAL/USDT', 'PDA/USDT',
                'AI/USDT', 'XAI/USDT', 'ACE/USDT', 'NFP/USDT', 'MANTA/USDT',
                'ALT/USDT', 'JUP/USDT', 'WLD/USDT', 'ARKM/USDT', 'SEI/USDT',
                'TIA/USDT', 'BLUR/USDT', 'SUI/USDT', 'OP/USDT', 'ARB/USDT',
                'APT/USDT', 'INJ/USDT', 'RENDER/USDT', 'FET/USDT', 'AGIX/USDT'
            ]
        
        # Smart Pump Filter - فیلتر هوشمند برای پیدا کردن کوین‌های نزدیک به پامپ
        if len(all_pairs) > 15:  # Only filter if we have many pairs
            if getattr(config, 'ENABLE_SMART_PUMP_FILTER', True):
                print("🔍 Applying Smart Pump Filter (finding coins close to pump)...")
                filtered_pairs = await self.monitor.filter_coins_by_pump_potential(all_pairs)
                print(f"✅ Found {len(filtered_pairs)} coins with high pump potential")
                if len(filtered_pairs) > 0:
                    all_pairs = filtered_pairs
            else:
                # Fallback to volume filter
                print("🔍 Filtering coins by volume...")
                filtered_pairs = await self.monitor.filter_coins_by_volume(all_pairs)
                print(f"Filtered to {len(filtered_pairs)} coins with sufficient volume")
                if len(filtered_pairs) > 0:
                    all_pairs = filtered_pairs
        
        self.monitored_symbols = all_pairs[:config.MAX_COINS_TO_MONITOR]
        print(f"✅ Monitoring {len(self.monitored_symbols)} coins (high pump potential)")
        self.logger.log_bot_activity("Coins selected for monitoring", {
            'total_pairs': len(all_pairs),
            'monitored_count': len(self.monitored_symbols),
            'coins': self.monitored_symbols[:10]  # First 10 for log
        })
        
        # Quick setup - skip heavy operations on startup
        print("⚡ Fast startup mode - skipping heavy operations...")
        historical_data = {}
        
        # Collect minimal historical data (only if training needed)
        if not config.SKIP_INITIAL_TRAINING:
            print("📚 Collecting historical data for AI training...")
            historical_data = await self.monitor.collect_historical_data(self.monitored_symbols[:30])  # Only first 30 for speed
            print(f"Collected data for {len(historical_data)} coins")
            
            if len(historical_data) > 5:  # Need at least 5 coins
                # Train AI models
                print("🧠 Training AI models (quick mode)...")
                try:
                    self.ai_analyzer.train_models(historical_data)
                    self.last_training_time = datetime.now()
                    print("✅ Models trained successfully!")
                except Exception as e:
                    print(f"⚠️  Training error (will use default models): {e}")
            else:
                print("⚠️  Not enough data for training, using default models")
        else:
            print("⏭️  Skipping initial training (models will train in background)")
        
        # Skip initial backtest for speed
        if not config.SKIP_INITIAL_BACKTEST and len(historical_data) > 0:
            print("📊 Running quick backtest...")
            try:
                backtest_results = self.backtester.run_backtest(historical_data)
                backtest_report = self.backtester.generate_backtest_report(backtest_results)
                await self.telegram.send_backtest_report(backtest_report)
            except Exception as e:
                print(f"⚠️  Backtest error: {e}")
        else:
            print("⏭️  Skipping initial backtest (will run in background)")
        
        # Send status update
        await self.telegram.send_status_update(
            f"✅ Bot setup complete!\n\n"
            f"📊 Monitoring {len(self.monitored_symbols)} coins\n"
            f"🧠 AI models trained and ready\n"
            f"🎯 Alert threshold: {config.MIN_CONFIDENCE_SCORE:.0%} probability\n"
            f"🎯 Confidence threshold: {getattr(config, 'MIN_AI_CONFIDENCE', 0.50):.0%}\n"
            f"⏱️  Check interval: {config.MONITORING_INTERVAL}s\n"
            f"🚀 Starting real-time monitoring..."
        )
        
        # Send test alert to verify Telegram connection
        print("\n📱 Sending test alert to Telegram...")
        test_analysis = {
            'symbol': 'TEST/USDT',
            'current_price': 1.0,
            'signal_type': 'PUMP',
            'signal_probability': 0.75,
            'pump_probability': 0.75,
            'dump_probability': 0.25,
            'confidence': 0.70,
            'price_change_10m': 0.05,
            'price_change_5m': 0.03,
            'volume_24h': 100000,
            'volume_change': 1.5,
            'timestamp': datetime.now(),
            'recommendation': 'BUY',
            'entry': 1.0,
            'exit1': 1.10,
            'exit2': 1.20,
            'exit3': 1.35,
            'stop_loss': 0.93
        }
        test_sent = await self.telegram.send_pump_alert(test_analysis)
        if test_sent:
            print("✅ Test alert sent successfully! Telegram is working.")
        else:
            print("⚠️  Test alert failed. Check Telegram configuration.")
        
        print("✅ Initial setup complete!")
    
    async def periodic_training(self):
        """Periodically retrain models with new data"""
        while self.running:
            try:
                await asyncio.sleep(config.MODEL_UPDATE_INTERVAL)
                
                if not self.running:
                    break
                
                print("\n🔄 Updating AI models with new data...")
                await self.telegram.send_status_update("🔄 Updating AI models with latest market data...")
                
                # Collect fresh data (include BTC/ETH for correlation analysis)
                symbols_to_collect = list(self.monitored_symbols)
                if 'BTC/USDT' not in symbols_to_collect:
                    symbols_to_collect.insert(0, 'BTC/USDT')  # Add BTC for correlation
                if 'ETH/USDT' not in symbols_to_collect:
                    symbols_to_collect.insert(1, 'ETH/USDT')  # Add ETH for multi-correlation
                historical_data = await self.monitor.collect_historical_data(symbols_to_collect)
                
                if len(historical_data) > 0:
                    # Retrain models
                    self.ai_analyzer.train_models(historical_data)
                    self.last_training_time = datetime.now()
                    
                    # Run backtest
                    backtest_results = self.backtester.run_backtest(historical_data)
                    backtest_report = self.backtester.generate_backtest_report(backtest_results)
                    await self.telegram.send_backtest_report(backtest_report)
                    
                    # Self-learning: Test and optimize strategies
                    if getattr(config, 'ENABLE_STRATEGY_OPTIMIZATION', True):
                        print("\n🧠 Running self-learning strategy optimization...")
                        await self.telegram.send_status_update("🧠 Testing and optimizing strategies...")
                        
                        optimization_results = self.strategy_optimizer.optimize_strategies(historical_data)
                        
                        opt_message = (
                            f"🧠 <b>Strategy Optimization Complete</b>\n\n"
                            f"📊 Tested: {optimization_results['total_tested']} strategies\n"
                            f"🏆 Best: {', '.join(optimization_results['best_strategies'][:3])}\n"
                            f"⭐ Best Score: {optimization_results['best_score']:.2f}\n\n"
                            f"✅ Best strategy applied to live trading"
                        )
                        await self.telegram.send_message(opt_message)
                    
                    print("✅ Models updated successfully!")
                    await self.telegram.send_status_update("✅ AI models updated successfully!")
                
            except Exception as e:
                print(f"Error in periodic training: {e}")
    
    async def periodic_optimization(self):
        """Periodically optimize parameters"""
        while self.running:
            try:
                await asyncio.sleep(config.OPTIMIZATION_INTERVAL)
                
                if not self.running:
                    break
                
                if config.AUTO_OPTIMIZE_ENABLED:
                    print("\n🔧 Running periodic optimization...")
                    await self.telegram.send_status_update("🔧 Running self-optimization...")
                    
                    # Collect data
                    historical_data = await self.monitor.collect_historical_data(self.monitored_symbols)
                    
                    if len(historical_data) > 0:
                        # Optimize
                        optimization_results = self.backtester.optimize_parameters(historical_data)
                        await self.telegram.send_optimization_report(optimization_results)
                        self.last_optimization_time = datetime.now()
                        
                        print("✅ Optimization complete!")
                
            except Exception as e:
                print(f"Error in periodic optimization: {e}")
    
    async def periodic_ai_self_improvement(self):
        """Periodically use AI to analyze and improve bot settings"""
        # Wait 24 hours before first run (need data to analyze)
        await asyncio.sleep(86400)
        
        while self.running:
            try:
                # Run every 2 days (don't change too frequently)
                await asyncio.sleep(172800)  # 48 hours
                
                if not self.running:
                    break
                
                # Check if we have enough data
                total_signals = len(self.signal_tracker.history)
                if total_signals < 20:
                    print(f"⏭️  Skipping AI self-improvement: Need more signals (have {total_signals}, need 20+)")
                    continue
                
                print("\n🤖 Running AI self-improvement analysis...")
                await self.telegram.send_status_update("🤖 Running AI self-improvement analysis...")
                
                # Run improvement cycle (auto_apply=False for safety - preview only)
                auto_apply = getattr(config, "AI_SELF_IMPROVE_AUTO_APPLY", False)
                result = await self.ai_self_improver.run_improvement_cycle(auto_apply=auto_apply)
                
                # Send report to Telegram
                changes = result['result'].get('changes', [])
                if len(changes) > 0:
                    report = "🤖 <b>AI Self-Improvement Report</b>\n\n"
                    report += f"📊 Analyzed {result['analysis']['stats']['total_signals']} signals\n"
                    report += f"📈 Win Rate: {result['analysis']['stats']['win_rate']:.2%}\n\n"
                    report += "<b>Recommended Changes:</b>\n"
                    for change in changes:
                        status = "✅ Applied" if change.get('applied') else "💡 Suggested"
                        report += f"{status}: {change['setting']} = {change['old']:.2f} → {change['new']:.2f}\n"
                        report += f"   Reason: {change.get('reason', 'N/A')}\n\n"
                    
                    if not auto_apply:
                        report += "⚠️ <i>Changes are suggestions only. Set AI_SELF_IMPROVE_AUTO_APPLY=True in config to auto-apply.</i>"
                    
                    await self.telegram.send_message(report)
                    self.last_ai_improvement_time = datetime.now()
                    print("✅ AI self-improvement analysis complete!")
                    self.logger.log_teacher_activity("AI self-improvement completed", {
                        'changes_suggested': len(changes)
                    })
                else:
                    print("ℹ️  AI found no improvements needed at this time")
                
            except Exception as e:
                print(f"Error in AI self-improvement: {e}")
                self.logger.log_error("AI Self-Improvement", str(e))
    
    async def periodic_self_teaching(self):
        """Periodically run self-teaching master to improve code - هر 2 ساعت با بک تست"""
        if not getattr(config, 'ENABLE_SELF_TEACHING_MASTER', True):
            return
        
        # Wait shorter time before first run (need some data)
        await asyncio.sleep(3600)  # 1 hour before first run
        
        while self.running:
            try:
                # Run teaching session every 2 hours with backtest
                interval = getattr(config, 'SELF_TEACHING_INTERVAL', 7200)  # Default 2 hours (7200 seconds)
                print(f"\n⏰ Waiting {interval/3600:.1f} hours until next teaching session with backtest...")
                await asyncio.sleep(interval)
                
                if not self.running:
                    break
                
                # Check if we have enough data
                if len(self.monitored_symbols) < 5:
                    print("⏭️  Skipping self-teaching: Need more monitored coins")
                    continue
                
                print("\n🎓 Starting self-teaching session with backtest (هر 2 ساعت)...")
                await self.telegram.send_status_update("🎓 Running self-teaching master session with backtest...")
                self.logger.log_teacher_activity("Teaching session started", {'interval_hours': interval/3600})
                
                # Run teaching session (includes backtest)
                session_result = await self.self_teaching_master.daily_teaching_session()
                
                # Log teaching session results
                if session_result:
                    self.logger.log_teacher_activity("Teaching session completed", {
                        'problems_found': len(session_result.get('problems_found', [])),
                        'fixes_applied': len(session_result.get('fixes_applied', [])),
                        'backtest_results': session_result.get('backtest_results', {})
                    })
                    
                    # Log backtest if available
                    if 'backtest_results' in session_result:
                        self.logger.log_backtest(session_result['backtest_results'])
                
                # Send report
                if session_result.get('fixes_applied'):
                    report = "🎓 <b>Self-Teaching Master Report</b>\n\n"
                    report += f"📊 Problems Found: {len(session_result.get('problems_found', []))}\n"
                    report += f"✅ Fixes Applied: {len(session_result.get('fixes_applied', []))}\n\n"
                    
                    if session_result.get('improvements'):
                        imp = session_result['improvements']
                        report += "<b>Improvements:</b>\n"
                        report += f"   Win Rate: {imp.get('win_rate_change', 0):+.2%}\n"
                        report += f"   Profit: {imp.get('profit_change', 0):+.2f} USDT\n"
                        report += f"   Timeouts: {imp.get('timeout_change', 0):+d}\n"
                    
                    await self.telegram.send_message(report)
                
                self.last_teaching_session_time = datetime.now()
                print("✅ Self-teaching session complete!")
                
            except Exception as e:
                print(f"Error in self-teaching: {e}")
                import traceback
                traceback.print_exc()

    async def daily_performance_reporter(self):
        """Send daily performance report and perform simple self-repair based on live stats."""
        while self.running:
            try:
                # Sleep a bit between checks to avoid tight loop
                await asyncio.sleep(300)  # 5 minutes

                if not self.running or not getattr(config, "DAILY_REPORT_ENABLED", True):
                    continue

                now = datetime.utcnow()
                current_date = now.date()

                # Only once per day after configured hour
                if self.last_daily_report_date == current_date:
                    continue

                report_hour = getattr(config, "DAILY_REPORT_HOUR_UTC", 0)
                if now.hour < report_hour:
                    continue

                # Build stats for yesterday (complete day)
                from datetime import timedelta as _td
                target_day = now - _td(days=1)
                stats = self.signal_tracker.get_daily_stats(target_day)
                
                extra_info = ""

                # Advanced self-repair: auto-tune thresholds based on real win rate & signal count
                if getattr(config, "SELF_REPAIR_ENABLED", True):
                    total = stats.get("total_signals", 0)
                    win_rate = stats.get("win_rate", 0.0)
                    target_win = getattr(config, "TARGET_WIN_RATE", 0.55)
                    step = getattr(config, "THRESHOLD_ADJUST_STEP", 0.05)
                    min_signals = getattr(config, "SELF_REPAIR_MIN_SIGNALS_PER_DAY", 8)

                    # Ranges
                    prob_min, prob_max = getattr(config, "MIN_PROBABILITY_RANGE", (0.4, 0.9))
                    ai_min, ai_max = getattr(config, "MIN_AI_CONFIDENCE_RANGE", (0.4, 0.9))
                    sym_min, sym_max = getattr(config, "SYMBOL_MIN_WIN_RATE_RANGE", (0.3, 0.7))

                    old_prob = config.MIN_CONFIDENCE_SCORE
                    old_ai = getattr(config, "MIN_AI_CONFIDENCE", 0.35)
                    old_sym_min_win = getattr(config, "SYMBOL_MIN_WIN_RATE", 0.45)

                    new_prob = old_prob
                    new_ai = old_ai
                    new_sym_min_win = old_sym_min_win

                    # Only adjust if ما داده کافی داریم
                    if total >= min_signals:
                        if win_rate < target_win - 0.05:
                            # عملکرد ضعیف → سخت‌گیرتر شو (فقط سیگنال‌های قوی‌تر)
                            new_prob = min(prob_max, old_prob + step)
                            new_ai = min(ai_max, old_ai + step)
                            new_sym_min_win = min(sym_max, old_sym_min_win + 0.05)
                            extra_info = (
                                f"⚠️ Win rate below target ({win_rate:.2%} < {target_win:.2%}). "
                                f"Stricter filters → prob: {old_prob:.2f}→{new_prob:.2f}, "
                                f"conf: {old_ai:.2f}→{new_ai:.2f}, "
                                f"symbol_min_win: {old_sym_min_win:.2f}→{new_sym_min_win:.2f}"
                            )
                        elif win_rate > target_win + 0.10:
                            # عملکرد خیلی خوب → کمی تهاجمی‌تر (فرصت‌های بیشتری بگیر)
                            new_prob = max(prob_min, old_prob - step)
                            new_ai = max(ai_min, old_ai - step)
                            new_sym_min_win = max(sym_min, old_sym_min_win - 0.05)
                            extra_info = (
                                f"✅ Win rate well above target ({win_rate:.2%} > {target_win:.2%}). "
                                f"Lighter filters → prob: {old_prob:.2f}→{new_prob:.2f}, "
                                f"conf: {old_ai:.2f}→{new_ai:.2f}, "
                                f"symbol_min_win: {old_sym_min_win:.2f}→{new_sym_min_win:.2f}"
                            )
                    else:
                        # سیگنال کم بوده: اگر برد خوب است، کمی درها را بازتر کن
                        if total > 0 and win_rate >= target_win:
                            new_prob = max(prob_min, old_prob - step / 2)
                            new_ai = max(ai_min, old_ai - step / 2)
                            extra_info = (
                                f"ℹ️ Few signals yesterday ({total}), but good win rate ({win_rate:.2%}). "
                                f"Slightly relaxing thresholds → prob: {old_prob:.2f}→{new_prob:.2f}, "
                                f"conf: {old_ai:.2f}→{new_ai:.2f}"
                            )

                    # Apply updates if changed
                    if new_prob != old_prob or new_ai != old_ai or new_sym_min_win != old_sym_min_win:
                        config.MIN_CONFIDENCE_SCORE = new_prob
                        setattr(config, "MIN_AI_CONFIDENCE", new_ai)
                        setattr(config, "SYMBOL_MIN_WIN_RATE", new_sym_min_win)

                await self.telegram.send_daily_performance(stats, extra_info=extra_info)
                self.last_daily_report_date = current_date

            except Exception as e:
                print(f"Error in daily performance reporter: {e}")
    
    async def monitoring_loop(self):
        """Main monitoring loop with better logging"""
        print("\n👀 Starting real-time monitoring...")
        
        # Fetch BTC data for correlation analysis
        print("📊 Fetching BTC data for correlation analysis...")
        try:
            btc_df = await self.monitor.get_ohlcv_data('BTC/USDT', limit=200)
            if btc_df is not None and len(btc_df) > 0:
                self.monitor.btc_data = btc_df
                print(f"✅ BTC data loaded: {len(btc_df)} candles")
            else:
                print("⚠️  Could not load BTC data, correlation analysis disabled")
        except Exception as e:
            print(f"⚠️  Error loading BTC data: {e}")
        
        # Detect market regime
        if self.monitor.btc_data is not None:
            market_regime = await self.monitor.advanced_features.detect_market_regime(self.monitor.btc_data)
            print(f"📈 Market regime: {market_regime}")
        print(f"📊 Monitoring {len(self.monitored_symbols)} coins")
        print(f"⏱️  Check interval: {config.MONITORING_INTERVAL} seconds")
        print(f"🎯 Confidence threshold: {config.MIN_CONFIDENCE_SCORE:.0%}")
        print("")
        
        try:
            async for result in self.monitor.start_monitoring(self.monitored_symbols):
                if not self.running:
                    break

                # Backwards compatibility: result may be list (alerts only) or dict
                if isinstance(result, dict):
                    alerts = result.get("alerts", [])
                    closed_signals = result.get("closed_signals", [])
                else:
                    alerts = result
                    closed_signals = []

                # Sort alerts: Premium signals first (for immediate sending)
                premium_alerts = [a for a in alerts if a.get('is_premium_signal', False)]
                regular_alerts = [a for a in alerts if not a.get('is_premium_signal', False)]
                
                # Send premium signals IMMEDIATELY (skip fundamental analysis for speed)
                for alert in premium_alerts:
                    signal_type = alert.get('signal_type', 'PUMP')
                    signal_score = alert.get('signal_score', 0)
                    print(f"\n⭐⭐⭐ PREMIUM {signal_type} SIGNAL ⭐⭐⭐")
                    print(f"   Coin: {alert['symbol']}")
                    print(f"   Price: ${alert['current_price']:.8f}")
                    print(f"   Score: {signal_score:.1f}%")
                    print(f"   Probability: {alert.get('signal_probability', alert.get('pump_probability', 0)):.2%}")
                    print(f"   AI Confidence: {alert['confidence']:.2%}")
                    print(f"   ⚡ SENDING IMMEDIATELY (skipping fundamental analysis for speed)")
                    print("")
                    
                    # ⚡ CRITICAL: Refresh price RIGHT BEFORE sending (double-check, prevent price drift)
                    alert = await self.monitor._refresh_price_and_recalculate_levels(alert)
                    
                    # Send immediately without fundamental analysis (for speed)
                    await self.telegram.send_pump_alert(alert)
                    
                    # Log signal
                    self.logger.log_signal(
                        alert.get('signal_type', 'PUMP'),
                        alert['symbol'],
                        {
                            'signal_score': alert.get('signal_score', 0),
                            'signal_probability': alert.get('signal_probability', 0),
                            'current_price': alert.get('current_price', 0),
                            'is_premium': True
                        }
                    )
                
                # Send regular signals with fundamental analysis (if enabled)
                for alert in regular_alerts:
                    signal_type = alert.get('signal_type', 'PUMP')
                    print(f"\n🚨🚨🚨 {signal_type} SIGNAL 🚨🚨🚨")
                    print(f"   Coin: {alert['symbol']}")
                    print(f"   Price: ${alert['current_price']:.8f}")
                    print(f"   Signal Type: {signal_type}")
                    print(f"   Probability: {alert.get('signal_probability', alert.get('pump_probability', 0)):.2%}")
                    print(f"   AI Confidence: {alert['confidence']:.2%}")
                    print(f"   10m Change: {alert['price_change_10m']:.2%}")
                    print("")
                    
                    # ⚡ CRITICAL: Refresh price RIGHT BEFORE sending (double-check, prevent price drift)
                    alert = await self.monitor._refresh_price_and_recalculate_levels(alert)
                    
                    # Fundamental analysis removed as requested
                    # Send alert directly
                    await self.telegram.send_pump_alert(alert)
                    
                    # Log signal
                    self.logger.log_signal(
                        alert.get('signal_type', 'PUMP'),
                        alert['symbol'],
                        {
                            'signal_score': alert.get('signal_score', 0),
                            'signal_probability': alert.get('signal_probability', 0),
                            'current_price': alert.get('current_price', 0),
                            'is_premium': False
                        }
                    )

                # Update scam detector blacklist based on poor performers
                if len(closed_signals) > 0:
                    blacklist_candidates = self.signal_tracker.get_blacklist_candidates(
                        min_signals=5, min_win_rate=0.3, min_profit=-50.0
                    )
                    for symbol in blacklist_candidates:
                        if symbol not in self.monitor.scam_detector.blacklisted_symbols:
                            print(f"   🚫 Blacklisting {symbol}: Poor performance")
                            self.monitor.scam_detector.blacklisted_symbols.add(symbol)
                
                # Send outcome confirmations for closed signals and feed them back to AI
                # REAL-TIME LEARNING: Analyze each closed signal immediately
                for sig in closed_signals:
                    try:
                        # 🎓 REAL-TIME SELF-TEACHING: Analyze signal immediately (BEFORE other learning)
                        if getattr(config, 'ENABLE_REAL_TIME_LEARNING', True) and hasattr(self, 'self_teaching_master'):
                            outcome = sig.status  # 'win', 'loss', or 'timeout'
                            try:
                                learning_result = await self.self_teaching_master.analyze_signal_realtime(sig, outcome)
                                if learning_result.get('micro_fixes_applied', 0) > 0:
                                    print(f"   🎓 Applied {learning_result['micro_fixes_applied']} micro-fixes from signal {sig.id}")
                            except Exception as e:
                                print(f"   ⚠️  Real-time learning error: {e}")
                        
                        # Learn from real outcomes of both PUMP and DUMP signals
                        if sig.features:
                            if sig.signal_type == "PUMP":
                                # PUMP: win = strong up move (good pump), loss = SL (bad pump)
                                if sig.status in ("win", "loss"):
                                    label = 1 if sig.status == "win" else 0
                                    self.ai_analyzer.add_live_example(sig.features, label)
                            elif sig.signal_type == "DUMP":
                                # DUMP: if short wins, price fell → negative pump example (label 0)
                                # If short loses (SL hit), price likely pumped → positive pump example (label 1)
                                if sig.status == "win":
                                    self.ai_analyzer.add_live_example(sig.features, 0)
                                elif sig.status == "loss":
                                    self.ai_analyzer.add_live_example(sig.features, 1)
                            # (Timeouts are ignored as "unclear" outcomes)
                        
                        # Calculate profit percentage
                        if sig.close_price and sig.entry:
                            if sig.signal_type == "PUMP":
                                profit_pct = (sig.close_price - sig.entry) / sig.entry * 100
                            else:  # DUMP
                                profit_pct = (sig.entry - sig.close_price) / sig.entry * 100
                        else:
                            profit_pct = 0.0
                        
                        # Create signal dict for learning
                        signal_dict = {
                            'symbol': sig.symbol,
                            'signal_type': sig.signal_type,
                            'signal_probability': sig.probability,
                            'confidence': sig.confidence,
                            'price_change_10m': 0,  # Would need to get from history
                            'volume_change': 1.0,
                            'risk_reward_ratio': 0,
                            'features': sig.features,
                            'timestamp': sig.timestamp
                        }
                        
                        # Learn patterns from signal outcomes
                        from pattern_learner import PatternLearner
                        pattern_learner = PatternLearner()
                        pattern_learner.learn_from_signal(signal_dict, sig.status, profit_pct)
                        
                        # Deep loss analysis - learn from failures
                        if sig.status in ('loss', 'timeout'):
                            # Get price history for analysis
                            price_history = self.monitor.price_history.get(sig.symbol)
                            
                            # Ensure we have close_price
                            final_price = sig.close_price if sig.close_price is not None else sig.entry
                            
                            loss_analysis = self.monitor.loss_analyzer.analyze_loss(
                                signal_dict, sig.status, 
                                final_price,
                                sig.entry, sig.stop_loss, price_history
                            )
                            
                            if loss_analysis:
                                if loss_analysis.get('should_blacklist'):
                                    print(f"   🚫 Auto-blacklisting {sig.symbol}: {loss_analysis.get('failure_reason')}")
                                    self.monitor.scam_detector.blacklisted_symbols.add(sig.symbol)
                                
                                # Log lessons learned
                                lessons = loss_analysis.get('lessons_learned', [])
                                if lessons:
                                    print(f"   📚 Lessons from {sig.symbol} {sig.status}:")
                                    for lesson in lessons[:3]:  # Top 3 lessons
                                        print(f"      - {lesson}")
                                
                                # Log failure reason
                                failure_reason = loss_analysis.get('failure_reason', 'Unknown')
                                failure_category = loss_analysis.get('failure_category', 'unknown')
                                print(f"   🔍 Failure: {failure_category} - {failure_reason}")
                        
                        # Record for adaptive filter learning
                        self.monitor.adaptive_filter.record_signal(signal_dict, sig.status, profit_pct)
                        
                        # Coin-specific learning - learn from each signal
                        price_history = self.monitor.price_history.get(sig.symbol)
                        self.monitor.coin_learner.learn_from_signal(
                            sig.symbol, signal_dict, sig.status, profit_pct, price_history
                        )

                        await self.telegram.send_signal_outcome(sig)
                        
                        # Log signal outcome
                        self.logger.log_signal_outcome(
                            sig.symbol,
                            sig.status,
                            {
                                'entry': sig.entry,
                                'close_price': sig.close_price,
                                'stop_loss': sig.stop_loss,
                                'pnl_pct': profit_pct,
                                'signal_type': sig.signal_type
                            }
                        )
                    except Exception as e:
                        print(f"Error processing signal outcome for {getattr(sig, 'symbol', '?')}: {e}")
                        self.logger.log_error("Signal Processing", str(e), {'symbol': getattr(sig, 'symbol', '?')})
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            raise
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
            import traceback
            traceback.print_exc()
            self.logger.log_error("Monitoring Loop", str(e), {'traceback': traceback.format_exc()})
            # Don't raise - continue monitoring instead of crashing
            # Wait a bit before continuing to avoid rapid error loops
            await asyncio.sleep(10)
            print("🔄 Continuing monitoring after error recovery...")
    
    async def run(self):
        """Main run method"""
        try:
            # Initial setup
            await self.initial_setup()
            
            if not self.running:
                return
            
            # Start background tasks
            training_task = asyncio.create_task(self.periodic_training())
            optimization_task = asyncio.create_task(self.periodic_optimization())
            daily_task = asyncio.create_task(self.daily_performance_reporter())
            ai_improvement_task = asyncio.create_task(self.periodic_ai_self_improvement())
            # Self-Teaching Master - Full autonomous code improvement
            teaching_task = asyncio.create_task(self.periodic_self_teaching())
            
            # Start monitoring (this is the main loop)
            await self.monitoring_loop()
            
            # Cancel background tasks
            training_task.cancel()
            optimization_task.cancel()
            daily_task.cancel()
            ai_improvement_task.cancel()
            
        except KeyboardInterrupt:
            print("\n\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"Fatal error: {e}")
            import traceback
            traceback.print_exc()
            try:
                await self.telegram.send_status_update(f"❌ Fatal error occurred: {str(e)}")
            except:
                pass  # Don't fail on telegram errors
        
        finally:
            # Cleanup
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        print("🧹 Cleaning up...")
        await self.monitor.close()
        
        # Save final logs and summary
        if hasattr(self, 'logger'):
            self.logger.log_bot_activity("Bot shutdown")
            summary = self.logger.get_daily_summary()
            print(f"\n📊 Daily Summary:")
            print(f"   Total Activities: {summary['total_activities']}")
            print(f"   Signals: {summary['signals']}")
            print(f"   Wins: {summary['signal_outcomes']['wins']}")
            print(f"   Losses: {summary['signal_outcomes']['losses']}")
            print(f"   Timeouts: {summary['signal_outcomes']['timeouts']}")
            print(f"   Teacher Fixes: {summary['teacher_fixes']}")
            print(f"   Backtests: {summary['backtests']}")
            print(f"   Errors: {summary['errors']}")
            self.logger.close()
            print(f"\n📝 Logs saved to logs/ directory")
            print(f"   - bot_{datetime.now().strftime('%Y%m%d')}.log")
            print(f"   - teacher_{datetime.now().strftime('%Y%m%d')}.log")
            print(f"   - signals_{datetime.now().strftime('%Y%m%d')}.log")
            print(f"   - activities_{datetime.now().strftime('%Y%m%d')}.json")
        
        print("✅ Cleanup complete")

async def main():
    """Main entry point"""
    bot = PumpDetectionBot()
    await bot.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

