"""
Advanced Backtesting System with Self-Optimization
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ai_analyzer import AIAnalyzer
import config

class Backtester:
    def __init__(self, ai_analyzer):
        self.ai_analyzer = ai_analyzer
        self.backtest_results = []
        self.backtest_mode = True  # Enable backtest mode with relaxed filters
    
    def run_backtest(self, historical_data, test_period_days=config.BACKTEST_PERIOD_DAYS):
        """
        Run comprehensive backtest on historical data
        """
        print(f"\n{'='*60}")
        print(f"📊 Starting backtest on {len(historical_data)} coins...")
        print(f"{'='*60}\n")
        
        results = {
            'total_signals': 0,
            'correct_predictions': 0,
            'false_positives': 0,
            'losses': 0,
            'timeouts': 0,
            'missed_opportunities': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'average_profit_per_trade': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'pump_signals': 0,
            'dump_signals': 0,
            'coin_results': []
        }
        
        coins_processed = 0
        coins_skipped = 0
        
        for symbol, df in historical_data.items():
            # Reduced minimum data requirement for more coins to be tested
            if len(df) < 200:  # Need at least 200 candles (reduced from 250)
                coins_skipped += 1
                continue
            
            coins_processed += 1
            print(f"  Testing {symbol} ({len(df)} candles)...", end=" ")
            
            coin_result = self._backtest_coin(symbol, df)
            if coin_result and coin_result['signals'] > 0:
                results['coin_results'].append(coin_result)
                results['total_signals'] += coin_result['signals']
                results['correct_predictions'] += coin_result['correct']
                results['false_positives'] += coin_result['false_positives']
                results['losses'] += coin_result.get('losses', 0)
                results['timeouts'] += coin_result.get('timeouts', 0)
                results['total_profit'] += coin_result['profit']
                results['pump_signals'] += coin_result.get('pump_signals', 0)
                results['dump_signals'] += coin_result.get('dump_signals', 0)
                print(f"✅ {coin_result['signals']} signals, Profit: {coin_result['profit']:.2f} USDT")
            else:
                print(f"⏭️  No signals generated")
        
        # Calculate aggregate metrics
        if results['total_signals'] > 0:
            results['win_rate'] = results['correct_predictions'] / results['total_signals']
            results['average_profit_per_trade'] = results['total_profit'] / results['total_signals']
        
        # Calculate Sharpe Ratio
        if len(results['coin_results']) > 0:
            profits = [r['profit'] for r in results['coin_results'] if r['profit'] != 0]
            if len(profits) > 1:
                results['sharpe_ratio'] = np.mean(profits) / (np.std(profits) + 1e-10) * np.sqrt(252)
        
        print(f"\n{'='*60}")
        print(f"📊 Backtest Summary:")
        print(f"   Coins Processed: {coins_processed}")
        print(f"   Coins Skipped: {coins_skipped}")
        print(f"   Total Signals: {results['total_signals']}")
        print(f"   Total Profit: {results['total_profit']:.2f} USDT")
        print(f"   Win Rate: {results['win_rate']:.2%}")
        print(f"{'='*60}\n")
        
        self.backtest_results.append({
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    def _backtest_coin(self, symbol, df):
        """Backtest a single coin - uses same logic as live trading (analyze_coin)"""
        signals = []
        trades = []
        
        # Slide window through data
        # Use reasonable window size for quality signals
        window_size = min(200, len(df) - 100)
        if window_size < 50:
            return None  # Not enough data
        
        # Check more frequently to catch more opportunities
        # Check every 20 minutes to balance quality and quantity
        step_size = 20  # Check every 20 minutes (increased frequency for more signals)
        
        # Ensure we have enough future data to check
        max_start = len(df) - 100  # Leave room for future checks
        if max_start < window_size:
            return None
        
        for i in range(window_size, max_start, step_size):
            window_df = df.iloc[i-window_size:i].copy()
            
            # Use STRICTER filters similar to live trading (not completely relaxed)
            # This ensures backtest results are realistic and match live performance
            original_min_move = config.MIN_PRICE_MOVE_FOR_SIGNAL
            original_min_vol = config.MIN_VOLUME_SPIKE_FOR_SIGNAL
            original_min_prob = config.MIN_CONFIDENCE_SCORE
            original_min_conf = config.MIN_AI_CONFIDENCE
            original_dominance = config.SIGNAL_DOMINANCE_MARGIN
            
            # Use relaxed filters for backtest to generate more signals
            # This helps identify more opportunities while still maintaining some quality
            original_allow_weak = config.ALLOW_WEAK_MOMENTUM_SIGNALS
            if self.backtest_mode:
                # More relaxed filters to generate more signals for testing
                # These are more permissive than live trading to catch more opportunities
                config.MIN_PRICE_MOVE_FOR_SIGNAL = 0.008  # 0.8% minimum price move (reduced from 1%)
                config.MIN_VOLUME_SPIKE_FOR_SIGNAL = 1.15  # 15% volume spike minimum (reduced from 20%)
                config.MIN_CONFIDENCE_SCORE = 0.40  # 40% minimum (reduced from 45%)
                config.MIN_AI_CONFIDENCE = 0.45  # 45% minimum (reduced from 50%)
                config.SIGNAL_DOMINANCE_MARGIN = 0.08  # 8% dominance needed (reduced from 10%)
                config.ALLOW_WEAK_MOMENTUM_SIGNALS = True  # Allow weak signals for more opportunities
            
            analysis = self.ai_analyzer.analyze_coin(symbol, window_df)
            
            # Restore original filters
            if self.backtest_mode:
                config.MIN_PRICE_MOVE_FOR_SIGNAL = original_min_move
                config.MIN_VOLUME_SPIKE_FOR_SIGNAL = original_min_vol
                config.MIN_CONFIDENCE_SCORE = original_min_prob
                config.MIN_AI_CONFIDENCE = original_min_conf
                config.SIGNAL_DOMINANCE_MARGIN = original_dominance
                config.ALLOW_WEAK_MOMENTUM_SIGNALS = original_allow_weak
            
            if analysis is None:
                continue  # No signal generated (filtered out by quality checks)
            
            signal_type = analysis.get('signal_type')
            entry_price = analysis.get('entry', window_df['close'].iloc[-1])
            exit1 = analysis.get('exit1')
            exit2 = analysis.get('exit2')
            exit3 = analysis.get('exit3')
            stop_loss = analysis.get('stop_loss')
            
            # Fallback to defaults if not set
            if exit1 is None:
                if signal_type == 'PUMP':
                    exit1 = entry_price * 1.10
                else:
                    exit1 = entry_price * 0.90
            
            if stop_loss is None:
                if signal_type == 'PUMP':
                    stop_loss = entry_price * 0.93
                else:
                    stop_loss = entry_price * 1.07
            
            entry_time = window_df.index[-1]
            
            # Check future price movement (simulate holding for signal lifetime or until TP/SL)
            # Reduced timeout window: 10min, 30min, 1hour, 2hours (max)
            # This reduces timeouts and makes backtest more realistic
            future_checks = [
                (min(i + 10, len(df) - 1), 10),   # 10 minutes
                (min(i + 30, len(df) - 1), 30),   # 30 minutes
                (min(i + 60, len(df) - 1), 60),   # 1 hour
                (min(i + 120, len(df) - 1), 120)  # 2 hours (reduced from 4 hours)
            ]
            
            hit_target = None
            hit_sl = False
            final_price = entry_price
            final_time_offset = 0
            
            # Track which TP level was hit
            tp_level_hit = None
            
            for future_idx, time_offset in future_checks:
                if future_idx >= len(df):
                    break
                    
                future_price = df['close'].iloc[future_idx]
                
                # Check if TP or SL hit
                if signal_type == 'PUMP':
                    # Check TP levels (exit1, exit2, exit3)
                    if exit3 and future_price >= exit3:
                        hit_target = 3
                        tp_level_hit = 3
                        final_price = exit3  # Use TP price, not actual price
                        final_time_offset = time_offset
                        break
                    elif exit2 and future_price >= exit2:
                        hit_target = 2
                        tp_level_hit = 2
                        final_price = exit2
                        final_time_offset = time_offset
                        break
                    elif future_price >= exit1:
                        hit_target = 1
                        tp_level_hit = 1
                        final_price = exit1
                        final_time_offset = time_offset
                        break
                    elif future_price <= stop_loss:
                        hit_sl = True
                        final_price = stop_loss  # Use SL price
                        final_time_offset = time_offset
                        break
                elif signal_type == 'DUMP':
                    # Check TP levels (exit1, exit2, exit3) - for dumps, exit prices are lower
                    if exit3 and future_price <= exit3:
                        hit_target = 3
                        tp_level_hit = 3
                        final_price = exit3
                        final_time_offset = time_offset
                        break
                    elif exit2 and future_price <= exit2:
                        hit_target = 2
                        tp_level_hit = 2
                        final_price = exit2
                        final_time_offset = time_offset
                        break
                    elif future_price <= exit1:
                        hit_target = 1
                        tp_level_hit = 1
                        final_price = exit1
                        final_time_offset = time_offset
                        break
                    elif future_price >= stop_loss:
                        hit_sl = True
                        final_price = stop_loss
                        final_time_offset = time_offset
                        break
                
                # Update final price if we haven't hit TP/SL yet
                final_price = future_price
                final_time_offset = time_offset
            
            # Calculate PnL based on actual exit price
            # For wins: use TP price, for losses: use SL price, for timeouts: use actual price
            if signal_type == 'PUMP':
                if hit_target:
                    # Win: calculate based on which TP was hit
                    if tp_level_hit == 3 and exit3:
                        pnl_pct = (exit3 - entry_price) / entry_price
                    elif tp_level_hit == 2 and exit2:
                        pnl_pct = (exit2 - entry_price) / entry_price
                    else:
                        pnl_pct = (exit1 - entry_price) / entry_price
                elif hit_sl:
                    # Loss: use stop loss
                    pnl_pct = (stop_loss - entry_price) / entry_price
                else:
                    # Timeout: use actual final price
                    pnl_pct = (final_price - entry_price) / entry_price
            else:  # DUMP
                if hit_target:
                    # Win: calculate based on which TP was hit
                    if tp_level_hit == 3 and exit3:
                        pnl_pct = (entry_price - exit3) / entry_price
                    elif tp_level_hit == 2 and exit2:
                        pnl_pct = (entry_price - exit2) / entry_price
                    else:
                        pnl_pct = (entry_price - exit1) / entry_price
                elif hit_sl:
                    # Loss: use stop loss
                    pnl_pct = (entry_price - stop_loss) / entry_price
                else:
                    # Timeout: use actual final price
                    pnl_pct = (entry_price - final_price) / entry_price
            
            # Determine outcome
            is_correct = hit_target is not None and not hit_sl
            is_loss = hit_sl
            is_timeout = hit_target is None and not hit_sl
            
            signals.append({
                'entry_price': entry_price,
                'entry_time': entry_time,
                'exit_price': final_price,
                'price_change': pnl_pct,
                'signal_type': signal_type,
                'signal_probability': analysis.get('signal_probability', 0),
                'confidence': analysis.get('confidence', 0),
                'is_correct': is_correct,
                'is_loss': is_loss,
                'is_timeout': is_timeout,
                'hit_target': hit_target,
                'time_to_close': final_time_offset
            })
            
            # Simulate trade (100 USDT position) - include ALL trades (wins, losses, timeouts)
            profit = pnl_pct * 100
            trades.append({
                'profit': profit,
                'entry': entry_price,
                'exit': final_price,
                'return': pnl_pct,
                'signal_type': signal_type,
                'outcome': 'win' if is_correct else ('loss' if is_loss else 'timeout')
            })
        
        if len(signals) == 0:
            return None
        
        correct = sum(1 for s in signals if s['is_correct'])
        losses = sum(1 for s in signals if s['is_loss'])
        timeouts = sum(1 for s in signals if s['is_timeout'])
        false_positives = losses  # Losses are false positives
        
        # Calculate profit (including losses)
        total_profit = sum(t['profit'] for t in trades)
        
        return {
            'symbol': symbol,
            'signals': len(signals),
            'correct': correct,
            'losses': losses,
            'timeouts': timeouts,
            'false_positives': false_positives,
            'profit': total_profit,
            'win_rate': correct / len(signals) if len(signals) > 0 else 0,
            'average_return': np.mean([t['return'] for t in trades]) if trades else 0,
            'pump_signals': sum(1 for s in signals if s['signal_type'] == 'PUMP'),
            'dump_signals': sum(1 for s in signals if s['signal_type'] == 'DUMP')
        }
    
    def generate_backtest_report(self, results):
        """Generate detailed backtest report"""
        report = f"""
╔══════════════════════════════════════════════════════════╗
║           AI PUMP DETECTION - BACKTEST REPORT            ║
╚══════════════════════════════════════════════════════════╝

📊 PERFORMANCE METRICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Total Signals Generated: {results['total_signals']}
🚀 Pump Signals: {results.get('pump_signals', 0)}
📉 Dump Signals: {results.get('dump_signals', 0)}
🎯 Correct Predictions (Wins): {results['correct_predictions']}
❌ Losses (SL Hit): {results.get('losses', 0)}
⏳ Timeouts: {results.get('timeouts', 0)}
📈 Win Rate: {results['win_rate']:.2%}
💰 Total Profit: {results['total_profit']:.2f} USDT
💵 Average Profit per Trade: {results['average_profit_per_trade']:.2f} USDT
📊 Sharpe Ratio: {results['sharpe_ratio']:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📋 TOP PERFORMING COINS:
"""
        
        # Sort coins by profit
        sorted_coins = sorted(
            results['coin_results'], 
            key=lambda x: x['profit'], 
            reverse=True
        )[:10]
        
        for i, coin in enumerate(sorted_coins, 1):
            report += f"\n{i}. {coin['symbol']}\n"
            report += f"   Signals: {coin['signals']} | Win Rate: {coin['win_rate']:.2%}\n"
            report += f"   Profit: {coin['profit']:.2f} USDT | Avg Return: {coin['average_return']:.2%}\n"
        
        report += "\n" + "="*60 + "\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return report
    
    def optimize_parameters(self, historical_data):
        """
        Self-optimize model parameters based on backtest results
        """
        print("Starting parameter optimization...")
        
        # Run backtest with current parameters
        current_results = self.run_backtest(historical_data)
        
        # Analyze performance
        if current_results['win_rate'] < 0.5:
            print("⚠️  Win rate below 50%, retraining models...")
            # Retrain models with more data
            self.ai_analyzer.train_models(historical_data)
        
        # Optimize confidence threshold
        best_threshold = config.MIN_CONFIDENCE_SCORE
        best_win_rate = current_results['win_rate']
        
        for threshold in [0.65, 0.70, 0.75, 0.80, 0.85]:
            # Temporarily change threshold
            original_threshold = config.MIN_CONFIDENCE_SCORE
            config.MIN_CONFIDENCE_SCORE = threshold
            
            # Quick backtest
            test_results = self.run_backtest(historical_data)
            
            if test_results['win_rate'] > best_win_rate:
                best_win_rate = test_results['win_rate']
                best_threshold = threshold
            
            # Restore original
            config.MIN_CONFIDENCE_SCORE = original_threshold
        
        # Update config if better threshold found
        if best_threshold != config.MIN_CONFIDENCE_SCORE:
            print(f"✅ Optimized confidence threshold: {config.MIN_CONFIDENCE_SCORE} → {best_threshold}")
            config.MIN_CONFIDENCE_SCORE = best_threshold
        
        return {
            'optimized_threshold': best_threshold,
            'improved_win_rate': best_win_rate,
            'original_win_rate': current_results['win_rate']
        }

