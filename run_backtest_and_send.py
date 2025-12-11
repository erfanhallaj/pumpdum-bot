"""
Run backtest and send results to Telegram
"""
import asyncio
import sys
from ai_analyzer import AIAnalyzer
from backtester import Backtester
from monitor import MarketMonitor
from telegram_bot import TelegramNotifier
import config

async def run_backtest_and_send():
    """Run backtest and send results to Telegram"""
    print("="*60)
    print("📊 Running Backtest and Sending to Telegram")
    print("="*60)
    
    # Initialize components
    print("\n1. Initializing components...")
    ai_analyzer = AIAnalyzer()
    monitor = MarketMonitor(ai_analyzer)
    backtester = Backtester(ai_analyzer)
    telegram = TelegramNotifier()
    
    print("2. Collecting historical data...")
    # Collect data for more volatile coins to get better backtest results
    # Using smaller/mid-cap coins that have more price movement
    test_symbols = [
        'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ATOM/USDT', 'LINK/USDT',
        'UNI/USDT', 'AAVE/USDT', 'SUSHI/USDT', 'CRV/USDT', 'MKR/USDT',
        'COMP/USDT', 'SNX/USDT', 'YFI/USDT', '1INCH/USDT', 'BAL/USDT',
        'DOGE/USDT', 'SHIB/USDT', 'PEPE/USDT', 'FLOKI/USDT', 'BONK/USDT'
    ]
    
    print(f"   Collecting data for {len(test_symbols)} coins...")
    historical_data = await monitor.collect_historical_data(test_symbols)
    
    if len(historical_data) == 0:
        error_msg = "❌ No historical data collected! Cannot run backtest."
        print(error_msg)
        await telegram.send_status_update(error_msg)
        return False
    
    print(f"   ✅ Collected data for {len(historical_data)} coins")
    
    # Train models if needed
    print("\n3. Training AI models...")
    try:
        ai_analyzer.train_models(historical_data)
        print("   ✅ Models trained")
    except Exception as e:
        print(f"   ⚠️  Model training error: {e}")
    
    # Run backtest
    print("\n4. Running backtest...")
    try:
        results = backtester.run_backtest(historical_data)
        
        # Generate report
        print("\n5. Generating backtest report...")
        report = backtester.generate_backtest_report(results)
        print(report)
        
        # Send to Telegram
        print("\n6. Sending results to Telegram...")
        success = await telegram.send_backtest_report(report)
        
        if success:
            print("   ✅ Report sent to Telegram successfully!")
        else:
            print("   ⚠️  Failed to send report to Telegram")
        
        # Also send a summary message
        summary = f"""
📊 <b>Backtest Complete</b>

✅ Total Signals: {results['total_signals']}
🚀 Pump Signals: {results.get('pump_signals', 0)}
📉 Dump Signals: {results.get('dump_signals', 0)}
🎯 Wins: {results['correct_predictions']}
❌ Losses: {results.get('losses', 0)}
⏳ Timeouts: {results.get('timeouts', 0)}
📈 Win Rate: {results['win_rate']:.2%}
💰 Total Profit: {results['total_profit']:.2f} USDT
💵 Avg Profit/Trade: {results['average_profit_per_trade']:.2f} USDT
"""
        await telegram.send_message(summary)
        
        print("\n" + "="*60)
        print("✅ Backtest completed and sent to Telegram!")
        print("="*60)
        return True
        
    except Exception as e:
        error_msg = f"❌ Backtest error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        await telegram.send_status_update(error_msg)
        return False

if __name__ == "__main__":
    success = asyncio.run(run_backtest_and_send())
    sys.exit(0 if success else 1)

