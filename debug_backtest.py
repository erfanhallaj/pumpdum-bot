"""
Debug backtest to see why no signals are generated
"""
import asyncio
import pandas as pd
import numpy as np
from ai_analyzer import AIAnalyzer
from monitor import MarketMonitor
import config

async def debug_backtest():
    """Debug why no signals are generated"""
    print("="*60)
    print("🔍 Debugging Backtest - Why No Signals?")
    print("="*60)
    
    ai_analyzer = AIAnalyzer()
    monitor = MarketMonitor(ai_analyzer)
    
    # Get data for one coin
    print("\n1. Collecting data for BTC/USDT...")
    historical_data = await monitor.collect_historical_data(['BTC/USDT'])
    
    if 'BTC/USDT' not in historical_data:
        print("❌ No data collected!")
        return
    
    df = historical_data['BTC/USDT']
    print(f"   ✅ Collected {len(df)} candles")
    
    # Check price movements
    print("\n2. Analyzing price movements...")
    price_changes = []
    volume_changes = []
    
    for i in range(50, len(df) - 10, 20):
        window_df = df.iloc[i-50:i].copy()
        if len(window_df) < 50:
            continue
        
        price_change_10m = (window_df['close'].iloc[-1] - window_df['close'].iloc[-10]) / window_df['close'].iloc[-10] if len(window_df) >= 10 else 0
        volume_change = window_df['volume'].iloc[-10:].mean() / (window_df['volume'].iloc[-20:-10].mean() + 1e-10) if len(window_df) >= 20 else 1.0
        
        price_changes.append(abs(price_change_10m))
        volume_changes.append(volume_change)
    
    print(f"   Price changes (10m): min={min(price_changes):.4f}, max={max(price_changes):.4f}, avg={np.mean(price_changes):.4f}")
    print(f"   Volume changes: min={min(volume_changes):.2f}x, max={max(volume_changes):.2f}x, avg={np.mean(volume_changes):.2f}x")
    print(f"   Current filters:")
    print(f"      MIN_PRICE_MOVE_FOR_SIGNAL: {config.MIN_PRICE_MOVE_FOR_SIGNAL}")
    print(f"      MIN_VOLUME_SPIKE_FOR_SIGNAL: {config.MIN_VOLUME_SPIKE_FOR_SIGNAL}")
    print(f"      MIN_CONFIDENCE_SCORE: {config.MIN_CONFIDENCE_SCORE}")
    print(f"      MIN_AI_CONFIDENCE: {config.MIN_AI_CONFIDENCE}")
    print(f"      SIGNAL_DOMINANCE_MARGIN: {config.SIGNAL_DOMINANCE_MARGIN}")
    print(f"      ALLOW_WEAK_MOMENTUM_SIGNALS: {config.ALLOW_WEAK_MOMENTUM_SIGNALS}")
    
    # Try with relaxed filters
    print("\n3. Testing with relaxed filters...")
    original_min_move = config.MIN_PRICE_MOVE_FOR_SIGNAL
    original_min_vol = config.MIN_VOLUME_SPIKE_FOR_SIGNAL
    original_min_prob = config.MIN_CONFIDENCE_SCORE
    original_min_conf = config.MIN_AI_CONFIDENCE
    original_dominance = config.SIGNAL_DOMINANCE_MARGIN
    original_allow_weak = config.ALLOW_WEAK_MOMENTUM_SIGNALS
    
    config.MIN_PRICE_MOVE_FOR_SIGNAL = 0.001  # 0.1%
    config.MIN_VOLUME_SPIKE_FOR_SIGNAL = 1.05  # 1.05x
    config.MIN_CONFIDENCE_SCORE = 0.30  # 30%
    config.MIN_AI_CONFIDENCE = 0.35  # 35%
    config.SIGNAL_DOMINANCE_MARGIN = 0.01  # 1%
    config.ALLOW_WEAK_MOMENTUM_SIGNALS = True
    
    signals_found = 0
    for i in range(200, len(df) - 150, 50):
        window_df = df.iloc[i-200:i].copy()
        if len(window_df) < 200:
            continue
        
        analysis = ai_analyzer.analyze_coin('BTC/USDT', window_df)
        if analysis:
            signals_found += 1
            print(f"   ✅ Signal found! Type: {analysis.get('signal_type')}, Prob: {analysis.get('signal_probability'):.2%}, Conf: {analysis.get('confidence'):.2%}")
            if signals_found >= 3:
                break
    
    if signals_found == 0:
        print("   ❌ Still no signals even with very relaxed filters!")
        print("   This suggests the models might not be predicting well on this data.")
    
    # Restore
    config.MIN_PRICE_MOVE_FOR_SIGNAL = original_min_move
    config.MIN_VOLUME_SPIKE_FOR_SIGNAL = original_min_vol
    config.MIN_CONFIDENCE_SCORE = original_min_prob
    config.MIN_AI_CONFIDENCE = original_min_conf
    config.SIGNAL_DOMINANCE_MARGIN = original_dominance
    config.ALLOW_WEAK_MOMENTUM_SIGNALS = original_allow_weak
    
    print(f"\n   Found {signals_found} signals with relaxed filters")

if __name__ == "__main__":
    asyncio.run(debug_backtest())

