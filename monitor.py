"""
Real-time monitoring system for cryptocurrency markets
"""
import asyncio
import ccxt.async_support as ccxt
import pandas as pd
from datetime import datetime, timedelta
import config
from ai_analyzer import AIAnalyzer
from multi_exchange import MultiExchange
from market_data import MarketDataAPI
from scam_detector import ScamDetector
from pattern_learner import PatternLearner
from adaptive_filter import AdaptiveFilter
from loss_analyzer import LossAnalyzer
from coin_specific_learner import CoinSpecificLearner
from advanced_features import AdvancedFeatures

class MarketMonitor:
    def __init__(self, ai_analyzer, signal_tracker=None):
        self.scam_detector = ScamDetector()
        self.pattern_learner = PatternLearner()
        self.adaptive_filter = AdaptiveFilter()
        self.loss_analyzer = LossAnalyzer()
        self.coin_learner = CoinSpecificLearner()  # Coin-specific learning
        self.advanced_features = AdvancedFeatures()  # Advanced features
        self.btc_data = None  # Cache for BTC data
        self.ai_analyzer = ai_analyzer
        self.exchange = None
        self.multi_exchange = MultiExchange()
        self.monitored_coins = {}
        self.price_history = {}
        self.alerts_sent = set()
        # Optional live signal tracker (can be None for backtests)
        self.signal_tracker = signal_tracker
        # Buffer to collect closed signals between cycles
        self._closed_signals_buffer = []
        # Market data API for enhanced data accuracy
        self.market_data_api = MarketDataAPI()
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """Initialize cryptocurrency exchange connection"""
        try:
            exchange_class = getattr(ccxt, config.EXCHANGE_NAME)
            self.exchange = exchange_class({
                'apiKey': config.EXCHANGE_API_KEY if config.EXCHANGE_API_KEY else None,
                'secret': config.EXCHANGE_API_SECRET if config.EXCHANGE_API_SECRET else None,
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
        except Exception as e:
            print(f"Warning: Could not initialize {config.EXCHANGE_NAME}: {e}")
            print("Using multi-exchange fallback...")
    
    async def get_all_trading_pairs(self):
        """Get all available trading pairs from multiple exchanges - skip problematic ones quickly"""
        # Skip primary exchange if it's binance (known to have rate limit issues)
        # Go directly to multi-exchange which uses reliable exchanges
        if self.exchange and config.EXCHANGE_NAME.lower() != 'binance':
            max_retries = 2  # Faster - only 2 retries
            for attempt in range(max_retries):
                try:
                    print(f"📊 Loading markets from {config.EXCHANGE_NAME} (attempt {attempt + 1}/{max_retries})...")
                    markets = await asyncio.wait_for(
                        self.exchange.load_markets(),
                        timeout=30.0  # 30 second timeout
                    )
                    usdt_pairs = [symbol for symbol in markets.keys() 
                                 if symbol.endswith('/USDT') and markets[symbol].get('active', True)]
                    if len(usdt_pairs) > 0:
                        print(f"✅ Successfully loaded {len(usdt_pairs)} USDT trading pairs from {config.EXCHANGE_NAME}")
                        return usdt_pairs
                except Exception as e:
                    error_msg = str(e)
                    if "403" in error_msg or "rate limit" in error_msg.lower() or "429" in error_msg:
                        print(f"⏭️  Skipping {config.EXCHANGE_NAME} (rate limit/access issue) - using multi-exchange...")
                        break  # Skip immediately
                    elif attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        print(f"⏭️  {config.EXCHANGE_NAME} failed - using multi-exchange...")
        
        # Use multi-exchange (primary method - uses ONLY LBank, CoinEx, KuCoin)
        print("\n🔄 Loading from exchanges: LBank, CoinEx, KuCoin...")
        try:
            pairs = await self.multi_exchange.get_all_trading_pairs()
            if len(pairs) > 0:
                print(f"✅ Successfully loaded {len(pairs)} unique pairs from LBank, CoinEx, KuCoin")
                return pairs
        except Exception as e:
            print(f"⚠️  Multi-exchange error: {e}")
        
        # Final fallback (should rarely be needed now)
        print("⚠️  Using fallback list of small cap coins...")
        return [
            'DYM/USDT', 'BANANA/USDT', 'PIXEL/USDT', 'PORTAL/USDT', 'PDA/USDT',
            'AI/USDT', 'XAI/USDT', 'ACE/USDT', 'NFP/USDT', 'MANTA/USDT',
            'ALT/USDT', 'JUP/USDT', 'WLD/USDT', 'ARKM/USDT', 'SEI/USDT',
            'TIA/USDT', 'BLUR/USDT', 'SUI/USDT', 'OP/USDT', 'ARB/USDT',
            'APT/USDT', 'INJ/USDT', 'RENDER/USDT', 'FET/USDT', 'AGIX/USDT'
        ]
    
    async def get_ticker_data(self, symbol):
        """Get current ticker data for a symbol"""
        # Try primary exchange
        if self.exchange:
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                if ticker:
                    return ticker
            except:
                pass
        
        # Fallback to multi-exchange
        return await self.multi_exchange.get_ticker(symbol)
    
    async def _refresh_price_and_recalculate_levels(self, analysis: dict) -> dict:
        """
        ⚡ CRITICAL: Refresh price RIGHT BEFORE sending signal
        Recalculates Entry/TP/SL based on REAL-TIME price to prevent price drift
        
        This ensures that:
        - Entry = Current real-time price (not old price from analysis)
        - TP/SL = Calculated from real-time Entry (not outdated)
        - Signal is valid when user receives it
        """
        try:
            symbol = analysis['symbol']
            signal_type = analysis.get('signal_type', 'PUMP')
            old_entry = analysis.get('entry', analysis.get('current_price', 0))
            old_exit1 = analysis.get('exit1', 0)
            old_exit2 = analysis.get('exit2', 0)
            old_exit3 = analysis.get('exit3', 0)
            old_sl = analysis.get('stop_loss', 0)
            
            # Get REAL-TIME price (fastest possible)
            ticker = await asyncio.wait_for(
                self.get_ticker_data(symbol),
                timeout=2.0  # 2 second timeout for real-time price
            )
            
            if not ticker:
                # Fallback: use last known price from analysis
                print(f"      ⚠️  {symbol}: Could not fetch real-time price, using analysis price")
                return analysis
            
            # Get real-time price
            real_time_price = float(ticker.get('last', ticker.get('close', analysis.get('current_price', 0))))
            
            if real_time_price <= 0:
                print(f"      ⚠️  {symbol}: Invalid real-time price, using analysis price")
                return analysis
            
            # Calculate price change from analysis time to now
            price_change_pct = ((real_time_price - old_entry) / old_entry) * 100 if old_entry > 0 else 0
            
            # If price changed significantly (>3%), log warning
            if abs(price_change_pct) > 3.0:
                print(f"      ⚠️  {symbol}: Price changed {price_change_pct:+.2f}% since analysis! (Old: ${old_entry:.8f} → New: ${real_time_price:.8f})")
            
            # Update current price
            analysis['current_price'] = real_time_price
            
            # Calculate TP/SL percentages from old levels (to maintain risk/reward ratio)
            if old_entry > 0:
                # Calculate what % the old TP/SL were from old entry
                if signal_type == 'PUMP':
                    tp1_pct = ((old_exit1 - old_entry) / old_entry) if old_exit1 > 0 else 0.10
                    tp2_pct = ((old_exit2 - old_entry) / old_entry) if old_exit2 > 0 else 0.15
                    tp3_pct = ((old_exit3 - old_entry) / old_entry) if old_exit3 > 0 else 0.20
                    sl_pct = ((old_entry - old_sl) / old_entry) if old_sl > 0 else 0.07
                else:  # DUMP
                    tp1_pct = ((old_entry - old_exit1) / old_entry) if old_exit1 > 0 else 0.10
                    tp2_pct = ((old_entry - old_exit2) / old_entry) if old_exit2 > 0 else 0.15
                    tp3_pct = ((old_entry - old_exit3) / old_entry) if old_exit3 > 0 else 0.20
                    sl_pct = ((old_sl - old_entry) / old_entry) if old_sl > 0 else 0.07
            else:
                # Default percentages if old entry was invalid
                tp1_pct = 0.10
                tp2_pct = 0.15
                tp3_pct = 0.20
                sl_pct = 0.07
            
            # Recalculate Entry = Real-time price
            new_entry = real_time_price
            
            # Recalculate TP/SL based on new Entry (maintaining same %)
            if signal_type == 'PUMP':
                new_exit1 = new_entry * (1 + tp1_pct)
                new_exit2 = new_entry * (1 + tp2_pct)
                new_exit3 = new_entry * (1 + tp3_pct)
                new_sl = new_entry * (1 - sl_pct)
            else:  # DUMP
                new_exit1 = new_entry * (1 - tp1_pct)
                new_exit2 = new_entry * (1 - tp2_pct)
                new_exit3 = new_entry * (1 - tp3_pct)
                new_sl = new_entry * (1 + sl_pct)
            
            # Update analysis with real-time levels
            analysis['entry'] = round(new_entry, 8)
            analysis['exit1'] = round(new_exit1, 8)
            analysis['exit2'] = round(new_exit2, 8)
            analysis['exit3'] = round(new_exit3, 8)
            analysis['stop_loss'] = round(new_sl, 8)
            
            # Update risk/reward ratio
            if new_entry > 0 and new_sl > 0:
                if signal_type == 'PUMP':
                    risk = abs(new_entry - new_sl) / new_entry
                    reward = abs(new_exit1 - new_entry) / new_entry
                else:
                    risk = abs(new_sl - new_entry) / new_entry
                    reward = abs(new_entry - new_exit1) / new_entry
                
                if risk > 0:
                    analysis['risk_reward_ratio'] = round(reward / risk, 2)
            
            # Log the refresh
            if abs(price_change_pct) > 0.5:  # Only log if significant change
                print(f"      ⚡ {symbol}: Price refreshed! Entry: ${old_entry:.8f} → ${new_entry:.8f} ({price_change_pct:+.2f}%)")
            
            return analysis
            
        except asyncio.TimeoutError:
            print(f"      ⚠️  {symbol}: Timeout fetching real-time price, using analysis price")
            return analysis
        except Exception as e:
            print(f"      ⚠️  {symbol}: Error refreshing price: {e}, using analysis price")
            return analysis
    
    async def get_ohlcv_data(self, symbol, limit=500):
        """Get OHLCV (candlestick) data with retry and fallback from multiple exchanges"""
        # Try primary exchange first
        if self.exchange:
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    ohlcv = await asyncio.wait_for(
                        self.exchange.fetch_ohlcv(symbol, config.CANDLE_INTERVAL, limit=limit),
                        timeout=20.0
                    )
                    if ohlcv and len(ohlcv) > 0:
                        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        return df
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                except Exception:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
        
        # Fallback to multi-exchange (tries all available exchanges)
        try:
            ohlcv = await self.multi_exchange.get_ohlcv(symbol, config.CANDLE_INTERVAL, limit)
            if ohlcv and len(ohlcv) > 0:
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                return df
        except:
            pass
        
        return None
    
    async def filter_coins_by_volume(self, symbols):
        """Filter small cap coins by volume and price - PARALLEL VERSION"""
        filtered = []
        total_to_check = min(len(symbols), config.MAX_COINS_TO_MONITOR * 3)  # Check more to find enough small caps
        symbols_to_check = symbols[:total_to_check]
        
        print(f"🔍 Filtering small cap coins from {len(symbols)} symbols...")
        print(f"   Volume range: ${config.MIN_VOLUME_24H:,.0f} - ${config.MAX_VOLUME_24H:,.0f}")
        print(f"   Checking {total_to_check} symbols in parallel...")
        
        # Process in batches for parallel execution
        batch_size = 20
        checked_count = 0
        for i in range(0, len(symbols_to_check), batch_size):
            batch = symbols_to_check[i:i+batch_size]
            tasks = [self._check_coin(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                checked_count += 1
                if isinstance(result, dict) and result.get('valid'):
                    filtered.append(result['symbol'])
                    if len(filtered) % 5 == 0:
                        print(f"   ✅ Found {len(filtered)}: {result['symbol']} (${result['price']:.6f}, Vol: ${result['volume']:,.0f})")
            
            if len(filtered) >= config.MAX_COINS_TO_MONITOR:
                print(f"   🎯 Reached target of {config.MAX_COINS_TO_MONITOR} coins!")
                break
            
            if checked_count % 100 == 0:
                print(f"   Progress: {checked_count}/{total_to_check} checked, {len(filtered)} found...")
        
        if len(filtered) < config.MAX_COINS_TO_MONITOR:
            print(f"   ⚠️  Only found {len(filtered)} coins (target: {config.MAX_COINS_TO_MONITOR})")
            print(f"   💡 Consider adjusting MIN_VOLUME_24H or MAX_VOLUME_24H in config.py")
        
        print(f"✅ Filtered to {len(filtered)} small cap coins")
        return filtered[:config.MAX_COINS_TO_MONITOR]
    
    async def filter_coins_by_pump_potential(self, symbols):
        """Smart filter to find coins close to pump - فیلتر هوشمند برای پیدا کردن کوین‌های نزدیک به پامپ"""
        total_symbols = len(symbols)
        print(f"\n{'='*70}")
        print(f"🔍 SMART PUMP FILTER - شروع فیلتر هوشمند")
        print(f"{'='*70}")
        print(f"📊 تعداد کل کوین‌ها: {total_symbols}")
        
        min_price_change = getattr(config, 'SMART_FILTER_MIN_PRICE_CHANGE', 0.03)
        min_volume_spike = getattr(config, 'SMART_FILTER_MIN_VOLUME_SPIKE', 1.3)
        min_momentum = getattr(config, 'SMART_FILTER_MIN_MOMENTUM', 0.02)
        max_coins = getattr(config, 'SMART_FILTER_MAX_COINS', 200)
        
        print(f"⚙️  پارامترهای فیلتر:")
        print(f"   • حداقل تغییر قیمت: {min_price_change*100:.1f}%")
        print(f"   • حداقل افزایش حجم: {min_volume_spike:.1f}x")
        print(f"   • حداقل مومنتوم: {min_momentum*100:.1f}%")
        print(f"   • حداکثر کوین انتخاب شده: {max_coins}")
        print(f"{'='*70}\n")
        
        pump_candidates = []
        checked_count = 0
        error_count = 0
        max_to_check = min(total_symbols, 1000)  # Check first 1000 for speed
        
        # Process in batches for speed
        batch_size = 50
        start_time = datetime.now()
        
        for i in range(0, max_to_check, batch_size):
            batch = symbols[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (max_to_check + batch_size - 1) // batch_size
            
            print(f"📦 Batch {batch_num}/{total_batches} | Checking {len(batch)} coins...")
            
            try:
                tasks = [self._check_pump_potential(symbol, min_price_change, min_volume_spike, min_momentum) 
                        for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for idx, result in enumerate(results):
                    checked_count += 1
                    
                    if isinstance(result, Exception):
                        error_count += 1
                        if error_count <= 5:  # Only log first 5 errors
                            print(f"      ⚠️  Error checking {batch[idx]}: {str(result)[:50]}")
                        continue
                    
                    if isinstance(result, dict) and result.get('pump_potential', 0) > 0:
                        pump_candidates.append(result)
                        # Log high potential coins
                        if result.get('pump_potential', 0) > 0.7:
                            print(f"      ⭐ {result['symbol']}: Pump Score = {result.get('pump_potential', 0):.2f} | Price: {result.get('price_change', 0)*100:+.1f}% | Volume: {result.get('volume_spike', 0):.2f}x")
                
                # Progress update every batch
                progress_pct = (checked_count / max_to_check) * 100
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time_per_coin = elapsed / checked_count if checked_count > 0 else 0
                remaining = max_to_check - checked_count
                eta_seconds = remaining * avg_time_per_coin
                
                print(f"      ✅ Progress: {checked_count}/{max_to_check} ({progress_pct:.1f}%) | Found: {len(pump_candidates)} | Errors: {error_count} | ETA: {eta_seconds:.0f}s\n")
                
                if len(pump_candidates) >= max_coins:
                    print(f"   🎯 Reached target of {max_coins} high-potential coins!")
                    break
                    
            except Exception as e:
                error_count += len(batch)
                print(f"      ❌ Batch error: {str(e)[:100]}")
                continue
        
        # Sort by pump potential (highest first)
        print(f"\n{'='*70}")
        print(f"📊 در حال مرتب‌سازی و انتخاب بهترین کوین‌ها...")
        pump_candidates.sort(key=lambda x: x.get('pump_potential', 0), reverse=True)
        
        # Take top candidates
        selected = pump_candidates[:max_coins]
        
        elapsed_total = (datetime.now() - start_time).total_seconds()
        print(f"{'='*70}")
        print(f"✅ SMART PUMP FILTER - تکمیل شد")
        print(f"{'='*70}")
        print(f"📊 نتایج:")
        print(f"   • کل کوین‌های بررسی شده: {checked_count}")
        print(f"   • کوین‌های با پتانسیل بالا: {len(pump_candidates)}")
        print(f"   • کوین‌های انتخاب شده: {len(selected)}")
        print(f"   • خطاها: {error_count}")
        print(f"   • زمان کل: {elapsed_total:.1f} ثانیه")
        
        if len(selected) > 0:
            avg_score = sum(s.get('pump_potential', 0) for s in selected) / len(selected)
            print(f"   • میانگین امتیاز پامپ: {avg_score:.2f}")
            print(f"   • 🏆 Top 5: {', '.join([s['symbol'] for s in selected[:5]])}")
        else:
            print(f"   ⚠️  هیچ کوینی با پتانسیل بالا پیدا نشد!")
            print(f"   💡 پیشنهاد: پارامترهای فیلتر را کاهش دهید")
        
        print(f"{'='*70}\n")
        
        return [s['symbol'] for s in selected]
    
    async def _check_pump_potential(self, symbol, min_price_change, min_volume_spike, min_momentum):
        """Quick check for pump potential - بررسی سریع پتانسیل پامپ"""
        try:
            # Get recent data (only last 30 candles for speed)
            df = await asyncio.wait_for(
                self.get_ohlcv_data(symbol, limit=30),
                timeout=3.0  # Reduced timeout for faster processing
            )
            
            if df is None or len(df) < 20:
                return {'symbol': symbol, 'pump_potential': 0}
            
            # Calculate pump indicators
            price_change_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
            price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
            volume_change = df['volume'].iloc[-10:].mean() / (df['volume'].iloc[-20:-10].mean() + 1e-10) if len(df) >= 20 else 1.0
            
            # Calculate momentum (trend)
            momentum = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] if len(df) >= 20 else 0
            
            # Calculate pump potential score (0-100)
            pump_score = 0.0
            
            # Price change score (0-40 points)
            if price_change_10m >= min_price_change:
                pump_score += min(40, (price_change_10m / min_price_change) * 20)
            elif price_change_5m >= min_price_change * 0.5:
                pump_score += min(20, (price_change_5m / min_price_change) * 10)
            
            # Volume spike score (0-30 points)
            if volume_change >= min_volume_spike:
                pump_score += min(30, ((volume_change - 1.0) / (min_volume_spike - 1.0)) * 20)
            
            # Momentum score (0-30 points)
            if momentum >= min_momentum:
                pump_score += min(30, (momentum / min_momentum) * 15)
            
            # Only return if meets minimum criteria
            if (price_change_10m >= min_price_change * 0.5 or 
                (volume_change >= min_volume_spike and momentum >= min_momentum * 0.5)):
                return {
                    'symbol': symbol,
                    'pump_potential': pump_score,
                    'price_change_10m': price_change_10m,
                    'volume_spike': volume_change,
                    'momentum': momentum
                }
            
            return {'symbol': symbol, 'pump_potential': 0}
        except:
            return {'symbol': symbol, 'pump_potential': 0}
    
    async def _check_coin(self, symbol):
        """Check if a coin meets criteria - STRICT FILTERING with enhanced API data"""
        try:
            # Try to get enhanced data from CoinGecko/CoinMarketCap first
            enhanced_data = await self.market_data_api.get_enhanced_market_data(symbol)
            
            if enhanced_data:
                # Use API data (more accurate)
                volume_24h = enhanced_data.get('volume_24h', 0)
                current_price = enhanced_data.get('price', 0)
                market_cap = enhanced_data.get('market_cap', 0)
            else:
                # Fallback to exchange ticker data
                ticker = await self.get_ticker_data(symbol)
                if not ticker:
                    return {'valid': False}
                
                # Get volume - try multiple methods
                volume_24h = 0
                if 'quoteVolume' in ticker and ticker['quoteVolume']:
                    volume_24h = float(ticker['quoteVolume'])
                elif 'baseVolume' in ticker and 'last' in ticker:
                    base_vol = float(ticker.get('baseVolume', 0))
                    price = float(ticker.get('last', 0) or ticker.get('close', 0) or 0)
                    volume_24h = base_vol * price
                
                current_price = float(ticker.get('last', 0) or ticker.get('close', 0) or 0)
                market_cap = 0  # Not available from exchange ticker
            
            # LENIENT filtering - only check volume and price > 0 (no price cap)
            if (volume_24h >= config.MIN_VOLUME_24H and 
                volume_24h <= config.MAX_VOLUME_24H and
                current_price > 0):  # Only check price > 0, no maximum
                
                # Additional market cap filter if available (very lenient)
                if market_cap > 0:
                    if market_cap < config.MIN_MARKET_CAP or market_cap > config.MAX_MARKET_CAP:
                        return {'valid': False}
                
                return {
                    'valid': True,
                    'symbol': symbol,
                    'price': current_price,
                    'volume': volume_24h,
                    'market_cap': market_cap,
                    'data_source': 'api' if enhanced_data else 'exchange'
                }
        except Exception as e:
            # Silent error
            pass
        
        return {'valid': False}
    
    async def collect_historical_data(self, symbols):
        """Collect historical data for training - FAST VERSION"""
        print(f"📚 Collecting historical data for {len(symbols)} coins (fast mode)...")
        historical_data = {}
        
        # Process in parallel batches
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [self._collect_coin_data(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, dict) and result.get('data') is not None:
                    historical_data[result['symbol']] = result['data']
                    if len(historical_data) % 5 == 0:
                        print(f"   ✅ Collected {len(historical_data)}/{len(symbols)}: {result['symbol']} ({len(result['data'])} candles)")
            
            if (i + batch_size) % 20 == 0:
                print(f"   Progress: {min(i+batch_size, len(symbols))}/{len(symbols)} coins...")
        
        print(f"✅ Collected data for {len(historical_data)} coins")
        return historical_data
    
    async def _collect_coin_data(self, symbol):
        """Collect data for a single coin"""
        try:
            df = await self.get_ohlcv_data(symbol, limit=config.HISTORICAL_DATA_LIMIT)
            if df is not None and len(df) >= config.MIN_HISTORICAL_CANDLES:
                return {'symbol': symbol, 'data': df}
        except Exception as e:
            pass
        return {'symbol': symbol, 'data': None}
    
    async def _quick_priority_check(self, symbol):
        """Quick check to prioritize coins with price/volume movement"""
        try:
            # Get only last 20 candles for speed (instead of 500)
            df = await asyncio.wait_for(
                self.get_ohlcv_data(symbol, limit=20),
                timeout=3.0  # Very fast timeout
            )
            
            if df is None or len(df) < 10:
                return {'symbol': symbol, 'priority': 0}
            
            # Calculate quick metrics
            price_change_10m = abs((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) if len(df) >= 10 else 0
            volume_change = df['volume'].iloc[-5:].mean() / (df['volume'].iloc[-10:-5].mean() + 1e-10) if len(df) >= 10 else 1.0
            
            # Priority score: higher price change and volume spike = higher priority
            priority = (price_change_10m * 100) + (volume_change - 1.0) * 10
            
            return {'symbol': symbol, 'priority': priority}
        except:
            return {'symbol': symbol, 'priority': 0}
    
    async def _quick_filter(self, symbol, df):
        """Fast pre-filter before full AI analysis - saves time"""
        if df is None or len(df) < 10:
            return False
        
        # Quick checks (no AI needed)
        price_change_10m = abs((df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]) if len(df) >= 10 else 0
        volume_change = df['volume'].iloc[-10:].mean() / (df['volume'].iloc[-20:-10].mean() + 1e-10) if len(df) >= 20 else 1.0
        
        # Skip if no significant movement (FASTEST MODE - very relaxed)
        min_move = getattr(config, 'MIN_PRICE_MOVE_FOR_SIGNAL', 0.01)  # Updated to match config
        min_vol = getattr(config, 'MIN_VOLUME_SPIKE_FOR_SIGNAL', 1.20)  # Updated to match config
        
        if price_change_10m < min_move or volume_change < min_vol:
            return False  # Skip - no significant movement
        
        return True  # Pass quick filter - worth full analysis
    
    async def _analyze_coin_independent(self, symbol):
        """
        Independent analysis protocol for a single coin
        Each coin analyzed completely independently - no shared state
        """
        try:
            return await self.monitor_coin(symbol)
        except asyncio.TimeoutError:
            return None
        except Exception:
            return None
    
    async def monitor_coin(self, symbol):
        """Monitor a single coin for pump signals - Enhanced for BTC/ETH and all coins
        INDEPENDENT PROTOCOL: Each coin analyzed separately with its own data and state"""
        try:
            # Optional per-symbol performance-based filter (skip very bad symbols)
            # But allow BTC/ETH even if they have poor stats (they're important)
            is_major_coin = symbol.upper() in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            
            if self.signal_tracker is not None and not is_major_coin:
                stats = self.signal_tracker.get_symbol_stats(symbol, days=getattr(config, "SYMBOL_STATS_LOOKBACK_DAYS", 7))
                min_signals = getattr(config, "SYMBOL_MIN_SIGNALS_FOR_FILTER", 10)
                min_winrate = getattr(config, "SYMBOL_MIN_WIN_RATE", 0.3)
                if stats["total_signals"] >= min_signals and stats["win_rate"] < min_winrate:
                    # Symbol has performed poorly recently - skip to improve overall quality
                    # (still update open signals prices below)
                    if self.signal_tracker is not None:
                        closed = self.signal_tracker.update_with_price(symbol, None)
                        if closed:
                            self._closed_signals_buffer.extend(closed)
                    return None

            # INDEPENDENT PROTOCOL: Get data for this coin only (no shared state)
            # Each coin fetches its own data independently and in parallel
            df = await asyncio.wait_for(
                self.get_ohlcv_data(symbol, limit=200),  # Reduced from 500 to 200 for speed
                timeout=8.0  # Reduced to 8 seconds for faster parallel processing
            )
            
            if df is None or len(df) < 50:
                return None
            
            # FAST PRE-FILTER: Skip coins with no significant movement (saves AI analysis time)
            if not await self._quick_filter(symbol, df):
                # Still update signal tracker for price updates
                if self.signal_tracker is not None:
                    closed = self.signal_tracker.update_with_price(symbol, float(df['close'].iloc[-1]))
                    if closed:
                        self._closed_signals_buffer.extend(closed)
                return None  # Skip - no significant movement
            
            # Scam detection - skip scam/low quality coins
            enhanced_data = await self.market_data_api.get_enhanced_market_data(symbol)
            if self.scam_detector.should_skip_coin(symbol, df, enhanced_data, 
                                                   min_quality_score=0.5):
                print(f"   ⚠️  Skipping {symbol}: Low quality or scam detected")
                return None
            
            # INDEPENDENT PROTOCOL: Store price history for this coin only
            self.price_history[symbol] = df
            
            # INDEPENDENT PROTOCOL: Each coin gets its own analysis (parallel, no blocking)
            # Advanced Feature 1: Multi-Timeframe Analysis (independent for each coin)
            multi_tf_analysis = await self.advanced_features.analyze_multi_timeframe(symbol, df)
            
            # Advanced Feature 2: BTC Correlation Analysis (independent for each coin)
            btc_correlation = {'correlation': 0.0, 'btc_dominance': False}
            if self.btc_data is not None and len(self.btc_data) > 0:
                btc_correlation = self.advanced_features.analyze_btc_correlation(df, self.btc_data)
            
            # INDEPENDENT PROTOCOL: AI Analysis for this coin only (no shared state)
            analysis = self.ai_analyzer.analyze_coin(symbol, df)
            
            if analysis is None:
                # Even if no new signal, still update tracker with latest price
                if self.signal_tracker is not None:
                    closed = self.signal_tracker.update_with_price(symbol, float(df['close'].iloc[-1]))
                    if closed:
                        self._closed_signals_buffer.extend(closed)
                return None
            
            # Apply Multi-Timeframe filter
            if multi_tf_analysis.get('timeframe_alignment'):
                # Boost signal if timeframes align
                if multi_tf_analysis.get('multi_tf_signal') == analysis.get('signal_type'):
                    trend_strength = multi_tf_analysis.get('trend_strength', 0)
                    # Boost confidence based on multi-timeframe alignment
                    analysis['confidence'] = min(1.0, analysis.get('confidence', 0) * (1 + trend_strength * 0.3))
                    analysis['signal_probability'] = min(1.0, analysis.get('signal_probability', 0) * (1 + trend_strength * 0.2))
                    analysis['multi_tf_boost'] = True
                    print(f"   ✅ Multi-timeframe alignment: {analysis.get('signal_type')} confirmed across timeframes")
                else:
                    # Timeframes don't align - reduce confidence
                    analysis['confidence'] = analysis.get('confidence', 0) * 0.8
                    analysis['signal_probability'] = analysis.get('signal_probability', 0) * 0.8
                    analysis['multi_tf_conflict'] = True
                    print(f"   ⚠️  Multi-timeframe conflict: {analysis.get('signal_type')} vs {multi_tf_analysis.get('multi_tf_signal')}")
            
            # Apply BTC Correlation filter - Enhanced for small coins
            # Check if this is a small cap coin that should follow BTC
            coin_category = self.coin_learner._detect_market_cap_category(symbol)
            is_small_cap = coin_category == 'small'
            
            if btc_correlation.get('btc_dominance'):
                # If BTC is dominating and correlation is high, skip signal
                print(f"   ⚠️  Skipping {symbol}: BTC dominance detected (correlation: {btc_correlation.get('correlation', 0):.2f})")
                return None
            
            # For small cap coins: Check BTC correlation more strictly
            if is_small_cap and self.btc_data is not None and len(self.btc_data) > 0:
                # Get recent BTC price change
                btc_current = self.btc_data['close'].iloc[-1]
                btc_10m_ago = self.btc_data['close'].iloc[-10] if len(self.btc_data) >= 10 else btc_current
                btc_change = (btc_current - btc_10m_ago) / btc_10m_ago
                
                # Get coin price change
                coin_current = df['close'].iloc[-1]
                coin_10m_ago = df['close'].iloc[-10] if len(df) >= 10 else coin_current
                coin_change = (coin_current - coin_10m_ago) / coin_10m_ago
                
                # For PUMP signals: BTC should not be dropping significantly
                if signal_type == 'PUMP' and btc_change < -0.015:  # BTC dropping >1.5%
                    print(f"   ⚠️  Skipping {symbol} (small cap): BTC dropping {btc_change:.2%}, blocking PUMP signal")
                    # Learn this correlation
                    self.coin_learner.learn_btc_correlation(symbol, btc_change, coin_change, 'blocked')
                    return None
                
                # For DUMP signals: BTC drop can be confirmation, but if BTC is rising, be cautious
                if signal_type == 'DUMP' and btc_change > 0.02:  # BTC rising >2%
                    print(f"   ⚠️  Skipping {symbol} (small cap): BTC rising {btc_change:.2%}, blocking DUMP signal")
                    self.coin_learner.learn_btc_correlation(symbol, btc_change, coin_change, 'blocked')
                    return None
                
                # Learn correlation for future use
                if signal_type:
                    # Will learn actual outcome later when signal closes
                    pass
            
            # Add multi-timeframe and BTC data to analysis
            analysis['multi_tf_analysis'] = multi_tf_analysis
            analysis['btc_correlation'] = btc_correlation
            
            # Re-calculate signal score with multi-timeframe and BTC data
            # (Score is already calculated in ai_analyzer, but we enhance it here)
            if 'signal_score' in analysis:
                # Enhance score with multi-timeframe and BTC data
                base_score = analysis['signal_score']
                enhancements = 0.0
                
                if multi_tf_analysis and multi_tf_analysis.get('timeframe_alignment'):
                    if multi_tf_analysis.get('multi_tf_boost'):
                        enhancements += 3.0  # Bonus for multi-timeframe alignment
                        print(f"   ✅ Multi-timeframe boost: +3.0 points")
                
                if btc_correlation and not btc_correlation.get('btc_dominance'):
                    btc_corr = btc_correlation.get('correlation', 0)
                    if abs(btc_corr) < 0.3:  # Low correlation = independent move
                        enhancements += 2.0
                        print(f"   ✅ Independent move bonus: +2.0 points")
                    elif btc_correlation.get('btc_dominance') == False:
                        enhancements += 1.0
                        print(f"   ✅ BTC alignment bonus: +1.0 points")
                
                final_score = min(100.0, base_score + enhancements)
                analysis['signal_score'] = round(final_score, 2)
                analysis['is_premium_signal'] = analysis['signal_score'] >= getattr(config, 'PREMIUM_SIGNAL_THRESHOLD', 97.0)
                
                if analysis['is_premium_signal']:
                    print(f"   ⭐⭐ PREMIUM SIGNAL DETECTED! Score: {analysis['signal_score']:.1f}% ⭐⭐")
            
            # Log high probability coins (for debugging)
            pump_prob = analysis['pump_probability']
            confidence = analysis['confidence']
            price_change = analysis['price_change_10m']
            
            # Check if signal is detected (PUMP or DUMP)
            signal_type = analysis.get('signal_type')
            signal_prob = analysis.get('signal_probability', 0)
            
            # Risk/Reward ratio filter - skip low-margin signals
            entry = analysis.get('entry', analysis.get('current_price', 0))
            exit1 = analysis.get('exit1', 0)
            stop_loss = analysis.get('stop_loss', 0)
            
            min_rr_ratio = getattr(config, 'MIN_RISK_REWARD_RATIO', 1.2)
            if self.scam_detector.is_low_margin_signal(entry, exit1, stop_loss, 
                                                      signal_type, min_rr_ratio):
                print(f"   ⚠️  Skipping {symbol}: Low risk/reward ratio")
                return None
            
            # Pattern learning - check if signal matches losing patterns
            if self.pattern_learner.should_skip_signal(analysis, df):
                print(f"   ⚠️  Skipping {symbol}: Matches known losing patterns")
                return None
            
            # Loss prevention - check if signal matches known failure patterns
            should_skip, skip_reason = self.loss_analyzer.should_skip_signal(analysis, df)
            if should_skip:
                print(f"   ⚠️  Skipping {symbol}: {skip_reason}")
                return None
            
            # Pattern matching - boost signals that match winning patterns
            pattern_match = self.pattern_learner.match_pattern(analysis, df)
            if pattern_match['should_boost']:
                # Boost signal probability and confidence
                analysis['signal_probability'] = min(1.0, analysis.get('signal_probability', 0) * pattern_match['boost_factor'])
                analysis['confidence'] = min(1.0, analysis.get('confidence', 0) * pattern_match['boost_factor'])
                analysis['pattern_boost'] = True
                analysis['pattern_match_score'] = pattern_match['match_score']
                print(f"   ✅ Pattern boost: {pattern_match['matches']} matches, score: {pattern_match['match_score']:.2f}")
            
            # Coin-specific learning - check optimal trading time
            should_trade, time_reason = self.coin_learner.should_trade_coin_now(symbol)
            if not should_trade:
                print(f"   ⚠️  Skipping {symbol}: {time_reason}")
                return None
            
            # Get coin-specific strategy
            coin_strategy = self.coin_learner.get_coin_strategy(symbol)
            
            # Apply coin-specific filters
            coin_filters = self.coin_learner.get_coin_specific_filters(symbol, analysis)
            if coin_filters.get('volume_too_low') or coin_filters.get('move_too_small'):
                print(f"   ⚠️  Skipping {symbol}: Below coin-specific thresholds")
                return None
            
            # Adjust confidence based on coin's profile
            if coin_strategy.get('has_profile'):
                if coin_strategy.get('optimal_min_confidence'):
                    required_conf = coin_strategy['optimal_min_confidence']
                    if analysis.get('confidence', 0) < required_conf:
                        print(f"   ⚠️  Skipping {symbol}: Confidence {analysis.get('confidence', 0):.2%} < coin optimal {required_conf:.2%}")
                        return None
                
                # Log coin-specific info
                if coin_strategy.get('win_rate', 0) > 0:
                    print(f"   📊 {symbol} profile: {coin_strategy['win_rate']:.1%} win rate, {coin_strategy['total_signals']} signals")
            
            # Adaptive filter check - use learned thresholds
            if not self.adaptive_filter.should_accept_signal(analysis):
                print(f"   ⚠️  Skipping {symbol}: Below adaptive thresholds")
                return None
            
            # Adjust filters periodically based on performance
            adjustments = self.adaptive_filter.adjust_filters()
            if adjustments:
                print(f"   🔧 Adaptive filter adjustments:")
                for key, adj in adjustments.items():
                    print(f"      {key}: {adj['old']:.3f} → {adj['new']:.3f} ({adj['reason']})")
            
            # Advanced Feature 3: Dynamic Position Sizing
            # Calculate optimal position size based on signal quality
            account_balance = getattr(config, 'ACCOUNT_BALANCE', 100.0)  # Default 100 USDT
            position_size = self.advanced_features.calculate_dynamic_position_size(analysis, account_balance)
            analysis['recommended_position_size'] = position_size
            analysis['position_size_pct'] = (position_size / account_balance * 100) if account_balance > 0 else 10.0
            
            # Log position sizing info
            print(f"   💰 Position size: {position_size:.2f} USDT ({analysis['position_size_pct']:.1f}% of balance)")
            
            # Apply coin-specific TP/SL if learned
            if coin_strategy.get('optimal_tp_pct') or coin_strategy.get('optimal_sl_pct'):
                entry = analysis.get('entry', analysis.get('current_price', 0))
                signal_type = analysis.get('signal_type')
                
                # Get optimal TP/SL for this coin
                current_tp = analysis.get('exit1', entry * 1.10)
                current_sl = analysis.get('stop_loss', entry * 0.93)
                
                optimal_tp, optimal_sl = self.coin_learner.get_coin_optimal_tp_sl(
                    symbol, signal_type, 
                    abs(current_tp - entry) / entry if entry > 0 else 0.10,
                    abs(entry - current_sl) / entry if entry > 0 else 0.07
                )
                
                # Adjust TP/SL based on coin's learned profile
                if signal_type == 'PUMP':
                    analysis['exit1'] = entry * (1 + optimal_tp)
                    analysis['stop_loss'] = entry * (1 - optimal_sl)
                else:  # DUMP
                    analysis['exit1'] = entry * (1 - optimal_tp)
                    analysis['stop_loss'] = entry * (1 + optimal_sl)
                
                print(f"   🎯 Coin-specific TP/SL applied for {symbol}")
            signal_conf = analysis.get('confidence', 0)
            
            if signal_type and signal_prob >= config.MIN_CONFIDENCE_SCORE and signal_conf >= getattr(config, 'MIN_AI_CONFIDENCE', 0.35):
                # Check volume again (double-check to avoid large caps like MBL with $405M)
                try:
                    ticker = await self.get_ticker_data(symbol)
                    if ticker:
                        volume_24h = 0
                        if 'quoteVolume' in ticker and ticker['quoteVolume']:
                            volume_24h = float(ticker['quoteVolume'])
                        elif 'baseVolume' in ticker and 'last' in ticker:
                            base_vol = float(ticker.get('baseVolume', 0))
                            price = float(ticker.get('last', 0) or 0)
                            volume_24h = base_vol * price
                        
                        if volume_24h > config.MAX_VOLUME_24H:
                            # Skip large cap coins (like MBL with $405M, SXP with $133M)
                            print(f"      ⚠️  {symbol}: Volume ${volume_24h:,.0f} exceeds limit ${config.MAX_VOLUME_24H:,.0f} - SKIPPED")
                            return None
                except:
                    pass
                
                # Get signal score
                signal_score = analysis.get('signal_score', 0.0)
                is_premium = analysis.get('is_premium_signal', False)
                
                # Premium signals (97%+) are sent immediately without cooldown
                if is_premium:
                    print(f"      ⭐⭐ PREMIUM SIGNAL ⭐⭐ {symbol}: Score={signal_score:.1f}% | {signal_type} | Prob={signal_prob:.1%}, Conf={signal_conf:.1%}")
                    print(f"      🚀 FAST TRACK: Sending immediately to Telegram (Score >= 97%)")
                    
                    # ⚡ CRITICAL: Refresh price RIGHT BEFORE sending (prevent price drift)
                    analysis = await self._refresh_price_and_recalculate_levels(analysis)
                    
                    # Register premium signal immediately
                    if self.signal_tracker is not None:
                        self.signal_tracker.add_signal(analysis)
                        closed = self.signal_tracker.update_with_price(symbol, analysis['current_price'])
                        if closed:
                            self._closed_signals_buffer.extend(closed)
                    
                    # Return immediately for fast Telegram send
                    return analysis
                
                # Regular signals: Avoid duplicate alerts (15 minute cooldown per coin)
                timestamp = analysis['timestamp']
                date_str = timestamp.strftime('%Y%m%d')
                hour = timestamp.hour
                minute_block = (timestamp.minute // 15) * 15  # 15-minute blocks (0, 15, 30, 45)
                alert_key = f"{symbol}_{signal_type}_{date_str}_{hour:02d}{minute_block:02d}"
                
                if alert_key not in self.alerts_sent:
                    self.alerts_sent.add(alert_key)
                    print(f"      🚨 {symbol}: {signal_type} DETECTED! Score={signal_score:.1f}% | Prob={signal_prob:.1%}, Conf={signal_conf:.1%}, Change={price_change:+.2%}")

                    # ⚡ CRITICAL: Refresh price RIGHT BEFORE sending (prevent price drift)
                    analysis = await self._refresh_price_and_recalculate_levels(analysis)

                    # Register this live signal for later evaluation
                    if self.signal_tracker is not None:
                        self.signal_tracker.add_signal(analysis)

                        # Also update tracker with current price (could instantly close in extreme spikes)
                        closed = self.signal_tracker.update_with_price(symbol, analysis['current_price'])
                        if closed:
                            self._closed_signals_buffer.extend(closed)

                    return analysis
            
            # No qualifying signal but still move open signals forward
            if self.signal_tracker is not None:
                closed = self.signal_tracker.update_with_price(symbol, float(df['close'].iloc[-1]))
                if closed:
                    self._closed_signals_buffer.extend(closed)
            return None
            
        except asyncio.TimeoutError:
            # Silent timeout
            return None
        except Exception as e:
            # Silent error for individual coins
            return None
    
    async def start_monitoring(self, symbols):
        """Start continuous monitoring with INDEPENDENT parallel analysis for each coin"""
        print(f"Starting monitoring for {len(symbols)} coins...")
        print(f"⚡ PARALLEL MODE: Each coin analyzed independently (no batch delays)")
        cycle_count = 0
        # Cache for coin priorities (price/volume changes)
        coin_priorities = {}
        
        while True:
            try:
                cycle_count += 1
                cycle_start = datetime.now()
                print(f"\n🔄 Monitoring Cycle #{cycle_count} - {cycle_start.strftime('%H:%M:%S')}")
                print(f"   ⚡ Analyzing {len(symbols)} coins in PARALLEL (independent protocols)...")
                
                # FAST PRIORITIZATION: Quickly check all coins for price/volume changes
                # This helps prioritize coins with actual movement
                if cycle_count == 1 or cycle_count % 5 == 0:  # Every 5 cycles, re-prioritize
                    print(f"   ⚡ Quick priority check for {len(symbols)} coins...")
                    priority_tasks = []
                    for symbol in symbols[:100]:  # Check first 100 for speed
                        priority_tasks.append(self._quick_priority_check(symbol))
                    
                    priority_results = await asyncio.gather(*priority_tasks, return_exceptions=True)
                    for i, result in enumerate(priority_results):
                        if isinstance(result, dict) and 'priority' in result:
                            coin_priorities[symbols[i]] = result['priority']
                    
                    # Sort symbols by priority (highest first)
                    symbols = sorted(symbols, key=lambda s: coin_priorities.get(s, 0), reverse=True)
                    print(f"   ✅ Prioritized {len([s for s in symbols if coin_priorities.get(s, 0) > 0])} active coins")
                
                # PARALLEL ANALYSIS: All coins analyzed simultaneously with independent protocols
                # Each coin has its own analysis protocol - no batching delays
                all_alerts = []
                checked = 0
                
                # Create independent analysis tasks for ALL coins at once
                print(f"   🚀 Launching {len(symbols)} independent analysis protocols (fully parallel)...")
                analysis_tasks = []
                for symbol in symbols:
                    # Each coin gets its own independent analysis protocol with timeout
                    task = asyncio.wait_for(
                        self._analyze_coin_independent(symbol),
                        timeout=12.0  # 12 seconds per coin (independent protocol)
                    )
                    analysis_tasks.append(task)
                
                # Execute ALL analyses in parallel (no batching - true parallel)
                try:
                    results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                    
                    for j, result in enumerate(results):
                        checked += 1
                        symbol = symbols[j]
                        
                        if isinstance(result, asyncio.TimeoutError):
                            # Silent timeout - coin skipped
                            pass
                        elif isinstance(result, dict) and result is not None:
                            all_alerts.append(result)
                            signal_type = result.get('signal_type', 'PUMP')
                            signal_score = result.get('signal_score', 0)
                            is_premium = result.get('is_premium_signal', False)
                            
                            if is_premium:
                                print(f"      ⭐⭐ {symbol}: PREMIUM {signal_type} SIGNAL! (Score: {signal_score:.1f}%) ⭐⭐")
                            else:
                                print(f"      ✅ {symbol}: {signal_type} detected! (Score: {signal_score:.1f}%)")
                        elif isinstance(result, Exception):
                            # Silent error for individual coins
                            pass
                        # No alert - silent (no spam)
                    
                    # Progress update
                    if checked % 50 == 0 or checked == len(symbols):
                        print(f"      ✓ Completed {checked}/{len(symbols)} coins (parallel)")
                
                except Exception as e:
                    print(f"      ⚠️  Parallel analysis error: {e}")
                
                cycle_time = (datetime.now() - cycle_start).total_seconds()
                print(f"   ✅ Cycle complete in {cycle_time:.1f}s - Checked {checked} coins")
                
                # Collect and reset closed signals for this cycle
                closed_signals = []
                if self.signal_tracker is not None and self._closed_signals_buffer:
                    closed_signals = self._closed_signals_buffer
                    self._closed_signals_buffer = []

                # Yield alerts and any closed signals
                if all_alerts:
                    pump_count = sum(1 for a in all_alerts if a.get('signal_type') == 'PUMP')
                    dump_count = sum(1 for a in all_alerts if a.get('signal_type') == 'DUMP')
                    print(f"   🚨 Found {len(all_alerts)} signals! ({pump_count} PUMP, {dump_count} DUMP)")
                else:
                    min_prob = config.MIN_CONFIDENCE_SCORE
                    min_conf = getattr(config, 'MIN_AI_CONFIDENCE', 0.35)
                    print(f"   ℹ️  No alerts in this cycle (threshold: {min_prob:.0%} prob, {min_conf:.0%} conf)")

                yield {
                    "alerts": all_alerts,
                    "closed_signals": closed_signals,
                }
                
                # Wait before next check
                print(f"   ⏳ Waiting {config.MONITORING_INTERVAL}s before next cycle...\n")
                await asyncio.sleep(config.MONITORING_INTERVAL)
                
            except Exception as e:
                print(f"❌ Error in monitoring loop: {e}")
                import traceback
                traceback.print_exc()
                await asyncio.sleep(config.MONITORING_INTERVAL)
    
    async def close(self):
        """Close exchange connections"""
        if self.exchange:
            try:
                await self.exchange.close()
            except:
                pass
        await self.multi_exchange.close_all()

