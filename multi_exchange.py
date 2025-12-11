"""
Multi-exchange support for better data availability
"""
import ccxt.async_support as ccxt
import asyncio
import config

class MultiExchange:
    def __init__(self):
        self.exchanges = {}
        self.primary_exchange = None
        self.initialize_exchanges()
    
    def initialize_exchanges(self):
        """Initialize only LBank, CoinEx, and KuCoin exchanges"""
        # ONLY these 3 exchanges as requested by user
        exchange_list = [
            'lbank',     # LBank exchange
            'coinex',    # CoinEx exchange
            'kucoin',    # KuCoin exchange
        ]
        
        for exchange_name in exchange_list:
            try:
                exchange_class = getattr(ccxt, exchange_name)
                # Special configuration for problematic exchanges
                exchange_config = {
                    'enableRateLimit': True,
                    'timeout': 45000,  # 45 second timeout
                    'rateLimit': 2000,  # More conservative rate limiting
                    'options': {
                        'defaultType': 'spot',
                        'adjustForTimeDifference': True
                    }
                }
                
                # KuCoin needs special handling
                if exchange_name == 'kucoin':
                    exchange_config['options']['adjustForTimeDifference'] = True
                # CoinEx needs special handling
                if exchange_name == 'coinex':
                    exchange_config['options']['defaultType'] = 'spot'
                # LBank needs special handling
                if exchange_name == 'lbank':
                    exchange_config['options']['defaultType'] = 'spot'
                
                self.exchanges[exchange_name] = exchange_class(exchange_config)
            except Exception as e:
                # Silent error - will try to use others
                pass
        
        # Set primary exchange (prefer kucoin, then coinex, then lbank)
        if 'kucoin' in self.exchanges:
            self.primary_exchange = self.exchanges['kucoin']
        elif 'coinex' in self.exchanges:
            self.primary_exchange = self.exchanges['coinex']
        elif 'lbank' in self.exchanges:
            self.primary_exchange = self.exchanges['lbank']
        elif len(self.exchanges) > 0:
            self.primary_exchange = list(self.exchanges.values())[0]
        
        print(f"✅ Initialized {len(self.exchanges)} exchanges: {', '.join(self.exchanges.keys())}")
    
    async def get_all_trading_pairs(self):
        """Get trading pairs from all exchanges with parallel loading and better error handling"""
        all_pairs = set()
        
        async def load_exchange_pairs(exchange_name, exchange):
            """Load pairs from a single exchange with retry"""
            max_retries = 2  # Reduced retries for faster startup
            for attempt in range(max_retries):
                try:
                    print(f"   📊 Loading markets from {exchange_name} (attempt {attempt + 1}/{max_retries})...")
                    markets = await asyncio.wait_for(
                        exchange.load_markets(),
                        timeout=30.0  # 30 second timeout per exchange
                    )
                    usdt_pairs = [s for s in markets.keys() 
                                 if s.endswith('/USDT') and markets[s].get('active', True)]
                    print(f"   ✅ {exchange_name}: {len(usdt_pairs)} USDT pairs loaded")
                    return usdt_pairs
                except asyncio.TimeoutError:
                    if attempt < max_retries - 1:
                        print(f"   ⏱️  {exchange_name} timeout, retrying...")
                        await asyncio.sleep(1)
                    else:
                        print(f"   ❌ {exchange_name}: Timeout - skipping")
                        return []  # Return empty instead of continuing
                except Exception as e:
                    error_msg = str(e)
                    # Skip exchanges with 403 (forbidden) or connection errors immediately
                    if "403" in error_msg or "forbidden" in error_msg.lower():
                        print(f"   ❌ {exchange_name}: Access forbidden (403) - skipping")
                        return []
                    elif "rate limit" in error_msg.lower() or "429" in error_msg:
                        if attempt < max_retries - 1:
                            wait_time = 3
                            print(f"   ⏳ {exchange_name}: Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"   ❌ {exchange_name}: Rate limited - skipping")
                            return []
                    elif attempt < max_retries - 1:
                        print(f"   ⚠️  {exchange_name} error: {error_msg[:50]}... retrying...")
                        await asyncio.sleep(1)
                    else:
                        print(f"   ❌ {exchange_name}: Failed - skipping")
                        return []
            return []
        
        # Load from all exchanges in parallel (but with rate limiting)
        tasks = []
        for exchange_name, exchange in self.exchanges.items():
            task = load_exchange_pairs(exchange_name, exchange)
            tasks.append(task)
            # Small delay between starting tasks to avoid overwhelming
            await asyncio.sleep(0.5)
        
        # Wait for all exchanges to load (with timeout)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all pairs
        for result in results:
            if isinstance(result, list):
                all_pairs.update(result)
            elif isinstance(result, Exception):
                # Already logged in load_exchange_pairs
                pass
        
        print(f"\n✅ Total unique pairs collected: {len(all_pairs)} from {len([r for r in results if isinstance(r, list)])} exchanges")
        return list(all_pairs)
    
    async def get_ticker(self, symbol):
        """Get ticker from any available exchange"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                ticker = await exchange.fetch_ticker(symbol)
                if ticker:
                    return ticker
            except:
                continue
        return None
    
    async def get_ohlcv(self, symbol, timeframe='1m', limit=500):
        """Get OHLCV from any available exchange"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if ohlcv and len(ohlcv) > 0:
                    return ohlcv
            except:
                continue
        return None
    
    async def close_all(self):
        """Close all exchange connections"""
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except:
                pass

