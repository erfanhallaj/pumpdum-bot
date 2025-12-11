"""
Market Data API Integration
Uses CoinGecko and CoinMarketCap APIs for more accurate market data
"""
import aiohttp
import asyncio
from typing import Dict, Optional, List
import config


class MarketDataAPI:
    """
    Fetches accurate market data from CoinGecko and CoinMarketCap APIs
    """
    
    def __init__(self):
        self.coingecko_key = getattr(config, 'COINGECKO_API_KEY', '')
        self.cmc_key = getattr(config, 'COINMARKETCAP_API_KEY', '')
        self.use_apis = getattr(config, 'USE_MARKET_DATA_APIS', True)
        
        # Rate limiting
        self.coingecko_last_call = 0
        self.cmc_last_call = 0
        self.min_interval = 1.0  # 1 second between calls
    
    def _get_base_symbol(self, symbol: str) -> str:
        """Convert exchange symbol to base symbol (e.g., BTC/USDT -> bitcoin)"""
        base = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '').upper()
        
        # Common symbol mappings
        symbol_map = {
            'BTC': 'bitcoin',
            'ETH': 'ethereum',
            'BNB': 'binancecoin',
            'SOL': 'solana',
            'XRP': 'ripple',
            'ADA': 'cardano',
            'DOGE': 'dogecoin',
            'DOT': 'polkadot',
            'MATIC': 'matic-network',
            'AVAX': 'avalanche-2',
            'LINK': 'chainlink',
            'UNI': 'uniswap',
            'ATOM': 'cosmos',
            'LTC': 'litecoin',
            'BCH': 'bitcoin-cash',
            'XLM': 'stellar',
            'ALGO': 'algorand',
            'VET': 'vechain',
            'FIL': 'filecoin',
            'TRX': 'tron',
            'ETC': 'ethereum-classic',
            'EOS': 'eos',
            'AAVE': 'aave',
            'MKR': 'maker',
            'COMP': 'compound-governance-token',
            'SUSHI': 'sushi',
            'YFI': 'yearn-finance',
            'SNX': 'havven',
            'CRV': 'curve-dao-token',
            '1INCH': '1inch',
            'BAL': 'balancer',
            'ZRX': '0x',
            'BAT': 'basic-attention-token',
            'ZEC': 'zcash',
            'DASH': 'dash',
            'XMR': 'monero',
            'WAVES': 'waves',
            'QTUM': 'qtum',
            'ZIL': 'zilliqa',
            'ONT': 'ontology',
            'IOST': 'iostoken',
            'THETA': 'theta-token',
            'ENJ': 'enjincoin',
            'MANA': 'decentraland',
            'SAND': 'the-sandbox',
            'AXS': 'axie-infinity',
            'CHZ': 'chiliz',
            'FLOW': 'flow',
            'NEAR': 'near',
            'FTM': 'fantom',
            'HBAR': 'hedera-hashgraph',
            'EGLD': 'elrond-erd-2',
            'ICP': 'internet-computer',
            'APT': 'aptos',
            'SUI': 'sui',
            'OP': 'optimism',
            'ARB': 'arbitrum',
            'INJ': 'injective-protocol',
            'TIA': 'celestia',
            'SEI': 'sei-network',
            'JUP': 'jupiter-exchange-solana',
            'WLD': 'worldcoin-wld',
            'PIXEL': 'pixels',
            'PORTAL': 'portal',
            'AI': 'sleepless-ai',
            'XAI': 'xai-blockchain',
            'ACE': 'fusionist',
            'NFP': 'nfp',
            'MANTA': 'manta-network',
            'ALT': 'altlayer',
            'ARKM': 'arkham',
            'DYM': 'dymension',
            'PDA': 'pda',
            'BANANA': 'banana',
        }
        
        # Try direct mapping first
        if base in symbol_map:
            return symbol_map[base]
        
        # Try lowercase
        return base.lower()
    
    async def get_coingecko_data(self, symbol: str) -> Optional[Dict]:
        """Get coin data from CoinGecko API"""
        if not self.use_apis or not self.coingecko_key:
            return None
        
        try:
            base_symbol = self._get_base_symbol(symbol)
            
            # Rate limiting
            await asyncio.sleep(max(0, self.min_interval - (asyncio.get_event_loop().time() - self.coingecko_last_call)))
            
            url = f"https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': base_symbol,
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true',
                'include_market_cap': 'true',
                'x_cg_demo_api_key': self.coingecko_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if base_symbol in data:
                            coin_data = data[base_symbol]
                            self.coingecko_last_call = asyncio.get_event_loop().time()
                            return {
                                'price': coin_data.get('usd', 0),
                                'volume_24h': coin_data.get('usd_24h_vol', 0),
                                'market_cap': coin_data.get('usd_market_cap', 0),
                                'change_24h': coin_data.get('usd_24h_change', 0) / 100 if coin_data.get('usd_24h_change') else 0
                            }
        except Exception as e:
            # Silent error - fallback to exchange data
            pass
        
        return None
    
    async def get_coinmarketcap_data(self, symbol: str) -> Optional[Dict]:
        """Get coin data from CoinMarketCap API"""
        if not self.use_apis or not self.cmc_key:
            return None
        
        try:
            base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '')
            
            # Rate limiting
            await asyncio.sleep(max(0, self.min_interval - (asyncio.get_event_loop().time() - self.cmc_last_call)))
            
            url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
            headers = {
                'X-CMC_PRO_API_KEY': self.cmc_key,
                'Accept': 'application/json'
            }
            params = {
                'symbol': base_symbol,
                'convert': 'USD'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and base_symbol in data['data']:
                            coin_data = data['data'][base_symbol]['quote']['USD']
                            self.cmc_last_call = asyncio.get_event_loop().time()
                            return {
                                'price': coin_data.get('price', 0),
                                'volume_24h': coin_data.get('volume_24h', 0),
                                'market_cap': coin_data.get('market_cap', 0),
                                'change_24h': coin_data.get('percent_change_24h', 0) / 100 if coin_data.get('percent_change_24h') else 0
                            }
        except Exception as e:
            # Silent error - fallback to exchange data
            pass
        
        return None
    
    async def get_enhanced_market_data(self, symbol: str) -> Optional[Dict]:
        """
        Get enhanced market data from APIs (tries CoinGecko first, then CoinMarketCap)
        Returns None if APIs unavailable (will fallback to exchange data)
        """
        if not self.use_apis:
            return None
        
        # Try CoinGecko first (usually more reliable for altcoins)
        data = await self.get_coingecko_data(symbol)
        if data and data.get('volume_24h', 0) > 0:
            return data
        
        # Fallback to CoinMarketCap
        data = await self.get_coinmarketcap_data(symbol)
        if data and data.get('volume_24h', 0) > 0:
            return data
        
        return None
    
    async def get_multiple_coins_data(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get market data for multiple coins in parallel (with rate limiting)
        """
        results = {}
        
        # Process in small batches to respect rate limits
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            tasks = [self.get_enhanced_market_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, data in zip(batch, batch_results):
                if isinstance(data, dict) and data:
                    results[symbol] = data
            
            # Small delay between batches
            if i + batch_size < len(symbols):
                await asyncio.sleep(0.5)
        
        return results

