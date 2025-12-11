"""
Exchange information and links for coins
"""
import aiohttp
import asyncio
import config


class ExchangeInfo:
    def __init__(self):
        # ONLY LBank, CoinEx, and KuCoin exchanges (as requested)
        self.spot_exchanges = {
            'lbank': {
                'url': 'https://www.lbank.com/trade/',
                'spot': True,
                'futures': True
            },
            'coinex': {
                'url': 'https://www.coinex.com/trade/',
                'spot': True,
                'futures': True
            },
            'kucoin': {
                'url': 'https://www.kucoin.com/trade/',
                'spot': True,
                'futures': True
            },
        }

        # Legacy support
        self.exchange_urls = {k: v['url'] for k, v in self.spot_exchanges.items()}

    def get_exchange_links(self, symbol):
        """Get exchange links for a symbol - ONLY LBank, CoinEx, KuCoin"""
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '')
        links = {}

        for exchange, base_url in self.exchange_urls.items():
            if exchange == 'kucoin':
                # KuCoin format: RZTO-USDT (uppercase, dash)
                links[exchange] = f"{base_url}{base_symbol.upper()}-USDT"
            elif exchange == 'lbank':
                # LBank format: RZTO_USDT (uppercase, underscore)
                links[exchange] = f"{base_url}{base_symbol.upper()}_USDT"
            elif exchange == 'coinex':
                # CoinEx format: RZTO_USDT (uppercase, underscore)
                links[exchange] = f"{base_url}{base_symbol.upper()}_USDT"

        return links

    def get_dex_links(self, symbol):
        """Get DEX and pool links"""
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '')

        return {
            'uniswap': f"https://app.uniswap.org/#/tokens/ethereum/{base_symbol}",
            'pancakeswap': f"https://pancakeswap.finance/swap?outputCurrency={base_symbol}",
            'jupiter': f"https://jup.ag/swap/USDT-{base_symbol}",
            'raydium': f"https://raydium.io/swap/?inputCurrency=USDT&outputCurrency={base_symbol}",
            # Use DexScreener search so it always resolves, even without exact pair URL
            'dexscreener': f"https://dexscreener.com/search?q={base_symbol}",
            'coingecko': f"https://www.coingecko.com/en/coins/{base_symbol.lower()}",
            'coinmarketcap': f"https://coinmarketcap.com/currencies/{base_symbol.lower()}/",
        }

    def get_explorer_links(self, symbol):
        """
        Basic smart-contract / explorer search links.
        We don't always know the exact on-chain contract, but these searches
        help you quickly find it on popular EVM explorers.
        """
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '')
        return {
            'etherscan': f"https://etherscan.io/search?q={base_symbol}",
            'bscscan': f"https://bscscan.com/search?q={base_symbol}",
            'polygonscan': f"https://polygonscan.com/search?q={base_symbol}",
            'arbiscan': f"https://arbiscan.io/search?q={base_symbol}",
        }

    def get_tradingview_link(self, symbol):
        """
        Build a TradingView chart link for the main exchange and USDT pair.
        Example: BINANCE:BTCUSDT
        """
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '')
        tv_exchange_map = {
            'kucoin': 'KUCOIN',
            'lbank': 'LBANK',
            'coinex': 'COINEX',
        }
        exchange_key = getattr(config, "EXCHANGE_NAME", "kucoin").lower()
        tv_prefix = tv_exchange_map.get(exchange_key, 'KUCOIN')
        # Format: BINANCE:RZTOUSDT (no slash, uppercase)
        pair_symbol = f"{base_symbol}USDT"
        return f"https://www.tradingview.com/chart/?symbol={tv_prefix}:{pair_symbol}"

    def format_exchange_list(self, symbol):
        """Format exchange list with Spot/Futures info - Enhanced version"""
        exchange_links = self.get_exchange_links(symbol)
        dex_links = self.get_dex_links(symbol)
        explorer_links = self.get_explorer_links(symbol)
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '')

        text = "📊 <b>Available Exchanges:</b>\n\n"

        # Separate Spot and Futures
        spot_exchanges = []
        futures_exchanges = []

        for exchange, info in self.spot_exchanges.items():
            if exchange in exchange_links:
                url = exchange_links[exchange]
                if info['spot']:
                    spot_exchanges.append((exchange, url))
                if info['futures']:
                    # Generate futures URL (uppercase for most exchanges)
                    if exchange == 'kucoin':
                        futures_url = f"https://www.kucoin.com/futures/{base_symbol.upper()}-USDT"
                    elif exchange == 'lbank':
                        futures_url = f"https://www.lbank.com/futures/{base_symbol.upper()}_USDT"
                    elif exchange == 'coinex':
                        futures_url = f"https://www.coinex.com/futures/{base_symbol.upper()}_USDT"
                    else:
                        futures_url = url

                    futures_exchanges.append((exchange, futures_url))

        # Spot Exchanges
        if spot_exchanges:
            text += "🟢 <b>Spot Trading (Available):</b>\n"
            for exchange, url in spot_exchanges[:10]:
                text += f"• <a href='{url}'>{exchange.upper()}</a>\n"
            if len(spot_exchanges) > 10:
                text += f"   ... and {len(spot_exchanges) - 10} more\n"

        # Futures Exchanges
        if futures_exchanges:
            text += "\n🔵 <b>Futures Trading (Available):</b>\n"
            for exchange, url in futures_exchanges[:10]:
                text += f"• <a href='{url}'>{exchange.upper()}</a>\n"
            if len(futures_exchanges) > 10:
                text += f"   ... and {len(futures_exchanges) - 10} more\n"

        # DEX
        text += "\n🔄 <b>DEX & Pools:</b>\n"
        dex_list = [
            ('Uniswap', dex_links['uniswap']),
            ('PancakeSwap', dex_links['pancakeswap']),
            ('Jupiter', dex_links['jupiter']),
            ('Raydium', dex_links['raydium'])
        ]
        for dex_name, url in dex_list:
            text += f"• <a href='{url}'>{dex_name}</a>\n"

        # Smart contract / explorers
        text += f"\n🧾 <b>Smart Contract & Explorers (search):</b>\n"
        text += f"• <a href='{explorer_links['etherscan']}'>Etherscan</a>\n"
        text += f"• <a href='{explorer_links['bscscan']}'>BscScan</a>\n"
        text += f"• <a href='{explorer_links['polygonscan']}'>PolygonScan</a> | "
        text += f"<a href='{explorer_links['arbiscan']}'>ArbiScan</a>\n"

        # Analytics + TradingView
        text += f"\n📈 <b>Analytics & Charts:</b>\n"
        text += f"• <a href='{dex_links['dexscreener']}'>DexScreener (Search)</a>\n"
        text += f"• <a href='{dex_links['coingecko']}'>CoinGecko</a> | "
        text += f"<a href='{dex_links['coinmarketcap']}'>CoinMarketCap</a>\n"
        # TradingView chart for the configured main exchange
        tv_link = self.get_tradingview_link(symbol)
        text += f"• <a href='{tv_link}'>TradingView Chart</a>\n"

        return text
