"""
Fundamental Analysis Module
Analyzes news and social sentiment for coins when signals are generated
"""
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import config
import json


class FundamentalAnalyzer:
    """
    Analyzes fundamental factors (news, sentiment) for cryptocurrency signals
    Uses NewsAPI and other sources to provide context for trading signals
    """
    
    def __init__(self):
        self.newsapi_key = getattr(config, 'NEWSAPI_KEY', '57491ef0988749878037caddc77e12c2')
        self.use_newsapi = True if self.newsapi_key else False
        
    def _get_coin_keywords(self, symbol: str) -> List[str]:
        """Get search keywords for a coin symbol"""
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '')
        
        # Common coin name mappings
        coin_names = {
            'BTC': ['bitcoin', 'BTC'],
            'ETH': ['ethereum', 'ETH'],
            'BNB': ['binance coin', 'BNB', 'binance'],
            'SOL': ['solana', 'SOL'],
            'XRP': ['ripple', 'XRP'],
            'ADA': ['cardano', 'ADA'],
            'DOGE': ['dogecoin', 'DOGE', 'doge'],
            'DOT': ['polkadot', 'DOT'],
            'MATIC': ['polygon', 'MATIC', 'polygon network'],
            'AVAX': ['avalanche', 'AVAX'],
            'LINK': ['chainlink', 'LINK'],
            'UNI': ['uniswap', 'UNI'],
            'ATOM': ['cosmos', 'ATOM'],
            'LTC': ['litecoin', 'LTC'],
            'BCH': ['bitcoin cash', 'BCH'],
            'XLM': ['stellar', 'XLM'],
            'ALGO': ['algorand', 'ALGO'],
            'VET': ['vechain', 'VET'],
            'FIL': ['filecoin', 'FIL'],
            'TRX': ['tron', 'TRX'],
            'ETC': ['ethereum classic', 'ETC'],
            'EOS': ['EOS'],
            'AAVE': ['aave', 'AAVE'],
            'MKR': ['maker', 'MKR'],
            'COMP': ['compound', 'COMP'],
            'SUSHI': ['sushi', 'SUSHI'],
            'YFI': ['yearn finance', 'YFI'],
            'SNX': ['synthetix', 'SNX'],
            'CRV': ['curve', 'CRV'],
            '1INCH': ['1inch', '1INCH'],
            'BAL': ['balancer', 'BAL'],
            'ZRX': ['0x', 'ZRX'],
            'BAT': ['basic attention token', 'BAT'],
            'ZEC': ['zcash', 'ZEC'],
            'DASH': ['dash', 'DASH'],
            'XMR': ['monero', 'XMR'],
            'WAVES': ['waves', 'WAVES'],
            'QTUM': ['qtum', 'QTUM'],
            'ZIL': ['zilliqa', 'ZIL'],
            'ONT': ['ontology', 'ONT'],
            'IOST': ['iostoken', 'IOST'],
            'THETA': ['theta', 'THETA'],
            'ENJ': ['enjin', 'ENJ'],
            'MANA': ['decentraland', 'MANA'],
            'SAND': ['sandbox', 'SAND'],
            'AXS': ['axie infinity', 'AXS'],
            'CHZ': ['chiliz', 'CHZ'],
            'FLOW': ['flow', 'FLOW'],
            'NEAR': ['near protocol', 'NEAR'],
            'FTM': ['fantom', 'FTM'],
            'HBAR': ['hedera', 'HBAR'],
            'EGLD': ['elrond', 'EGLD'],
            'ICP': ['internet computer', 'ICP'],
            'APT': ['aptos', 'APT'],
            'SUI': ['sui', 'SUI'],
            'OP': ['optimism', 'OP'],
            'ARB': ['arbitrum', 'ARB'],
            'INJ': ['injective', 'INJ'],
            'TIA': ['celestia', 'TIA'],
            'SEI': ['sei', 'SEI'],
            'JUP': ['jupiter', 'JUP'],
            'WLD': ['worldcoin', 'WLD'],
            'PIXEL': ['pixels', 'PIXEL'],
            'PORTAL': ['portal', 'PORTAL'],
            'AI': ['ai coin', 'AI'],
            'XAI': ['xai', 'XAI'],
            'ACE': ['fusionist', 'ACE'],
            'NFP': ['nfp', 'NFP'],
            'MANTA': ['manta', 'MANTA'],
            'ALT': ['altlayer', 'ALT'],
            'ARKM': ['arkham', 'ARKM'],
            'DYM': ['dymension', 'DYM'],
            'PDA': ['pda', 'PDA'],
            'BANANA': ['banana', 'BANANA'],
            'WIN': ['wink', 'WIN', 'tronbet'],
            'MDT': ['measurable data token', 'MDT'],
        }
        
        if base_symbol in coin_names:
            return coin_names[base_symbol]
        else:
            # Default: use symbol name
            return [base_symbol.lower(), base_symbol]
    
    async def get_news_articles(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get news articles for a coin from NewsAPI"""
        if not self.use_newsapi:
            return []
        
        keywords = self._get_coin_keywords(symbol)
        all_articles = []
        
        # Search for each keyword
        for keyword in keywords[:2]:  # Limit to 2 keywords to avoid rate limits
            try:
                # NewsAPI endpoint
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': f"{keyword} cryptocurrency OR {keyword} crypto",
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': self.newsapi_key,
                    'from': (datetime.now() - timedelta(hours=hours)).isoformat()
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=10) as response:
                        if response.status == 200:
                            data = await response.json()
                            articles = data.get('articles', [])
                            
                            # Filter and format articles
                            for article in articles:
                                if article.get('title') and article.get('description'):
                                    all_articles.append({
                                        'title': article.get('title', ''),
                                        'description': article.get('description', ''),
                                        'url': article.get('url', ''),
                                        'source': article.get('source', {}).get('name', 'Unknown'),
                                        'publishedAt': article.get('publishedAt', ''),
                                        'relevance': self._calculate_relevance(article, symbol)
                                    })
                        elif response.status == 429:
                            # Rate limited - wait a bit
                            await asyncio.sleep(2)
                        await asyncio.sleep(0.5)  # Rate limiting
            except Exception as e:
                # Silent error
                pass
        
        # Remove duplicates and sort by relevance
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_lower = article['title'].lower()
            if title_lower not in seen_titles:
                seen_titles.add(title_lower)
                unique_articles.append(article)
        
        # Sort by relevance (most relevant first)
        unique_articles.sort(key=lambda x: x['relevance'], reverse=True)
        
        return unique_articles[:5]  # Return top 5 most relevant
    
    def _calculate_relevance(self, article: Dict, symbol: str) -> float:
        """Calculate relevance score for an article (0-1)"""
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '').upper()
        title = article.get('title', '').lower()
        description = article.get('description', '').lower()
        text = title + ' ' + description
        
        score = 0.0
        
        # Exact symbol match
        if base_symbol.lower() in text:
            score += 0.5
        
        # Keywords that indicate high relevance
        high_relevance_keywords = ['listing', 'partnership', 'launch', 'upgrade', 'hack', 'regulation', 
                                  'adoption', 'integration', 'announcement', 'breakthrough']
        for keyword in high_relevance_keywords:
            if keyword in text:
                score += 0.1
        
        # Negative indicators (less relevant)
        if 'bitcoin' in text and base_symbol != 'BTC':
            score -= 0.2  # Generic bitcoin news
        
        return min(1.0, max(0.0, score))
    
    def _analyze_sentiment(self, articles: List[Dict]) -> Dict:
        """Simple sentiment analysis based on keywords"""
        if not articles:
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
        
        positive_keywords = ['bullish', 'surge', 'rally', 'gain', 'up', 'rise', 'growth', 
                            'partnership', 'adoption', 'launch', 'upgrade', 'breakthrough',
                            'positive', 'success', 'win', 'profit', 'increase']
        negative_keywords = ['bearish', 'crash', 'drop', 'fall', 'down', 'decline', 'loss',
                            'hack', 'scam', 'regulation', 'ban', 'warning', 'risk',
                            'negative', 'failure', 'decrease', 'dump']
        
        total_score = 0.0
        total_words = 0
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            words = text.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_keywords:
                    total_score += 1.0
                elif word in negative_keywords:
                    total_score -= 1.0
        
        if total_words == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = total_score / total_words * 10  # Normalize
        
        # Determine sentiment
        if sentiment_score > 0.3:
            sentiment = 'positive'
        elif sentiment_score < -0.3:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        confidence = min(1.0, abs(sentiment_score) / 2.0)
        
        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': confidence
        }
    
    def _assess_impact(self, articles: List[Dict], sentiment: Dict) -> Dict:
        """Assess potential market impact of news"""
        if not articles:
            return {
                'impact_level': 'unknown',
                'impact_score': 0.0,
                'reasoning': 'No recent news found'
            }
        
        # Count high-impact keywords
        high_impact_keywords = ['listing', 'partnership', 'hack', 'regulation', 'launch', 
                               'upgrade', 'mainnet', 'burn', 'buyback']
        medium_impact_keywords = ['announcement', 'integration', 'adoption', 'collaboration']
        
        high_impact_count = 0
        medium_impact_count = 0
        
        for article in articles:
            text = (article.get('title', '') + ' ' + article.get('description', '')).lower()
            for keyword in high_impact_keywords:
                if keyword in text:
                    high_impact_count += 1
                    break
            for keyword in medium_impact_keywords:
                if keyword in text:
                    medium_impact_count += 1
                    break
        
        # Calculate impact score
        impact_score = (high_impact_count * 2.0) + (medium_impact_count * 1.0)
        impact_score += abs(sentiment['score']) * 0.5  # Sentiment contributes
        
        # Determine impact level
        if impact_score >= 3.0:
            impact_level = 'high'
        elif impact_score >= 1.5:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        # Generate reasoning
        reasoning_parts = []
        if high_impact_count > 0:
            reasoning_parts.append(f"{high_impact_count} high-impact news items")
        if sentiment['sentiment'] != 'neutral':
            reasoning_parts.append(f"{sentiment['sentiment']} sentiment")
        if len(articles) > 3:
            reasoning_parts.append(f"{len(articles)} recent articles")
        
        reasoning = ', '.join(reasoning_parts) if reasoning_parts else 'Limited news activity'
        
        return {
            'impact_level': impact_level,
            'impact_score': impact_score,
            'reasoning': reasoning
        }
    
    async def analyze_fundamentals(self, symbol: str, signal_type: str = 'PUMP') -> Dict:
        """
        Complete fundamental analysis for a coin signal
        Returns analysis dict with news, sentiment, and impact assessment
        """
        print(f"📰 Analyzing fundamentals for {symbol}...")
        
        # Get news articles
        articles = await self.get_news_articles(symbol, hours=24)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(articles)
        
        # Assess impact
        impact = self._assess_impact(articles, sentiment)
        
        # Generate AI-style summary
        summary = self._generate_summary(symbol, articles, sentiment, impact, signal_type)
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'articles': articles,
            'sentiment': sentiment,
            'impact': impact,
            'summary': summary,
            'timestamp': datetime.now()
        }
    
    def _generate_summary(self, symbol: str, articles: List[Dict], sentiment: Dict, 
                         impact: Dict, signal_type: str) -> str:
        """Generate human-readable summary of fundamental analysis (HTML format for Telegram)"""
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '')
        
        if not articles:
            return f"⚠️ No recent news found for {base_symbol}. Technical analysis only."
        
        summary_parts = []
        
        # Sentiment summary (HTML format)
        if sentiment['sentiment'] == 'positive':
            summary_parts.append(f"📈 Recent news sentiment is <b>positive</b> (confidence: {sentiment['confidence']:.0%})")
        elif sentiment['sentiment'] == 'negative':
            summary_parts.append(f"📉 Recent news sentiment is <b>negative</b> (confidence: {sentiment['confidence']:.0%})")
        else:
            summary_parts.append(f"➡️ Recent news sentiment is <b>neutral</b>")
        
        # Impact summary (HTML format)
        if impact['impact_level'] == 'high':
            summary_parts.append(f"⚠️ <b>High impact</b> news detected: {impact['reasoning']}")
        elif impact['impact_level'] == 'medium':
            summary_parts.append(f"📊 <b>Medium impact</b> news: {impact['reasoning']}")
        else:
            summary_parts.append(f"ℹ️ <b>Low impact</b> news activity")
        
        # Key news highlights
        if articles:
            top_article = articles[0]
            summary_parts.append(f"\n🔍 <b>Key News:</b> {top_article['title'][:100]}...")
            summary_parts.append(f"   Source: {top_article['source']}")
        
        # Signal alignment
        if signal_type == 'PUMP' and sentiment['sentiment'] == 'positive':
            summary_parts.append(f"\n✅ <b>Alignment:</b> Positive news supports PUMP signal")
        elif signal_type == 'DUMP' and sentiment['sentiment'] == 'negative':
            summary_parts.append(f"\n✅ <b>Alignment:</b> Negative news supports DUMP signal")
        elif signal_type == 'PUMP' and sentiment['sentiment'] == 'negative':
            summary_parts.append(f"\n⚠️ <b>Warning:</b> Negative news contradicts PUMP signal - be cautious")
        elif signal_type == 'DUMP' and sentiment['sentiment'] == 'positive':
            summary_parts.append(f"\n⚠️ <b>Warning:</b> Positive news contradicts DUMP signal - be cautious")
        
        return '\n'.join(summary_parts)

