"""
Telegram Bot for sending alerts and reports
"""
from telegram import Bot
from telegram.error import TelegramError
import asyncio
import config
from datetime import datetime
from exchange_info import ExchangeInfo
from fundamental_analyzer import FundamentalAnalyzer

class TelegramNotifier:
    def __init__(self):
        self.bot = None
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.exchange_info = ExchangeInfo()
        self.fundamental_analyzer = FundamentalAnalyzer()
        self.initialize_bot()
    
    def initialize_bot(self):
        """Initialize Telegram bot"""
        if config.TELEGRAM_BOT_TOKEN:
            try:
                self.bot = Bot(token=config.TELEGRAM_BOT_TOKEN)
                print("Telegram bot initialized successfully")
            except Exception as e:
                print(f"Error initializing Telegram bot: {e}")
        else:
            print("⚠️  Telegram bot token not configured")
    
    async def send_message(self, message, parse_mode='HTML'):
        """Send message to Telegram"""
        if not self.bot or not self.chat_id:
            print("⚠️  Telegram not configured, message not sent:")
            print(message)
            return False
        
        try:
            # Ensure chat_id starts with @ if it's a username
            chat_id = self.chat_id
            if not chat_id.startswith('@') and not chat_id.lstrip('-').isdigit():
                chat_id = '@' + chat_id.lstrip('@')
            
            await self.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )
            return True
        except TelegramError as e:
            error_msg = str(e)
            if "Chat not found" in error_msg:
                print(f"⚠️  Telegram Chat Error: Make sure the bot is started and you've sent a message to @pythontrade_ai")
                print(f"   Or check if the chat_id '{self.chat_id}' is correct")
            else:
                print(f"Error sending Telegram message: {e}")
            return False
    
    def format_pump_alert(self, analysis):
        """Format trading signal alert (PUMP or DUMP) - Clean and Beautiful"""
        symbol = analysis['symbol']
        price = analysis['current_price']
        signal_type = analysis.get('signal_type', 'PUMP')  # PUMP or DUMP
        signal_prob = analysis.get('signal_probability', analysis.get('pump_probability', 0))
        confidence = analysis.get('confidence', 0)
        
        # Get signal score
        signal_score = analysis.get('signal_score', 0.0)
        is_premium = analysis.get('is_premium_signal', False)
        
        # Get trading levels - Enhanced with optimal entry/exit
        entry = analysis.get('entry', price)
        exit1 = analysis.get('exit1', price * 1.10)
        exit2 = analysis.get('exit2', price * 1.20)
        exit3 = analysis.get('exit3', price * 1.30)
        sl = analysis.get('stop_loss', price * 0.93)
        
        # Calculate percentages
        exit1_pct = ((exit1 - entry) / entry) * 100
        exit2_pct = ((exit2 - entry) / entry) * 100
        exit3_pct = ((exit3 - entry) / entry) * 100
        sl_pct = ((sl - entry) / entry) * 100
        
        # Emoji and title based on signal type and score
        if is_premium:
            if signal_type == 'PUMP':
                emoji = "⭐⭐🚀"
                title = "⭐⭐ PREMIUM PUMP SIGNAL ⭐⭐"
                action = "BUY"
            else:
                emoji = "⭐⭐📉"
                title = "⭐⭐ PREMIUM DUMP SIGNAL ⭐⭐"
                action = "SELL/SHORT"
        else:
            if signal_type == 'PUMP':
                emoji = "🚀"
                title = "🚀 PUMP SIGNAL 🚀"
                action = "BUY"
            else:
                emoji = "📉"
                title = "📉 DUMP SIGNAL 📉"
                action = "SELL/SHORT"
        
        # Get exchange links for LBank, CoinEx, KuCoin
        exchange_links = self.exchange_info.get_exchange_links(symbol)
        base_symbol = symbol.replace('/USDT', '').replace('/USD', '').replace('/BTC', '')
        
        # Calculate risk/reward ratio
        risk = abs(entry - sl) / entry * 100
        reward = abs(exit1_pct)
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Format price nicely
        if price >= 1:
            price_str = f"${price:,.2f}"
        elif price >= 0.01:
            price_str = f"${price:,.4f}"
        else:
            price_str = f"${price:,.8f}"
        
        # Premium signal header
        premium_header = ""
        if is_premium:
            premium_header = f"⭐ <b>SIGNAL SCORE: {signal_score:.1f}%</b> ⭐\n🚀 <b>PREMIUM SIGNAL - FAST TRACK</b> 🚀\n\n"
        
        # Exchange purchase links (only LBank, CoinEx, KuCoin)
        exchange_links_text = "🛒 <b>خرید از صرافی‌ها / Buy on Exchanges:</b>\n\n"
        if 'kucoin' in exchange_links:
            exchange_links_text += f"• <a href='{exchange_links['kucoin']}'>🔵 KuCoin - خرید / Buy</a>\n"
        if 'lbank' in exchange_links:
            exchange_links_text += f"• <a href='{exchange_links['lbank']}'>🟢 LBank - خرید / Buy</a>\n"
        if 'coinex' in exchange_links:
            exchange_links_text += f"• <a href='{exchange_links['coinex']}'>🟡 CoinEx - خرید / Buy</a>\n"
        
        message = f"""
{emoji} <b>{title}</b> {emoji}

{premium_header}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>Coin:</b> <code>{symbol}</code>
💵 <b>Current Price:</b> <code>{price_str}</code>
⭐ <b>Signal Score:</b> <b>{signal_score:.1f}%</b>
📈 <b>Probability:</b> <b>{signal_prob:.1%}</b>
🎯 <b>AI Confidence:</b> <b>{confidence:.1%}</b>
📊 <b>10m Change:</b> {analysis.get('price_change_10m', 0):+.2%}
💹 <b>24h Volume:</b> ${analysis.get('volume_24h', 0):,.0f}
📈 <b>Volume Change:</b> {analysis.get('volume_change', 1.0):.2f}x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>TRADING LEVELS:</b>

🎯 <b>Entry Point:</b> <code>${entry:,.8f}</code>

✅ <b>Take Profit:</b>
   • <b>TP1:</b> <code>${exit1:,.8f}</code> <b>({exit1_pct:+.1f}%)</b>
   • <b>TP2:</b> <code>${exit2:,.8f}</code> <b>({exit2_pct:+.1f}%)</b>
   • <b>TP3:</b> <code>${exit3:,.8f}</code> <b>({exit3_pct:+.1f}%)</b>

🛑 <b>Stop Loss:</b> <code>${sl:,.8f}</code> <b>({sl_pct:+.1f}%)</b>

💡 <b>Risk/Reward:</b> <b>1:{rr_ratio:.1f}</b>
📉 <b>Risk:</b> {risk:.1f}% | <b>Reward:</b> {reward:.1f}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 <b>Action:</b> <b>{action}</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{exchange_links_text}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ <b>Time:</b> {analysis['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
"""
        return message
    
    async def send_pump_alert(self, analysis):
        """Send pump detection alert"""
        message = self.format_pump_alert(analysis)
        return await self.send_message(message)
    
    async def send_backtest_report(self, report):
        """Send backtest report"""
        message = f"<pre>{report}</pre>"
        return await self.send_message(message)
    
    async def send_status_update(self, status):
        """Send status update"""
        message = f"""
🤖 <b>Bot Status Update</b>

{status}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return await self.send_message(message)
    
    async def send_optimization_report(self, optimization_results):
        """Send optimization report"""
        message = f"""
🔧 <b>Self-Optimization Complete</b>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 <b>Optimization Results:</b>

✅ Optimized Threshold: {optimization_results['optimized_threshold']:.2f}
📈 Improved Win Rate: {optimization_results['improved_win_rate']:.2%}
📉 Original Win Rate: {optimization_results['original_win_rate']:.2%}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return await self.send_message(message)

    async def send_daily_performance(self, stats, extra_info: str = ""):
        """Send daily performance summary for live signals"""
        message = f"""
📊 <b>Daily Signal Performance</b>

📅 Date: {stats.get('date')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 Total Signals: {stats.get('total_signals', 0)}
🚀 Pump Signals: {stats.get('pump_signals', 0)}
📉 Dump Signals: {stats.get('dump_signals', 0)}

✅ Wins: {stats.get('wins', 0)}
❌ Losses: {stats.get('losses', 0)}
⏳ Timeouts: {stats.get('timeouts', 0)}
🎯 Win Rate: {stats.get('win_rate', 0.0):.2%}

{extra_info}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return await self.send_message(message)

    async def send_signal_outcome(self, signal):
        """
        Send confirmation when a signal is closed (TP/SL/Timeout).
        `signal` is a TrackedSignal instance from signal_tracker.
        """
        status = getattr(signal, "status", "open")
        symbol = getattr(signal, "symbol", "?")
        s_type = getattr(signal, "signal_type", "PUMP")
        entry = float(getattr(signal, "entry", 0.0) or 0.0)
        stop_loss = float(getattr(signal, "stop_loss", 0.0) or 0.0)
        close_price = getattr(signal, "close_price", None)
        hit_target = getattr(signal, "hit_target", None)
        opened_at = getattr(signal, "timestamp", datetime.now())
        closed_at = getattr(signal, "close_time", datetime.now())

        if not entry or close_price is None:
            return False

        # Calculate PnL percentage depending on signal type
        if s_type == "PUMP":
            pnl_pct = (close_price - entry) / entry * 100
            risk_pct = (entry - stop_loss) / entry * 100 if stop_loss else 0.0
        else:
            pnl_pct = (entry - close_price) / entry * 100
            risk_pct = (stop_loss - entry) / entry * 100 if stop_loss else 0.0

        rr = pnl_pct / risk_pct if risk_pct > 0 else 0.0

        if status == "win":
            emoji = "✅"
            title = "SIGNAL WIN"
        elif status == "loss":
            emoji = "❌"
            title = "SIGNAL LOSS"
        else:
            emoji = "⏳"
            title = "SIGNAL TIMEOUT"

        target_text = ""
        if status == "win" and hit_target:
            target_text = f"🎯 Hit TP{hit_target}"
        elif status == "loss":
            target_text = "🛑 Stopped Out (SL)"
        elif status == "timeout":
            target_text = "⏳ Timed Out (no TP/SL hit in time window)"

        message = f"""
{emoji} <b>{title}</b> {emoji}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 <b>Coin:</b> <code>{symbol}</code>
📊 <b>Type:</b> {s_type}
📈 <b>Entry:</b> <code>${entry:,.8f}</code>
📉 <b>Close:</b> <code>${close_price:,.8f}</code>
🛑 <b>Stop Loss:</b> <code>${stop_loss:,.8f}</code>

{target_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📈 <b>PnL:</b> {pnl_pct:+.2f}%
💡 <b>Risk:</b> {risk_pct:.2f}% | <b>R Multiple:</b> {rr:.2f}

⏱️ <b>Opened:</b> {opened_at.strftime('%Y-%m-%d %H:%M:%S')}
⏱️ <b>Closed:</b> {closed_at.strftime('%Y-%m-%d %H:%M:%S')}
🆔 <b>Signal ID:</b> {getattr(signal, "id", 0)}
"""
        return await self.send_message(message)

