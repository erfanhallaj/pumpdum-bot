"""
📝 Advanced Logging System for Bot and Teacher Activities
Logs all important activities for later review
"""
import os
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Optional

class BotLogger:
    """Advanced logger for bot and teacher activities"""
    
    def __init__(self, log_dir='logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files
        self.bot_log_file = self.log_dir / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
        self.teacher_log_file = self.log_dir / f"teacher_{datetime.now().strftime('%Y%m%d')}.log"
        self.signals_log_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.log"
        self.activities_log_file = self.log_dir / f"activities_{datetime.now().strftime('%Y%m%d')}.json"
        
        # Setup loggers
        self.setup_loggers()
        
        # Activities history (JSON format for easy parsing)
        self.activities = []
        
    def setup_loggers(self):
        """Setup logging configuration"""
        # Bot logger
        self.bot_logger = logging.getLogger('bot')
        self.bot_logger.setLevel(logging.INFO)
        bot_handler = logging.FileHandler(self.bot_log_file, encoding='utf-8')
        bot_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.bot_logger.addHandler(bot_handler)
        
        # Teacher logger
        self.teacher_logger = logging.getLogger('teacher')
        self.teacher_logger.setLevel(logging.INFO)
        teacher_handler = logging.FileHandler(self.teacher_log_file, encoding='utf-8')
        teacher_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.teacher_logger.addHandler(teacher_handler)
        
        # Signals logger
        self.signals_logger = logging.getLogger('signals')
        self.signals_logger.setLevel(logging.INFO)
        signals_handler = logging.FileHandler(self.signals_log_file, encoding='utf-8')
        signals_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.signals_logger.addHandler(signals_handler)
        
    def log_bot_activity(self, activity: str, details: Optional[Dict] = None):
        """Log bot activity"""
        message = f"🤖 {activity}"
        if details:
            message += f" | Details: {json.dumps(details, ensure_ascii=False)}"
        self.bot_logger.info(message)
        self._add_activity('bot', activity, details)
    
    def log_teacher_activity(self, activity: str, details: Optional[Dict] = None):
        """Log teacher (Self-Teaching Master) activity"""
        message = f"🎓 {activity}"
        if details:
            message += f" | Details: {json.dumps(details, ensure_ascii=False)}"
        self.teacher_logger.info(message)
        self._add_activity('teacher', activity, details)
    
    def log_signal(self, signal_type: str, symbol: str, details: Optional[Dict] = None):
        """Log trading signal"""
        message = f"🚨 {signal_type} SIGNAL | {symbol}"
        if details:
            message += f" | Score: {details.get('signal_score', 0):.1f}%"
            message += f" | Prob: {details.get('signal_probability', 0):.1%}"
            message += f" | Price: ${details.get('current_price', 0):,.8f}"
        self.signals_logger.info(message)
        self._add_activity('signal', f"{signal_type} - {symbol}", details)
    
    def log_signal_outcome(self, symbol: str, outcome: str, details: Optional[Dict] = None):
        """Log signal outcome (win/loss/timeout)"""
        emoji = "✅" if outcome == "win" else "❌" if outcome == "loss" else "⏳"
        message = f"{emoji} {outcome.upper()} | {symbol}"
        if details:
            message += f" | PnL: {details.get('pnl_pct', 0):+.2f}%"
            message += f" | Entry: ${details.get('entry', 0):,.8f}"
            message += f" | Close: ${details.get('close_price', 0):,.8f}"
        self.signals_logger.info(message)
        self._add_activity('signal_outcome', f"{outcome} - {symbol}", details)
    
    def log_backtest(self, results: Dict):
        """Log backtest results"""
        message = f"📊 BACKTEST RESULTS"
        message += f" | Win Rate: {results.get('win_rate', 0):.2%}"
        message += f" | Signals: {results.get('total_signals', 0)}"
        message += f" | Profit: {results.get('total_profit', 0):.2f} USDT"
        message += f" | Timeouts: {results.get('timeouts', 0)}"
        message += f" | Losses: {results.get('losses', 0)}"
        self.teacher_logger.info(message)
        self._add_activity('backtest', 'Backtest completed', results)
    
    def log_teacher_fix(self, fix_id: str, problem: str, fix_description: str, result: str):
        """Log teacher fix application"""
        message = f"🔧 FIX APPLIED | ID: {fix_id}"
        message += f" | Problem: {problem}"
        message += f" | Fix: {fix_description}"
        message += f" | Result: {result}"
        self.teacher_logger.info(message)
        self._add_activity('teacher_fix', fix_id, {
            'problem': problem,
            'fix': fix_description,
            'result': result
        })
    
    def log_config_change(self, setting: str, old_value: any, new_value: any, reason: str):
        """Log configuration changes"""
        message = f"⚙️ CONFIG CHANGE | {setting}: {old_value} → {new_value}"
        message += f" | Reason: {reason}"
        self.teacher_logger.info(message)
        self._add_activity('config_change', setting, {
            'old_value': str(old_value),
            'new_value': str(new_value),
            'reason': reason
        })
    
    def log_error(self, error_type: str, error_message: str, details: Optional[Dict] = None):
        """Log errors"""
        message = f"❌ ERROR | {error_type}: {error_message}"
        if details:
            message += f" | Details: {json.dumps(details, ensure_ascii=False)}"
        self.bot_logger.error(message)
        # Combine error message with details safely
        error_details = {'message': error_message}
        if details:
            error_details.update(details)
        self._add_activity('error', error_type, error_details)
    
    def log_performance_stats(self, stats: Dict):
        """Log performance statistics"""
        message = f"📈 PERFORMANCE STATS"
        message += f" | Total Signals: {stats.get('total_signals', 0)}"
        message += f" | Win Rate: {stats.get('win_rate', 0):.2%}"
        message += f" | Total Profit: {stats.get('total_profit', 0):.2f} USDT"
        self.bot_logger.info(message)
        self._add_activity('performance', 'Performance stats', stats)
    
    def _add_activity(self, category: str, activity: str, details: Optional[Dict] = None):
        """Add activity to JSON log"""
        activity_entry = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'activity': activity,
            'details': details or {}
        }
        self.activities.append(activity_entry)
        
        # Save to file periodically (every 10 activities)
        if len(self.activities) % 10 == 0:
            self._save_activities()
    
    def _save_activities(self):
        """Save activities to JSON file"""
        try:
            with open(self.activities_log_file, 'w', encoding='utf-8') as f:
                json.dump(self.activities, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving activities log: {e}")
    
    def get_daily_summary(self) -> Dict:
        """Get daily summary of activities"""
        today = datetime.now().date()
        today_activities = [
            a for a in self.activities 
            if datetime.fromisoformat(a['timestamp']).date() == today
        ]
        
        summary = {
            'date': today.isoformat(),
            'total_activities': len(today_activities),
            'signals': len([a for a in today_activities if a['category'] == 'signal']),
            'signal_outcomes': {
                'wins': len([a for a in today_activities if a.get('activity', '').startswith('win')]),
                'losses': len([a for a in today_activities if a.get('activity', '').startswith('loss')]),
                'timeouts': len([a for a in today_activities if a.get('activity', '').startswith('timeout')])
            },
            'teacher_fixes': len([a for a in today_activities if a['category'] == 'teacher_fix']),
            'backtests': len([a for a in today_activities if a['category'] == 'backtest']),
            'config_changes': len([a for a in today_activities if a['category'] == 'config_change']),
            'errors': len([a for a in today_activities if a['category'] == 'error'])
        }
        
        return summary
    
    def close(self):
        """Close loggers and save final activities"""
        self._save_activities()
        for handler in self.bot_logger.handlers + self.teacher_logger.handlers + self.signals_logger.handlers:
            handler.close()

