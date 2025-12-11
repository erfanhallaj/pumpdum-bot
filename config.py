"""
Configuration file for the AI Pump Detection Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '7804159113:AAGeTdxOEXuhxibCcZqjtfFiuQ_jjmIQIVU')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '@pythontrade_ai')

# Exchange Configuration - ONLY LBank, CoinEx, and KuCoin
EXCHANGE_NAME = os.getenv('EXCHANGE_NAME', 'kucoin')  # Default to KuCoin
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY', '')
EXCHANGE_API_SECRET = os.getenv('EXCHANGE_API_SECRET', '')

# Market Data APIs (for more accurate data)
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY', 'CG-EYFYZQqnJwvLAuuxkEkX8R9w')
COINMARKETCAP_API_KEY = os.getenv('COINMARKETCAP_API_KEY', '3b3a33e157dd48ffb61c65e97d4416c2')
USE_MARKET_DATA_APIS = True  # Enable CoinGecko/CoinMarketCap for better data

# News & Fundamental Analysis
NEWSAPI_KEY = os.getenv('NEWSAPI_KEY', '57491ef0988749878037caddc77e12c2')
ENABLE_FUNDAMENTAL_ANALYSIS = True  # Analyze news when signals are generated

# Self-Learning Strategy Optimization
ENABLE_STRATEGY_OPTIMIZATION = True  # Automatically test and optimize strategies
STRATEGY_OPTIMIZATION_INTERVAL = 86400  # Optimize strategies once per day (24 hours)

# Trading Configuration - Smart Pump Filter
MIN_VOLUME_24H = 10000        # Minimum 24h volume in USDT (reduced to include more coins)
MAX_VOLUME_24H = 10000000000  # Maximum 24h volume (10B - include all large caps)
# NO PRICE FILTER - BTC, ETH, and all coins can be analyzed
MIN_PRICE_CHANGE_THRESHOLD = 0.05  # 5% minimum price change to trigger
PUMP_DETECTION_WINDOW = 600  # 10 minutes in seconds
MONITORING_INTERVAL = 10      # Check every 10 seconds
MAX_COINS_TO_MONITOR = 200    # Monitor fewer coins but with high pump potential
MIN_MARKET_CAP = 0  # Minimum market cap (0 = no limit)
MAX_MARKET_CAP = 999999999999  # Maximum market cap (essentially no limit - include all coins)

# Smart Pump Filter - فیلتر هوشمند برای پیدا کردن کوین‌های نزدیک به پامپ
ENABLE_SMART_PUMP_FILTER = True  # فعال کردن فیلتر هوشمند
SMART_FILTER_MIN_PRICE_CHANGE = 0.03  # حداقل 3% تغییر قیمت در 10 دقیقه
SMART_FILTER_MIN_VOLUME_SPIKE = 1.3  # حداقل 1.3x اسپایک حجم
SMART_FILTER_MIN_MOMENTUM = 0.02  # حداقل 2% momentum (روند صعودی)
SMART_FILTER_MAX_COINS = 200  # حداکثر 200 کوین با پتانسیل بالا

# AI Model Configuration (FASTEST MODE - Maximum signals)
MODEL_UPDATE_INTERVAL = 3600   # Update model every hour
BACKTEST_PERIOD_DAYS = 7       # Reduced for faster backtest
# FASTEST thresholds: maximum signals while maintaining basic quality
MIN_CONFIDENCE_SCORE = 0.45    # Required signal probability (45%+) - FASTEST MODE
MIN_AI_CONFIDENCE = 0.50       # Required AI confidence (50%+) - FASTEST MODE
# Extra filters for signal quality (very relaxed for speed)
SIGNAL_DOMINANCE_MARGIN = 0.08       # pump_prob vs dump_prob gap (8%+) - FASTEST MODE
MIN_PRICE_MOVE_FOR_SIGNAL = 0.01     # Minimum 10m move (~1%) to allow any signal - FASTEST MODE
MIN_VOLUME_SPIKE_FOR_SIGNAL = 1.20   # Minimum volume_change multiple (e.g. 1.2x) - FASTEST MODE
ALLOW_WEAK_MOMENTUM_SIGNALS = False  # Disable heuristic-only weak signals
SHOW_POTENTIAL_SIGNALS = False       # Don't show potential, only real alerts
DETECT_DUMPS = True                  # Also detect dumps
DUMP_THRESHOLD = 0.40                # Threshold for dump detection
SKIP_INITIAL_TRAINING = False        # Set True to skip training on startup
SKIP_INITIAL_BACKTEST = True         # Skip backtest on startup for speed

# Data Collection
CANDLE_INTERVAL = '1m'  # 1 minute candles
HISTORICAL_DATA_LIMIT = 500  # Reduced from 10000 for faster collection
MIN_HISTORICAL_CANDLES = 100  # Minimum candles needed

# Self-Optimization
AUTO_OPTIMIZE_ENABLED = True
OPTIMIZATION_INTERVAL = 86400  # Optimize once per day
PERFORMANCE_TRACKING_WINDOW = 7  # Track performance over 7 days

# Live signal tracking / self-evaluation
SIGNAL_MAX_LIFETIME_HOURS = 4      # After this, open signals are marked as timeout (increased from 2 to reduce timeouts)
EARLY_TIMEOUT_ENABLED = True       # Enable early timeout if price is stuck
EARLY_TIMEOUT_MIN_HOURS = 2.0      # After this many hours with no move, allow early timeout (increased from 1.0)
EARLY_TIMEOUT_MAX_MOVE_PCT = 0.015  # If price moved less than 1.5% from entry, consider it "stuck" (reduced from 2%)
DAILY_REPORT_ENABLED = True
DAILY_REPORT_HOUR_UTC = 0          # When to send daily summary (UTC hour, 0-23)

# Self-repair / auto-tuning behaviour
SELF_REPAIR_ENABLED = True
TARGET_WIN_RATE = 0.55             # Desired minimum daily win rate
THRESHOLD_ADJUST_STEP = 0.05       # Base step for probability/confidence tuning
SELF_REPAIR_MIN_SIGNALS_PER_DAY = 8  # Minimum signals/day before trusting stats
# Allowed ranges for auto-tuning
MIN_PROBABILITY_RANGE = (0.40, 0.90)     # For MIN_CONFIDENCE_SCORE
MIN_AI_CONFIDENCE_RANGE = (0.40, 0.90)   # For MIN_AI_CONFIDENCE
SYMBOL_MIN_WIN_RATE_RANGE = (0.30, 0.70) # For SYMBOL_MIN_WIN_RATE

# Per-symbol performance filtering (prefer coins که خوب جواب داده‌اند)
SYMBOL_STATS_LOOKBACK_DAYS = 7      # Look back this many days per symbol
SYMBOL_MIN_SIGNALS_FOR_FILTER = 8   # Need at least this many signals to judge a symbol
SYMBOL_MIN_WIN_RATE = 0.45          # If symbol winrate below this, temporarily skip new signals

# Scam Detection & Quality Filters
ENABLE_SCAM_DETECTION = True         # Enable scam coin detection
MIN_QUALITY_SCORE = 0.5             # Minimum quality score (0-1) to accept a coin
MIN_RISK_REWARD_RATIO = 1.2         # Minimum risk/reward ratio (1.2 = reward 20% > risk)

# Pattern Learning (Self-Learning Signal Detection)
ENABLE_PATTERN_LEARNING = True      # Enable pattern learning from successful signals
PATTERN_BOOST_ENABLED = True        # Boost signals that match winning patterns
PATTERN_SKIP_ENABLED = True         # Skip signals that match losing patterns
MIN_PATTERN_MATCHES = 3             # Minimum pattern matches to boost/skip
PATTERN_SIMILARITY_THRESHOLD = 0.7  # Minimum similarity (0-1) to consider a match

# Adaptive Filter System
ENABLE_ADAPTIVE_FILTER = True       # Enable adaptive filter adjustment
ADAPTIVE_ADJUSTMENT_INTERVAL_HOURS = 6  # Adjust filters every N hours

# Loss Learning & Prevention
ENABLE_LOSS_LEARNING = True         # Enable learning from losses
LOSS_ANALYSIS_ENABLED = True        # Deep analysis of losing signals
AUTO_BLACKLIST_ON_LOSSES = True     # Auto-blacklist symbols with many losses
MIN_LOSSES_FOR_BLACKLIST = 5       # Blacklist after N losses in 7 days
LOSS_REPORT_INTERVAL_HOURS = 24     # Send loss report every N hours

# Advanced Features
ENABLE_MULTI_TIMEFRAME = True       # Multi-timeframe analysis
ENABLE_BTC_CORRELATION = True       # BTC correlation filter
ENABLE_DYNAMIC_POSITION_SIZING = True  # Dynamic position sizing
ACCOUNT_BALANCE = 100.0             # Account balance for position sizing (USDT)

# AI Self-Improvement (uses free AI APIs to analyze and improve bot)
AI_SELF_IMPROVE_AUTO_APPLY = False  # Set True to auto-apply AI suggestions (default: preview only)

# Self-Teaching Master - Full autonomous code improvement system
ENABLE_SELF_TEACHING_MASTER = True  # Enable the master teacher system
SELF_TEACHING_INTERVAL = 7200  # Run teaching session every 2 hours (7200 seconds) - با بک تست
SELF_TEACHING_TEST_BEFORE_APPLY = True  # Test fixes before applying (recommended: True)
SELF_TEACHING_AUTO_ROLLBACK = True  # Rollback if test fails (recommended: True)
SELF_TEACHING_USE_AI_CODE_GENERATION = True  # Use AI for code generation (requires API key)
SELF_TEACHING_AI_PROVIDER = 'huggingface'  # 'huggingface', 'openai', or 'local'
ENABLE_BACKTEST_IN_TEACHING = True  # Run backtest in teaching session (هر 2 ساعت)

# Real-time Learning - Learn from every signal immediately
ENABLE_REAL_TIME_LEARNING = True  # Analyze each signal immediately when it closes
SIGNALS_FOR_BACKTEST = 10  # Run backtest after N signals (reduced for faster learning)
ENABLE_MICRO_ADJUSTMENTS = True  # Allow small immediate adjustments

# Signal Scoring System
ENABLE_SIGNAL_SCORING = True  # Enable signal scoring (0-100)
PREMIUM_SIGNAL_THRESHOLD = 97.0  # Signals with 97%+ score are premium (fast track)
PREMIUM_SIGNAL_FAST_TRACK = True  # Send premium signals immediately without cooldown

# Advanced Self-Teaching Master - معلم آقای حلاج
ENABLE_ADVANCED_SELF_REPAIR = True  # فعال کردن خود تعمیری پیشرفته
ENABLE_FILE_CREATION = True  # اجازه ساخت فایل جدید
ENABLE_SUB_TEACHERS = True  # اجازه ساخت معلم‌های جدید (شاگردان)
MAX_TEACHER_LEVELS = 3  # حداکثر سطح معلم‌ها (Master + 2 level students)
PROTECTED_FILES_ONLY_CRITICAL = True  # فقط فایل‌های حساس محافظت شوند

