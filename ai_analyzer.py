"""
Advanced AI Analyzer for Pump Detection
Uses multiple ML models to predict potential pumps
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, Optional
import warnings
import config
warnings.filterwarnings('ignore')

# Suppress sklearn warnings
import logging
logging.getLogger('sklearn').setLevel(logging.ERROR)

class AIAnalyzer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_dir = 'models'
        self.performance_history = []
        # Live training data from real-time signals
        self.live_data_path = os.path.join(self.model_dir, 'live_training_data.joblib')
        self.live_X = None  # pandas.DataFrame or None
        self.live_y = None  # pandas.Series or None
        os.makedirs(self.model_dir, exist_ok=True)
        self._initialize_models()
        self._load_live_data()
    
    def _initialize_models(self):
        """Initialize multiple ML models for ensemble prediction"""
        # Random Forest Model
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting Model
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            random_state=42
        )
        
        # Scalers for feature normalization
        self.scalers['random_forest'] = StandardScaler()
        self.scalers['gradient_boosting'] = StandardScaler()
        
        # Try to load existing models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f'{model_name}_model.joblib')
            scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                try:
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    print(f"Loaded {model_name} model successfully")
                except Exception as e:
                    print(f"Error loading {model_name}: {e}")
    
    def _save_models(self):
        """Save trained models"""
        for model_name in self.models.keys():
            model_path = os.path.join(self.model_dir, f'{model_name}_model.joblib')
            scaler_path = os.path.join(self.model_dir, f'{model_name}_scaler.joblib')
            
            try:
                joblib.dump(self.models[model_name], model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
            except Exception as e:
                print(f"Error saving {model_name}: {e}")

    def _load_live_data(self):
        """Load previously stored live training data if available."""
        if os.path.exists(self.live_data_path):
            try:
                data = joblib.load(self.live_data_path)
                self.live_X = data.get("X")
                self.live_y = data.get("y")
                if self.live_X is not None:
                    print(f"Loaded {len(self.live_X)} live training examples")
            except Exception as e:
                print(f"Error loading live training data: {e}")
                self.live_X, self.live_y = None, None

    def _save_live_data(self):
        """Persist live training data to disk."""
        try:
            joblib.dump({"X": self.live_X, "y": self.live_y}, self.live_data_path)
        except Exception as e:
            print(f"Error saving live training data: {e}")

    def add_live_example(self, features: dict, label: int):
        """
        Add a single live training example based on real signal outcome.
        features: dict of feature_name -> value (must match extract_features columns)
        label: 1 for successful pump (TP hit), 0 for failed (SL hit)
        """
        import pandas as pd  # local import to avoid circular issues

        if features is None:
            return

        row = pd.DataFrame([features])

        if self.live_X is None or self.live_y is None:
            self.live_X = row
            self.live_y = pd.Series([int(label)])
        else:
            self.live_X = pd.concat([self.live_X, row], ignore_index=True)
            self.live_y = pd.concat([self.live_y, pd.Series([int(label)])], ignore_index=True)

        # Keep live buffer from exploding: cap to last N examples
        max_examples = 5000
        if len(self.live_X) > max_examples:
            excess = len(self.live_X) - max_examples
            self.live_X = self.live_X.iloc[excess:].reset_index(drop=True)
            self.live_y = self.live_y.iloc[excess:].reset_index(drop=True)

        self._save_live_data()
    
    def extract_features(self, df):
        """
        Extract advanced technical features from price data
        """
        features = pd.DataFrame()
        
        # Price features
        features['price_change_1m'] = df['close'].pct_change(1)
        features['price_change_5m'] = df['close'].pct_change(5)
        features['price_change_10m'] = df['close'].pct_change(10)
        features['price_change_30m'] = df['close'].pct_change(30)
        
        # Volume features
        features['volume_change_1m'] = df['volume'].pct_change(1)
        features['volume_change_5m'] = df['volume'].pct_change(5)
        features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_price_trend'] = (df['volume'] * df['close']).pct_change(5)
        
        # Technical indicators
        features['rsi'] = self._calculate_rsi(df['close'], 14)
        features['macd'] = self._calculate_macd(df['close'])
        features['bollinger_position'] = self._calculate_bollinger_position(df['close'])
        features['stochastic'] = self._calculate_stochastic(df)
        
        # Price patterns
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        features['price_momentum'] = df['close'].pct_change(10)
        
        # Volatility
        features['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        features['price_range'] = (df['high'] - df['low']) / df['close']
        
        # Market structure
        features['support_distance'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        features['resistance_distance'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        
        # Order book imbalance (simulated from price action)
        features['buy_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        
        # Fill NaN values and replace infinity
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # Clip extreme values
        features = features.clip(lower=-1e10, upper=1e10)
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def _calculate_bollinger_position(self, prices, period=20, std_dev=2):
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        position = (prices - lower) / (upper - lower + 1e-10)
        return position.fillna(0.5)
    
    def _calculate_stochastic(self, df, period=14):
        """Calculate Stochastic Oscillator"""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()
        stoch = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        return stoch.fillna(50)
    
    def train_models(self, historical_data):
        """
        Train models on historical data
        historical_data: dict with coin symbols as keys and DataFrames as values
        """
        print("Starting model training...")
        
        all_features = []
        all_labels = []
        
        for symbol, df in historical_data.items():
            if len(df) < 100:
                continue
            
            # Extract features
            features = self.extract_features(df)
            
            # Create labels: 1 if price increases >10% in next 10 minutes
            future_price = df['close'].shift(-10)
            price_change = (future_price - df['close']) / df['close']
            labels = (price_change > 0.10).astype(int)
            
            # Align features and labels
            min_len = min(len(features), len(labels))
            features = features.iloc[:min_len]
            labels = labels.iloc[:min_len]
            
            # Remove rows with NaN labels
            valid_mask = ~labels.isna()
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            if len(features) > 0:
                all_features.append(features)
                all_labels.append(labels)
        
        if len(all_features) == 0:
            print("No valid training data found")
            return
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)

        # Append live training data (real signal outcomes) if available
        if self.live_X is not None and self.live_y is not None and len(self.live_X) > 0:
            try:
                # Align columns
                live_X_aligned = self.live_X.reindex(columns=X.columns, fill_value=0)
                X = pd.concat([X, live_X_aligned], ignore_index=True)
                y = pd.concat([y, self.live_y], ignore_index=True)
                print(f"Included {len(self.live_X)} live examples in training")
            except Exception as e:
                print(f"Error merging live training data: {e}")
        
        # Remove any remaining NaN and infinity
        X = X.replace([np.inf, -np.inf], np.nan)
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Clip extreme values
        X = X.clip(lower=-1e10, upper=1e10)
        
        if len(X) == 0:
            print("No valid data after cleaning")
            return
        
        # Check for infinity again
        if np.isinf(X.values).any():
            print("Warning: Still contains infinity, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train each model
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            scaler = self.scalers[model_name]
            
            # Scale features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"{model_name} - Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")
        
        # Save models
        self._save_models()
        print("Model training completed!")
    
    def predict_pump_probability(self, df):
        """
        Predict probability of pump in next 10 minutes
        Returns: probability (0-1), confidence score
        """
        if len(df) < 50:
            return 0.0, 0.0
        
        # Extract features from latest data
        features = self.extract_features(df)
        
        if len(features) == 0:
            return 0.0, 0.0
        
        # Get latest feature vector
        latest_features = features.iloc[-1:].values
        
        # Replace infinity and clip
        latest_features = np.nan_to_num(latest_features, nan=0.0, posinf=1e10, neginf=-1e10)
        latest_features = np.clip(latest_features, -1e10, 1e10)
        
        # Ensemble prediction
        predictions = []
        confidences = []
        
        for model_name, model in self.models.items():
            try:
                scaler = self.scalers[model_name]
                
                # Check if scaler is fitted
                if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
                    # Use simple fallback prediction
                    continue
                
                features_scaled = scaler.transform(latest_features)
                
                # Check for infinity in scaled features
                if np.isinf(features_scaled).any() or np.isnan(features_scaled).any():
                    continue
                
                # Get prediction probability
                prob = model.predict_proba(features_scaled)[0]
                prediction = model.predict(features_scaled)[0]
                
                # Use probability of positive class
                if len(prob) > 1:
                    pump_prob = prob[1]
                else:
                    pump_prob = prob[0] if prediction == 1 else 1 - prob[0]
                
                predictions.append(pump_prob)
                
                # Confidence based on probability distance from 0.5
                confidence = abs(pump_prob - 0.5) * 2
                confidences.append(confidence)
                
            except Exception as e:
                # Silent error - models not trained yet
                continue
        
        if len(predictions) == 0:
            # Fallback: use simple technical analysis
            return self._fallback_prediction(df)
        
        # Average predictions
        avg_probability = np.mean(predictions)
        avg_confidence = np.mean(confidences)
        
        return avg_probability, avg_confidence
    
    def _fallback_prediction(self, df):
        """Simple fallback prediction when models aren't trained - More aggressive"""
        if len(df) < 10:
            return 0.0, 0.0
        
        # Simple momentum-based prediction
        price_change_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
        price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
        
        # Volume analysis
        volume_recent = df['volume'].iloc[-10:].mean()
        volume_previous = df['volume'].iloc[-20:-10].mean() if len(df) >= 20 else volume_recent
        volume_change = volume_recent / (volume_previous + 1e-10)
        
        # RSI-like calculation
        price_changes = df['close'].pct_change().dropna()
        gains = price_changes[price_changes > 0].tail(14).sum()
        losses = abs(price_changes[price_changes < 0].tail(14).sum())
        rs = gains / (losses + 1e-10) if losses > 0 else 10
        rsi = 100 - (100 / (1 + rs))
        
        # More aggressive heuristics
        if price_change_10m > 0.08 and volume_change > 2.0:
            prob = 0.70
            conf = 0.65
        elif price_change_10m > 0.05 and volume_change > 1.5:
            prob = 0.60
            conf = 0.55
        elif price_change_5m > 0.03 and volume_change > 1.3:
            prob = 0.55
            conf = 0.50
        elif price_change_10m > 0.03 and volume_change > 1.2:
            prob = 0.50
            conf = 0.45
        elif price_change_10m > 0.02 and volume_change > 1.1:
            prob = 0.45
            conf = 0.40
        elif rsi > 60 and price_change_10m > 0.01:
            prob = 0.50
            conf = 0.45
        else:
            prob = 0.35
            conf = 0.30
        
        return prob, conf
    
    def predict_dump_probability(self, df):
        """Predict probability of dump in next 10 minutes"""
        if len(df) < 50:
            return 0.0, 0.0
        
        # Use same method but inverted for dumps
        pump_prob, confidence = self.predict_pump_probability(df)
        
        # For dump, we look for negative price action
        price_change_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
        
        # If price is dropping, increase dump probability
        if price_change_10m < -0.03:  # 3% drop
            dump_prob = 0.6 + abs(price_change_10m) * 2  # Scale with drop
            dump_prob = min(dump_prob, 0.9)
            dump_conf = confidence + 0.1
        elif price_change_10m < -0.02:  # 2% drop
            dump_prob = 0.5
            dump_conf = confidence
        else:
            dump_prob = 0.3
            dump_conf = 0.3

        # Clamp confidence to [0, 1] to avoid >100% in reports
        dump_conf = max(0.0, min(dump_conf, 1.0))
        
        return dump_prob, dump_conf
    
    def analyze_coin(self, symbol, df):
        """
        Complete analysis of a coin - detects both PUMP and DUMP
        Returns analysis dictionary
        """
        if len(df) < 50:
            return None
        
        # Predict both pump and dump
        pump_prob, pump_conf = self.predict_pump_probability(df)
        dump_prob, dump_conf = self.predict_dump_probability(df)
        
        # Calculate additional metrics
        current_price = df['close'].iloc[-1]
        price_change_10m = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10] if len(df) >= 10 else 0
        price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] if len(df) >= 5 else 0
        volume_24h = df['volume'].tail(1440).sum() if len(df) >= 1440 else df['volume'].sum()
        volume_change = df['volume'].iloc[-10:].mean() / (df['volume'].iloc[-20:-10].mean() + 1e-10) if len(df) >= 20 else 1.0
        
        # Determine signal type (PUMP or DUMP) - high precision logic
        signal_type = None
        signal_prob = 0.0
        signal_conf = 0.0

        min_prob = getattr(config, "MIN_CONFIDENCE_SCORE", 0.4)
        min_conf = getattr(config, "MIN_AI_CONFIDENCE", 0.35)
        dominance = getattr(config, "SIGNAL_DOMINANCE_MARGIN", 0.1)
        min_move = getattr(config, "MIN_PRICE_MOVE_FOR_SIGNAL", 0.0)
        min_vol_spike = getattr(config, "MIN_VOLUME_SPIKE_FOR_SIGNAL", 1.0)
        allow_weak_momentum = getattr(config, "ALLOW_WEAK_MOMENTUM_SIGNALS", True)

        # Global quality filters: نیاز به حرکت قیمت و اسپایک حجم
        if abs(price_change_10m) < min_move or volume_change < min_vol_spike:
            # حرکت و ولوم کافی نیست؛ اصلاً سیگنال تولید نکن
            return None

        # فقط وقتی پامپ یا دامپ از دیگری به‌وضوح قوی‌تر است سیگنال بده
        if pump_prob >= dump_prob + dominance and pump_prob >= min_prob and pump_conf >= min_conf:
            signal_type = 'PUMP'
            signal_prob = pump_prob
            signal_conf = pump_conf
        elif dump_prob >= pump_prob + dominance and dump_prob >= min_prob and dump_conf >= min_conf:
            signal_type = 'DUMP'
            signal_prob = dump_prob
            signal_conf = dump_conf
        # در حالت عادی، سیگنال‌های صرفاً مومنتومی را حذف می‌کنیم مگر این‌که صراحتاً فعال شده باشند
        elif allow_weak_momentum and abs(price_change_10m) > 0.03:  # Strong movement
            if price_change_10m > 0:
                signal_type = 'PUMP'
                signal_prob = max(0.50, pump_prob)
                signal_conf = max(0.45, pump_conf)
            else:
                signal_type = 'DUMP'
                signal_prob = max(0.50, dump_prob)
                signal_conf = max(0.45, dump_conf)
        
        # Calculate Entry, Exit, SL based on signal type
        entry_price = current_price
        
        # Try to get coin-specific TP/SL first (if coin has learned profile)
        from coin_specific_learner import CoinSpecificLearner
        coin_learner = CoinSpecificLearner()
        coin_strategy = coin_learner.get_coin_strategy(symbol)
        
        # Default TP/SL based on signal probability
        if signal_type == 'PUMP':
            # For pump: Entry at current, Exit at higher levels
            if signal_prob > 0.7:
                default_tp = 0.12  # 12% exit
                default_sl = 0.08  # 8% stop loss
            elif signal_prob > 0.6:
                default_tp = 0.10  # 10% exit
                default_sl = 0.07  # 7% stop loss
            else:
                default_tp = 0.08  # 8% exit
                default_sl = 0.06  # 6% stop loss
            
            # Get coin-specific optimal TP/SL
            optimal_tp, optimal_sl = coin_learner.get_coin_optimal_tp_sl(
                symbol, signal_type, default_tp, default_sl
            )
            
            exit1 = entry_price * (1 + optimal_tp)
            exit2 = entry_price * (1 + optimal_tp * 1.5)
            exit3 = entry_price * (1 + optimal_tp * 2.0)
            sl = entry_price * (1 - optimal_sl)
            
            recommendation = 'BUY'
            
        elif signal_type == 'DUMP':
            # For dump: Entry at current (SHORT), Exit at lower levels
            if signal_prob > 0.7:
                default_tp = 0.12  # 12% down exit
                default_sl = 0.08  # 8% stop loss
            elif signal_prob > 0.6:
                default_tp = 0.10  # 10% down exit
                default_sl = 0.07  # 7% stop loss
            else:
                default_tp = 0.08  # 8% down exit
                default_sl = 0.06  # 6% stop loss
            
            # Get coin-specific optimal TP/SL
            optimal_tp, optimal_sl = coin_learner.get_coin_optimal_tp_sl(
                symbol, signal_type, default_tp, default_sl
            )
            
            exit1 = entry_price * (1 - optimal_tp)
            exit2 = entry_price * (1 - optimal_tp * 1.5)
            exit3 = entry_price * (1 - optimal_tp * 2.0)
            sl = entry_price * (1 + optimal_sl)
            
            recommendation = 'SELL/SHORT'
        else:
            # No clear signal
            return None
        
        # Also keep the latest feature vector so we can learn from real outcomes later
        try:
            features_df = self.extract_features(df)
            latest_features = features_df.iloc[-1].to_dict()
        except Exception:
            latest_features = None

        # Calculate risk/reward ratio
        from scam_detector import ScamDetector
        from pattern_learner import PatternLearner
        scam_detector = ScamDetector()
        rr_ratio = scam_detector.get_risk_reward_ratio(entry_price, exit1, sl, signal_type)
        
        # Additional quality filter: minimum risk/reward ratio
        min_rr_ratio = getattr(config, 'MIN_RISK_REWARD_RATIO', 1.2)
        if rr_ratio < min_rr_ratio:
            return None  # Skip low-margin signals
        
        # Create temporary analysis dict for pattern matching
        temp_analysis = {
            'symbol': symbol,
            'signal_type': signal_type,
            'signal_probability': signal_prob,
            'confidence': signal_conf,
            'price_change_10m': price_change_10m,
            'volume_change': volume_change,
            'risk_reward_ratio': rr_ratio
        }
        
        # Pattern matching - boost signals that match winning patterns
        pattern_learner = PatternLearner()
        pattern_match = pattern_learner.match_pattern(temp_analysis, df)
        
        # Skip if matches losing patterns
        if pattern_learner.should_skip_signal(temp_analysis, df):
            return None
        
        # Boost confidence if matches winning patterns
        pattern_boost = False
        pattern_match_score = 0.0
        if pattern_match['should_boost']:
            signal_conf = min(1.0, signal_conf * pattern_match['boost_factor'])
            signal_prob = min(1.0, signal_prob * pattern_match['boost_factor'])
            pattern_boost = True
            pattern_match_score = pattern_match['match_score']
        
        # Calculate comprehensive signal score (0-100)
        signal_score = self._calculate_signal_score(
            signal_prob, signal_conf, price_change_10m, volume_change,
            rr_ratio, pattern_boost, pattern_match_score, 
            multi_tf_analysis if 'multi_tf_analysis' in locals() else None,
            btc_correlation if 'btc_correlation' in locals() else None
        )
        
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'signal_type': signal_type,  # PUMP or DUMP
            'signal_probability': signal_prob,
            'confidence': signal_conf,
            'pump_probability': pump_prob,
            'dump_probability': dump_prob,
            'price_change_10m': price_change_10m,
            'price_change_5m': price_change_5m,
            'volume_24h': volume_24h,
            'volume_change': volume_change,
            'timestamp': datetime.now(),
            'recommendation': recommendation,
            # Trading levels - Enhanced entry/exit calculation
            'entry': self._calculate_optimal_entry(entry_price, signal_type, df),
            'exit1': exit1,
            'exit2': exit2,
            'exit3': exit3,
            'stop_loss': self._calculate_optimal_stop_loss(sl, signal_type, df),
            'risk_reward_ratio': rr_ratio,
            # Pattern learning info
            'pattern_boost': pattern_boost,
            'pattern_match_score': pattern_match_score,
            # Signal scoring
            'signal_score': signal_score,  # 0-100 score
            'is_premium_signal': signal_score >= 97.0,  # Premium signal flag
            # Raw feature vector at signal time (for live training)
            'features': latest_features
        }
        
        return analysis
    
    def _calculate_signal_score(self, signal_prob: float, signal_conf: float, 
                               price_change_10m: float, volume_change: float,
                               rr_ratio: float, pattern_boost: bool, 
                               pattern_match_score: float,
                               multi_tf_analysis: Optional[Dict] = None,
                               btc_correlation: Optional[Dict] = None) -> float:
        """
        Calculate comprehensive signal score (0-100)
        Higher score = better signal quality
        """
        score = 0.0
        
        # 1. Probability component (0-30 points)
        prob_score = min(30.0, signal_prob * 30)
        score += prob_score
        
        # 2. Confidence component (0-25 points)
        conf_score = min(25.0, signal_conf * 25)
        score += conf_score
        
        # 3. Price movement strength (0-15 points)
        move_strength = min(abs(price_change_10m) * 500, 15.0)  # 3% move = 15 points
        score += move_strength
        
        # 4. Volume spike strength (0-10 points)
        vol_score = min((volume_change - 1.0) * 5, 10.0)  # 2x volume = 10 points
        score += vol_score
        
        # 5. Risk/Reward ratio (0-10 points)
        rr_score = min((rr_ratio - 1.0) * 5, 10.0)  # 3.0 RR = 10 points
        score += rr_score
        
        # 6. Pattern matching boost (0-5 points)
        if pattern_boost:
            pattern_score = min(pattern_match_score * 5, 5.0)
            score += pattern_score
        
        # 7. Multi-timeframe confirmation (0-3 points)
        if multi_tf_analysis and multi_tf_analysis.get('timeframe_alignment'):
            if multi_tf_analysis.get('multi_tf_boost'):
                score += 3.0
        
        # 8. BTC correlation bonus (0-2 points)
        if btc_correlation:
            btc_corr = btc_correlation.get('correlation', 0)
            if abs(btc_corr) < 0.3:  # Low correlation = independent move (good)
                score += 2.0
            elif btc_correlation.get('btc_dominance') == False:
                score += 1.0
        
        # Ensure score is between 0-100
        score = max(0.0, min(100.0, score))
        
        return round(score, 2)
    
    def _calculate_optimal_entry(self, current_price: float, signal_type: str, df: pd.DataFrame) -> float:
        """
        Calculate optimal entry point using support/resistance levels
        """
        if len(df) < 20:
            return current_price
        
        # For PUMP: try to enter slightly below current (support level)
        if signal_type == 'PUMP':
            # Find recent support (lowest low in last 20 candles)
            recent_low = df['low'].iloc[-20:].min()
            # Entry at support or current, whichever is lower
            optimal_entry = min(current_price, recent_low * 1.001)  # 0.1% above support
            return round(optimal_entry, 8)
        
        # For DUMP: try to enter slightly above current (resistance level)
        elif signal_type == 'DUMP':
            # Find recent resistance (highest high in last 20 candles)
            recent_high = df['high'].iloc[-20:].max()
            # Entry at resistance or current, whichever is higher
            optimal_entry = max(current_price, recent_high * 0.999)  # 0.1% below resistance
            return round(optimal_entry, 8)
        
        return current_price
    
    def _calculate_optimal_stop_loss(self, default_sl: float, signal_type: str, df: pd.DataFrame) -> float:
        """
        Calculate optimal stop loss using ATR (Average True Range)
        """
        if len(df) < 14:
            return default_sl
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        
        current_price = df['close'].iloc[-1]
        
        if signal_type == 'PUMP':
            # SL below entry by 1.5x ATR
            optimal_sl = current_price - (atr * 1.5)
            # Use tighter of default or ATR-based
            return round(min(default_sl, optimal_sl), 8)
        
        elif signal_type == 'DUMP':
            # SL above entry by 1.5x ATR
            optimal_sl = current_price + (atr * 1.5)
            # Use tighter of default or ATR-based
            return round(max(default_sl, optimal_sl), 8)
        
        return default_sl

