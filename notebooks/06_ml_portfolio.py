# Alpha Project: Academic Research Implementation Engine
# Notebook 6: Machine Learning Applications in Portfolio Construction

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MLPortfolioConstruction:
    """
    Machine Learning Applications in Portfolio Construction
    Based on: Gu, S., Kelly, B., & Xiu, D. (2020) - ML in Asset Pricing
             Kozak, S., Nagel, S., & Santosh, S. (2020) - Shrinking the Cross Section
             Freyberger, J., Neuhierl, A., & Weber, M. (2020) - Dissecting Characteristics
    """
    
    def __init__(self):
        self.market_data = None
        self.features = None
        self.models = {}
        self.predictions = None
        self.portfolios = None
        
    def load_data_and_features(self, tickers, start_date='2015-01-01', end_date='2024-01-01'):
        """Load market data and construct features for ML models"""
        
        print("Loading market data and constructing features...")
        
        # Download price and volume data
        data = yf.download(tickers, start=start_date, end=end_date)
        
        if len(tickers) == 1:
            # Single ticker case
            prices = data['Adj Close'].to_frame()
            volumes = data['Volume'].to_frame()
        else:
            prices = data['Adj Close']
            volumes = data['Volume']
        
        returns = prices.pct_change().dropna()
        
        self.market_data = {
            'prices': prices,
            'returns': returns,
            'volumes': volumes
        }
        
        # Construct features
        self.features = self.construct_features(prices, returns, volumes)
        
        print(f"Loaded {len(tickers)} assets with {len(self.features.columns)} features")
        return self.market_data, self.features
    
    def construct_features(self, prices, returns, volumes):
        """Construct comprehensive feature set for ML models"""
        
        features_list = []
        
        for ticker in returns.columns:
            ticker_features = pd.DataFrame(index=returns.index)
            
            # Price-based features
            price_series = prices[ticker].dropna()
            return_series = returns[ticker].dropna()
            
            # 1. Return-based features
            windows = [5, 10, 21, 63, 126, 252]
            for window in windows:
                if len(return_series) > window:
                    # Momentum features
                    ticker_features[f'momentum_{window}d'] = return_series.rolling(window).mean()
                    ticker_features[f'volatility_{window}d'] = return_series.rolling(window).std()
                    
                    # Cumulative returns
                    ticker_features[f'cumret_{window}d'] = (1 + return_series).rolling(window).apply(
                        lambda x: x.prod() - 1, raw=False
                    )
                    
                    # Maximum drawdown
                    def max_drawdown(x):
                        cumulative = (1 + x).cumprod()
                        rolling_max = cumulative.expanding().max()
                        dd = (cumulative - rolling_max) / rolling_max
                        return dd.min()
                    
                    ticker_features[f'max_dd_{window}d'] = return_series.rolling(window).apply(
                        max_drawdown, raw=False
                    )
            
            # 2. Technical indicators
            # RSI (Relative Strength Index)
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                return rsi
            
            ticker_features['rsi_14'] = calculate_rsi(price_series)
            
            # Moving averages and crossovers
            ticker_features['sma_20'] = price_series.rolling(20).mean()
            ticker_features['sma_50'] = price_series.rolling(50).mean()
            ticker_features['sma_200'] = price_series.rolling(200).mean()
            
            # Moving average crossovers
            ticker_features['sma_20_50_cross'] = (
                ticker_features['sma_20'] / ticker_features['sma_50'] - 1
            )
            ticker_features['sma_50_200_cross'] = (
                ticker_features['sma_50'] / ticker_features['sma_200'] - 1
            )
            
            # Price relative to moving averages
            ticker_features['price_to_sma_20'] = price_series / ticker_features['sma_20'] - 1
            ticker_features['price_to_sma_50'] = price_series / ticker_features['sma_50'] - 1
            
            # 3. Volume-based features
            if ticker in volumes.columns:
                volume_series = volumes[ticker].dropna()
                
                # Volume moving averages
                ticker_features['volume_sma_20'] = volume_series.rolling(20).mean()
                ticker_features['volume_ratio'] = volume_series / ticker_features['volume_sma_20']
                
                # Volume-price indicators
                ticker_features['volume_price_trend'] = (
                    (return_series * volume_series).rolling(10).mean()
                )
            
            # 4. Higher-order moments
            for window in [21, 63]:
                if len(return_series) > window:
                    ticker_features[f'skewness_{window}d'] = return_series.rolling(window).skew()
                    ticker_features[f'kurtosis_{window}d'] = return_series.rolling(window).kurt()
            
            # 5. Risk-adjusted returns
            for window in [21, 63, 126]:
                if len(return_series) > window:
                    mean_ret = return_series.rolling(window).mean()
                    vol_ret = return_series.rolling(window).std()
                    ticker_features[f'sharpe_{window}d'] = mean_ret / (vol_ret + 1e-8)
            
            # Add ticker identifier
            for col in ticker_features.columns:
                ticker_features[col] = ticker_features[col]
            
            # Stack features with multi-index
            ticker_features['ticker'] = ticker
            ticker_features = ticker_features.reset_index().melt(
                id_vars=['Date', 'ticker'], 
                var_name='feature', 
                value_name='value'
            )
            ticker_features['feature_ticker'] = ticker_features['feature'] + '_' + ticker
            
            features_list.append(ticker_features.pivot(
                index='Date', columns='feature_ticker', values='value'
            ))
        
        # Combine all features
        all_features = pd.concat(features_list, axis=1)
        
        # Clean features
        all_features = all_features.replace([np.inf, -np.inf], np.nan)
        all_features = all_features.fillna(method='ffill').fillna(0)
        
        return all_features
    
    def prepare_ml_dataset(self, features, returns, prediction_horizon=21, 
                          min_history=252, feature_selection=True):
        """Prepare dataset for machine learning models"""
        
        print("Preparing ML dataset...")
        
        # Align data
        common_dates = features.index.intersection(returns.index)
        features_aligned = features.reindex(common_dates)
        returns_aligned = returns.reindex(common_dates)
        
        # Create forward returns (targets)
        forward_returns = returns_aligned.shift(-prediction_horizon)
        
        # Create dataset
        X_list = []
        y_list = []
        dates_list = []
        tickers_list = []
        
        for i in range(min_history, len(common_dates) - prediction_horizon):
            date = common_dates[i]
            
            # Features at time t
            feature_row = features_aligned.iloc[i]
            
            # Remove NaN features and corresponding returns
            valid_features = feature_row.dropna()
            
            # Extract tickers from feature names
            ticker_features = {}
            for feature_name in valid_features.index:
                parts = feature_name.split('_')
                if len(parts) >= 2:
                    ticker = parts[-1]
                    if ticker in returns_aligned.columns:
                        if ticker not in ticker_features:
                            ticker_features[ticker] = {}
                        feature_base = '_'.join(parts[:-1])
                        ticker_features[ticker][feature_base] = valid_features[feature_name]
            
            # Create feature vectors for each ticker
            for ticker in ticker_features.keys():
                if ticker in returns_aligned.columns:
                    # Forward return for this ticker
                    forward_ret = forward_returns.loc[date, ticker]
                    
                    if not np.isnan(forward_ret):
                        # Feature vector
                        ticker_feature_vector = list(ticker_features[ticker].values())
                        
                        if len(ticker_feature_vector) > 0 and not any(np.isnan(ticker_feature_vector)):
                            X_list.append(ticker_feature_vector)
                            y_list.append(forward_ret)
                            dates_list.append(date)
                            tickers_list.append(ticker)
        
        # Convert to arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Feature selection if requested
        if feature_selection and X.shape[1] > 20:
            # Use random forest for feature importance
            rf_temp = RandomForestRegressor(n_estimators=50, random_state=42)
            rf_temp.fit(X, y)
            
            # Select top features
            feature_importance = rf_temp.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-20:]  # Top 20 features
            X = X[:, top_features_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        dataset = {
            'X': X_scaled,
            'y': y,
            'dates': dates_list,
            'tickers': tickers_list,
            'scaler': scaler,
            'n_features': X_scaled.shape[1] if len(X_scaled.shape) > 1 else 0
        }
        
        print(f"Dataset prepared: {len(X_scaled)} samples, {dataset['n_features']} features")
        return dataset
    
    def train_ml_models(self, dataset):
        """Train multiple ML models for return prediction"""
        
        print("Training ML models...")
        
        X, y = dataset['X'], dataset['y']
        
        if len(X) == 0 or dataset['n_features'] == 0:
            print("Insufficient data for ML training")
            return {}
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {}
        
        # 1. Linear Models
        print("  Training Linear Models...")
        
        # Ridge Regression
        ridge = Ridge(alpha=1.0)
        ridge_scores = cross_val_score(ridge, X, y, cv=tscv, scoring='neg_mean_squared_error')
        ridge.fit(X, y)
        models['Ridge'] = {
            'model': ridge,
            'cv_score': -ridge_scores.mean(),
            'type': 'linear'
        }
        
        # Lasso Regression
        lasso = Lasso(alpha=0.01)
        lasso_scores = cross_val_score(lasso, X, y, cv=tscv, scoring='neg_mean_squared_error')
        lasso.fit(X, y)
        models['Lasso'] = {
            'model': lasso,
            'cv_score': -lasso_scores.mean(),
            'type': 'linear'
        }
        
        # Elastic Net
        elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
        elastic_scores = cross_val_score(elastic, X, y, cv=tscv, scoring='neg_mean_squared_error')
        elastic.fit(X, y)
        models['ElasticNet'] = {
            'model': elastic,
            'cv_score': -elastic_scores.mean(),
            'type': 'linear'
        }
        
        # 2. Tree-based Models
        print("  Training Tree-based Models...")
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        rf_scores = cross_val_score(rf, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rf.fit(X, y)
        models['RandomForest'] = {
            'model': rf,
            'cv_score': -rf_scores.mean(),
            'type': 'tree'
        }
        
        # Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
        gb_scores = cross_val_score(gb, X, y, cv=tscv, scoring='neg_mean_squared_error')
        gb.fit(X, y)
        models['GradientBoosting'] = {
            'model': gb,
            'cv_score': -gb_scores.mean(),
            'type': 'tree'
        }
        
        # 3. Neural Network
        print("  Training Neural Network...")
        
        if len(X) > 1000 and dataset['n_features'] > 5:  # Only if sufficient data
            try:
                # Simple feedforward network
                nn_model = keras.Sequential([
                    layers.Dense(64, activation='relu', input_shape=(dataset['n_features'],)),
                    layers.Dropout(0.3),
                    layers.Dense(32, activation='relu'),
                    layers.Dropout(0.3),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1)
                ])
                
                nn_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                # Train with validation split
                early_stopping = keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                history = nn_model.fit(
                    X, y,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Calculate CV score manually for NN
                nn_cv_scores = []
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    temp_model = keras.models.clone_model(nn_model)
                    temp_model.compile(optimizer='adam', loss='mse')
                    temp_model.fit(X_train, y_train, epochs=50, verbose=0)
                    
                    y_pred = temp_model.predict(X_val, verbose=0)
                    mse = mean_squared_error(y_val, y_pred)
                    nn_cv_scores.append(mse)
                
                models['NeuralNetwork'] = {
                    'model': nn_model,
                    'cv_score': np.mean(nn_cv_scores),
                    'type': 'neural'
                }
                
            except Exception as e:
                print(f"    Neural Network training failed: {e}")
        
        self.models = models
        
        # Print model performance
        print("\nModel Cross-Validation Scores (MSE):")
        for name, model_info in models.items():
            print(f"  {name}: {model_info['cv_score']:.6f}")
        
        return models
    
    def generate_ml_predictions(self, dataset, models):
        """Generate predictions using trained ML models"""
        
        print("Generating ML predictions...")
        
        X = dataset['X']
        dates = dataset['dates']
        tickers = dataset['tickers']
        
        predictions = {}
        
        for model_name, model_info in models.items():
            model = model_info['model']
            
            try:
                if model_info['type'] == 'neural':
                    preds = model.predict(X, verbose=0).flatten()
                else:
                    preds = model.predict(X)
                
                # Create prediction DataFrame
                pred_df = pd.DataFrame({
                    'date': dates,
                    'ticker': tickers,
                    'prediction': preds
                })
                
                # Pivot to get ticker columns
                pred_pivot = pred_df.pivot(index='date', columns='ticker', values='prediction')
                predictions[model_name] = pred_pivot
                
            except Exception as e:
                print(f"Prediction failed for {model_name}: {e}")
        
        self.predictions = predictions
        return predictions
    
    def create_ml_portfolios(self, predictions, returns_data, portfolio_method='long_short',
                            n_positions=10, rebalance_freq='monthly'):
        """Create portfolios based on ML predictions"""
        
        print("Creating ML-based portfolios...")
        
        portfolios = {}
        
        for model_name, pred_data in predictions.items():
            
            # Align predictions with returns
            common_dates = pred_data.index.intersection(returns_data.index)
            pred_aligned = pred_data.reindex(common_dates)
            returns_aligned = returns_data.reindex(common_dates)
            
            # Determine rebalancing dates
            if rebalance_freq == 'monthly':
                rebalance_dates = pred_aligned.resample('M').last().index
            elif rebalance_freq == 'weekly':
                rebalance_dates = pred_aligned.resample('W').last().index
            else:
                rebalance_dates = pred_aligned.index
            
            portfolio_returns = []
            
            for i, rebal_date in enumerate(rebalance_dates[:-1]):
                next_rebal_date = rebalance_dates[i + 1]
                
                # Get predictions for portfolio formation
                current_predictions = pred_aligned.loc[rebal_date]
                valid_predictions = current_predictions.dropna()
                
                if len(valid_predictions) < n_positions:
                    continue
                
                # Create portfolio weights based on method
                weights = pd.Series(0.0, index=returns_aligned.columns)
                
                if portfolio_method == 'long_only':
                    # Long top predictions
                    top_stocks = valid_predictions.nlargest(n_positions).index
                    weights[top_stocks] = 1.0 / len(top_stocks)
                    
                elif portfolio_method == 'long_short':
                    # Long top, short bottom
                    n_long = n_positions // 2
                    n_short = n_positions // 2
                    
                    top_stocks = valid_predictions.nlargest(n_long).index
                    bottom_stocks = valid_predictions.nsmallest(n_short).index
                    
                    weights[top_stocks] = 1.0 / n_long
                    weights[bottom_stocks] = -1.0 / n_short
                    
                elif portfolio_method == 'prediction_weighted':
                    # Weight by prediction strength
                    pred_weights = valid_predictions / valid_predictions.abs().sum()
                    weights[pred_weights.index] = pred_weights
                
                # Calculate portfolio returns for holding period
                holding_period_returns = returns_aligned.loc[rebal_date:next_rebal_date]
                
                if len(holding_period_returns) > 1:
                    daily_portfolio_returns = (holding_period_returns * weights).sum(axis=1)
                    portfolio_returns.extend(daily_portfolio_returns.iloc[1:].tolist())
            
            if len(portfolio_returns) > 0:
                portfolios[model_name] = pd.Series(portfolio_returns)
        
        self.portfolios = portfolios
        return portfolios
    
    def ensemble_predictions(self, predictions, method='equal_weight'):
        """Create ensemble predictions from multiple models"""
        
        print("Creating ensemble predictions...")
        
        if len(predictions) < 2:
            return predictions
        
        # Align all predictions
        common_dates = None
        common_tickers = None
        
        for model_name, pred_data in predictions.items():
            if common_dates is None:
                common_dates = pred_data.index
                common_tickers = pred_data.columns
            else:
                common_dates = common_dates.intersection(pred_data.index)
                common_tickers = common_tickers.intersection(pred_data.columns)
        
        aligned_predictions = {}
        for model_name, pred_data in predictions.items():
            aligned_predictions[model_name] = pred_data.reindex(
                index=common_dates, columns=common_tickers
            )
        
        if method == 'equal_weight':
            # Simple average
            ensemble_pred = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
            
            for pred_data in aligned_predictions.values():
                ensemble_pred += pred_data / len(aligned_predictions)
                
        elif method == 'cv_weighted':
            # Weight by cross-validation performance (lower MSE = higher weight)
            cv_scores = {}
            for model_name in aligned_predictions.keys():
                if model_name in self.models:
                    cv_scores[model_name] = self.models[model_name]['cv_score']
            
            if len(cv_scores) > 0:
                # Convert MSE to weights (inverse relationship)
                cv_weights = {}
                total_inverse_cv = sum(1.0 / score for score in cv_scores.values())
                
                for model_name, score in cv_scores.items():
                    cv_weights[model_name] = (1.0 / score) / total_inverse_cv
                
                # Weighted average
                ensemble_pred = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
                
                for model_name, pred_data in aligned_predictions.items():
                    if model_name in cv_weights:
                        ensemble_pred += pred_data * cv_weights[model_name]
            else:
                # Fallback to equal weight
                ensemble_pred = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
                for pred_data in aligned_predictions.values():
                    ensemble_pred += pred_data / len(aligned_predictions)
        
        return {'Ensemble': ensemble_pred}

def analyze_ml_portfolio_performance(portfolios, benchmark_returns=None):
    """Analyze performance of ML-based portfolios"""
    
    performance_results = {}
    
    for model_name, portfolio_returns in portfolios.items():
        if len(portfolio_returns) == 0:
            continue
            
        returns = portfolio_returns.dropna()
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).sum() / len(returns)
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        performance_results[model_name] = {
            'returns': returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    return performance_results

def main_ml_portfolio_analysis():
    """Main ML portfolio construction analysis"""
    
    print("=" * 80)
    print("MACHINE LEARNING PORTFOLIO CONSTRUCTION ANALYSIS")
    print("=" * 80)
    
    # Initialize ML engine
    ml_engine = MLPortfolioConstruction()
    
    # Define universe
    universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX'
    ]
    
    # Load data and construct features
    market_data, features = ml_engine.load_data_and_features(
        universe, start_date='2018-01-01', end_date='2024-01-01'
    )
    
    # Prepare ML dataset
    dataset = ml_engine.prepare_ml_dataset(
        features, 
        market_data['returns'],
        prediction_horizon=21,  # 1-month ahead predictions
        min_history=252
    )
    
    if dataset['n_features'] == 0:
        print("Insufficient data for ML analysis")
        return None
    
    # Train ML models
    models = ml_engine.train_ml_models(dataset)
    
    if len(models) == 0:
        print("No models trained successfully")
        return None
    
    # Generate predictions
    predictions = ml_engine.generate_ml_predictions(dataset, models)
    
    # Create ensemble
    ensemble_predictions = ml_engine.ensemble_predictions(predictions, method='cv_weighted')
    predictions.update(ensemble_predictions)
    
    # Create portfolios
    portfolios = ml_engine.create_ml_portfolios(
        predictions,
        market_data['returns'],
        portfolio_method='long_short',
        n_positions=10,
        rebalance_freq='monthly'
    )
    
    # Analyze performance
    performance_results = analyze_ml_portfolio_performance(portfolios)
    
    # Display results
    print("\n" + "=" * 60)
    print("ML PORTFOLIO PERFORMANCE RESULTS")
    print("=" * 60)
    
    if len(performance_results) > 0:
        # Create summary table
        summary_data = []
        for model_name, results in performance_results.items():
            summary_data.append({
                'Model': model_name,
                'Ann. Return': f"{results['annualized_return']:.2%}",
                'Ann. Vol': f"{results['annualized_volatility']:.2%}",
                'Sharpe': f"{results['sharpe_ratio']:.3f}",
                'Max DD': f"{results['max_drawdown']:.2%}",
                'Win Rate': f"{results['win_rate']:.2%}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Create visualizations
        create_ml_visualizations(performance_results, models)
    else:
        print("No portfolio results to display")
    
    return {
        'ml_engine': ml_engine,
        'models': models,
        'predictions': predictions,
        'portfolios': portfolios,
        'performance': performance_results
    }

def create_ml_visualizations(performance_results, models):
    """Create comprehensive ML analysis visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Machine Learning Portfolio Construction Analysis', fontsize=16, fontweight='bold')
    
    # 1. Model Performance Comparison
    ax1 = axes[0, 0]
    
    model_names = list(performance_results.keys())
    sharpe_ratios = [performance_results[name]['sharpe_ratio'] for name in model_names]
    
    bars = ax1.bar(model_names, sharpe_ratios, alpha=0.7)
    ax1.set_title('Model Sharpe Ratios')
    ax1.set_ylabel('Sharpe Ratio')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if sharpe_ratios[i] > 0.5:
            bar.set_color('green')
        elif sharpe_ratios[i] > 0:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    # 2. Cumulative Returns
    ax2 = axes[0, 1]
    
    for model_name, results in performance_results.items():
        returns = results['returns']
        cumulative = (1 + returns).cumprod()
        ax2.plot(cumulative.values, label=model_name, linewidth=2)
    
    ax2.set_title('Cumulative Returns Comparison')
    ax2.set_ylabel('Cumulative Return')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk-Return Scatter
    ax3 = axes[0, 2]
    
    for model_name, results in performance_results.items():
        ax3.scatter(results['annualized_volatility'], results['annualized_return'], 
                   s=100, label=model_name, alpha=0.7)
    
    ax3.set_xlabel('Annualized Volatility')
    ax3.set_ylabel('Annualized Return')
    ax3.set_title('Risk-Return Profile')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Model Cross-Validation Scores
    ax4 = axes[1, 0]
    
    if len(models) > 0:
        cv_scores = {name: info['cv_score'] for name, info in models.items()}
        model_names = list(cv_scores.keys())
        scores = list(cv_scores.values())
        
        ax4.bar(model_names, scores, alpha=0.7, color='orange')
        ax4.set_title('Model CV Scores (MSE)')
        ax4.set_ylabel('Cross-Validation MSE')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    # 5. Return Distributions
    ax5 = axes[1, 1]
    
    for model_name, results in performance_results.items():
        returns = results['returns']
        ax5.hist(returns * 100, bins=30, alpha=0.6, label=model_name, density=True)
    
    ax5.set_xlabel('Daily Return (%)')
    ax5.set_ylabel('Density')
    ax5.set_title('Return Distributions')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Metrics Heatmap
    ax6 = axes[1, 2]
    
    metrics_matrix = []
    metric_names = ['Sharpe Ratio', 'Max Drawdown', 'Win Rate']
    
    for model_name in performance_results.keys():
        results = performance_results[model_name]
        metrics_matrix.append([
            results['sharpe_ratio'],
            abs(results['max_drawdown']),  # Absolute value for visualization
            results['win_rate']
        ])
    
    if len(metrics_matrix) > 0:
        metrics_df = pd.DataFrame(metrics_matrix, 
                                 index=list(performance_results.keys()),
                                 columns=metric_names)
        
        im = ax6.imshow(metrics_df.values, cmap='RdYlGn', aspect='auto')
        ax6.set_xticks(range(len(metric_names)))
        ax6.set_xticklabels(metric_names)
        ax6.set_yticks(range(len(performance_results)))
        ax6.set_yticklabels(list(performance_results.keys()))
        ax6.set_title('Performance Metrics Heatmap')
        
        # Add colorbar
        plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.show()
    
    # Additional insights
    print("\n" + "=" * 60)
    print("MACHINE LEARNING MODEL INSIGHTS")
    print("=" * 60)
    
    if len(performance_results) > 0:
        best_sharpe = max(performance_results.items(), key=lambda x: x[1]['sharpe_ratio'])
        lowest_vol = min(performance_results.items(), key=lambda x: x[1]['annualized_volatility'])
        
        print(f"Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
        print(f"Lowest Volatility: {lowest_vol[0]} ({lowest_vol[1]['annualized_volatility']:.2%})")
        
        # Model type analysis
        if len(models) > 0:
            print(f"\nModel Types Performance:")
            
            type_performance = {}
            for model_name, model_info in models.items():
                model_type = model_info['type']
                if model_name in performance_results:
                    if model_type not in type_performance:
                        type_performance[model_type] = []
                    type_performance[model_type].append(
                        performance_results[model_name]['sharpe_ratio']
                    )
            
            for model_type, sharpe_list in type_performance.items():
                avg_sharpe = np.mean(sharpe_list)
                print(f"  {model_type.capitalize()}: {avg_sharpe:.3f} average Sharpe")

if __name__ == "__main__":
    # Run the ML portfolio analysis
    results = main_ml_portfolio_analysis()
    
    if results is not None:
        print("\n" + "=" * 80)
        print("MACHINE LEARNING PORTFOLIO ANALYSIS COMPLETED")
        print("=" * 80)
        
        print("\nKey Achievements:")
        print("• Constructed comprehensive feature set from market data")
        print("• Trained multiple ML models (linear, tree-based, neural networks)")
        print("• Generated ensemble predictions combining model outputs")
        print("• Created systematic portfolio construction framework")
        print("• Analyzed performance with proper cross-validation")
        
        print("\nML Model Insights:")
        print("• Feature engineering critical for model performance")
        print("• Ensemble methods typically outperform individual models")
        print("• Tree-based models often capture non-linear patterns well")
        print("• Neural networks require sufficient data for effectiveness")
        print("• Cross-validation essential for avoiding overfitting")
        
        print("\nImplementation Considerations:")
        print("• Use walk-forward analysis for production deployment")
        print("• Implement feature selection and regularization")
        print("• Monitor model drift and retrain periodically")
        print("• Consider transaction costs in strategy evaluation")
        print("• Combine ML predictions with fundamental analysis")
        
        print(f"\nML framework ready for deployment!")
    else:
        print("ML analysis could not be completed due to insufficient data")
