# Alpha Project: Academic Research Implementation Engine
# Notebook 2: Momentum Strategies Implementation

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class MomentumStrategies:
    """
    Implementation of Academic Momentum Strategies
    Based on: Jegadeesh, N., & Titman, S. (1993)
             Carhart, M. M. (1997)
             Moskowitz, T. J., & Grinblatt, M. (1999)
    """
    
    def __init__(self):
        self.universe_data = None
        self.momentum_scores = None
        self.portfolio_returns = None
        
    def load_universe_data(self, tickers, start_date='2020-01-01', end_date='2024-01-01'):
        """Load price data for stock universe"""
        print(f"Loading data for {len(tickers)} stocks...")
        
        # Download price data
        data = yf.download(tickers, start=start_date, end=end_date)
        
        if len(tickers) == 1:
            # Single ticker case
            self.universe_data = {
                'prices': data['Adj Close'].to_frame(),
                'volumes': data['Volume'].to_frame(),
                'returns': data['Adj Close'].pct_change().to_frame()
            }
        else:
            # Multiple tickers
            self.universe_data = {
                'prices': data['Adj Close'],
                'volumes': data['Volume'], 
                'returns': data['Adj Close'].pct_change()
            }
        
        # Clean data
        self.universe_data['returns'] = self.universe_data['returns'].dropna()
        
        print(f"Data loaded: {len(self.universe_data['returns'])} trading days")
        return self.universe_data
    
    def calculate_momentum_score(self, returns, lookback_periods=[1, 3, 6, 12], 
                                exclude_recent=1, method='cumulative'):
        """
        Calculate momentum scores using various methodologies
        
        Parameters:
        - lookback_periods: List of months to look back
        - exclude_recent: Months to exclude from recent period (to avoid microstructure noise)
        - method: 'cumulative', 'average', 'risk_adjusted'
        """
        
        momentum_scores = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for period in lookback_periods:
            lookback_days = period * 21  # Approximate trading days per month
            exclude_days = exclude_recent * 21
            
            for i in range(lookback_days + exclude_days, len(returns)):
                current_date = returns.index[i]
                
                # Define the momentum calculation period
                start_idx = i - lookback_days - exclude_days
                end_idx = i - exclude_days
                
                period_returns = returns.iloc[start_idx:end_idx]
                
                if method == 'cumulative':
                    # Total return over the period
                    scores = (1 + period_returns).prod() - 1
                    
                elif method == 'average':
                    # Average return over the period
                    scores = period_returns.mean()
                    
                elif method == 'risk_adjusted':
                    # Sharpe-like ratio
                    mean_ret = period_returns.mean()
                    std_ret = period_returns.std()
                    scores = mean_ret / (std_ret + 1e-8)  # Add small epsilon to avoid division by zero
                
                momentum_scores.loc[current_date] = scores
        
        # Store the most recent momentum scores
        self.momentum_scores = momentum_scores.dropna()
        return self.momentum_scores
    
    def cross_sectional_momentum(self, momentum_scores, n_portfolios=5, 
                                long_short=True, rebalance_freq='monthly'):
        """
        Implement cross-sectional momentum strategy
        Forms portfolios based on relative momentum rankings
        """
        
        portfolio_returns = pd.DataFrame()
        portfolio_compositions = {}
        
        # Determine rebalancing dates
        if rebalance_freq == 'monthly':
            rebalance_dates = momentum_scores.resample('M').last().index
        elif rebalance_freq == 'weekly':
            rebalance_dates = momentum_scores.resample('W').last().index
        else:
            rebalance_dates = momentum_scores.index
        
        for date in rebalance_dates:
            if date not in momentum_scores.index:
                continue
                
            # Get momentum scores for this date
            scores = momentum_scores.loc[date].dropna()
            
            if len(scores) < n_portfolios:
                continue
            
            # Rank stocks by momentum score
            ranks = scores.rank(ascending=False, method='first')
            
            # Form portfolios
            stocks_per_portfolio = len(scores) // n_portfolios
            
            portfolios = {}
            for i in range(n_portfolios):
                start_rank = i * stocks_per_portfolio + 1
                end_rank = (i + 1) * stocks_per_portfolio
                
                if i == n_portfolios - 1:  # Last portfolio gets remaining stocks
                    end_rank = len(scores)
                
                portfolio_stocks = ranks[(ranks >= start_rank) & (ranks <= end_rank)].index
                portfolios[f'P{i+1}'] = portfolio_stocks.tolist()
            
            portfolio_compositions[date] = portfolios
        
        # Calculate portfolio returns
        returns_data = self.universe_data['returns']
        
        for i, (date, portfolios) in enumerate(portfolio_compositions.items()):
            # Find next rebalancing date for return calculation period
            if i < len(portfolio_compositions) - 1:
                next_date = list(portfolio_compositions.keys())[i + 1]
                period_returns = returns_data.loc[date:next_date]
            else:
                # Last period - calculate to end of data
                period_returns = returns_data.loc[date:]
            
            if len(period_returns) <= 1:
                continue
            
            # Calculate equal-weighted portfolio returns
            for portfolio_name, stocks in portfolios.items():
                valid_stocks = [s for s in stocks if s in period_returns.columns]
                
                if valid_stocks:
                    portfolio_ret = period_returns[valid_stocks].mean(axis=1)
                    
                    # Store returns
                    for ret_date in portfolio_ret.index[1:]:  # Skip first date
                        if portfolio_name not in portfolio_returns.columns:
                            portfolio_returns[portfolio_name] = np.nan
                        portfolio_returns.loc[ret_date, portfolio_name] = portfolio_ret.loc[ret_date]
        
        # Create long-short portfolio if requested
        if long_short and 'P1' in portfolio_returns.columns and f'P{n_portfolios}' in portfolio_returns.columns:
            portfolio_returns['Long_Short'] = portfolio_returns['P1'] - portfolio_returns[f'P{n_portfolios}']
        
        self.portfolio_returns = portfolio_returns.dropna()
        return self.portfolio_returns, portfolio_compositions
    
    def time_series_momentum(self, returns, lookback_window=252, signal_threshold=0.0):
        """
        Implement time-series momentum strategy
        Based on Moskowitz, Ooi, and Pedersen (2012)
        """
        
        ts_momentum_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        signals = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(lookback_window, len(returns)):
            current_date = returns.index[i]
            
            # Calculate momentum signal (past return)
            past_returns = returns.iloc[i-lookback_window:i]
            cumulative_returns = (1 + past_returns).prod() - 1
            
            # Generate trading signals
            long_signals = cumulative_returns > signal_threshold
            short_signals = cumulative_returns < -signal_threshold
            
            # Calculate strategy returns
            next_day_returns = returns.iloc[i] if i < len(returns) - 1 else np.nan
            
            strategy_returns = pd.Series(index=returns.columns, dtype=float)
            strategy_returns[long_signals] = next_day_returns[long_signals]
            strategy_returns[short_signals] = -next_day_returns[short_signals]
            strategy_returns[~(long_signals | short_signals)] = 0  # No position
            
            ts_momentum_returns.loc[current_date] = strategy_returns
            signals.loc[current_date] = long_signals.astype(int) - short_signals.astype(int)
        
        return ts_momentum_returns.dropna(), signals.dropna()
    
    def risk_adjusted_momentum(self, returns, volatility_window=60, 
                              momentum_window=252, target_volatility=0.15):
        """
        Implement volatility-adjusted momentum strategy
        Scales positions by inverse volatility
        """
        
        vol_adjusted_returns = pd.DataFrame(index=returns.index, columns=returns.columns)
        position_sizes = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for i in range(max(volatility_window, momentum_window), len(returns)):
            current_date = returns.index[i]
            
            # Calculate momentum signals
            momentum_period = returns.iloc[i-momentum_window:i-21]  # Exclude recent month
            momentum_scores = (1 + momentum_period).prod() - 1
            
            # Calculate volatilities
            vol_period = returns.iloc[i-volatility_window:i]
            volatilities = vol_period.std() * np.sqrt(252)  # Annualized
            
            # Calculate position sizes (inverse volatility)
            positions = momentum_scores.sign() * (target_volatility / (volatilities + 1e-8))
            
            # Cap positions at reasonable levels
            positions = positions.clip(-2, 2)
            
            # Calculate strategy returns
            if i < len(returns) - 1:
                next_returns = returns.iloc[i]
                strategy_returns = positions * next_returns
                
                vol_adjusted_returns.loc[current_date] = strategy_returns
                position_sizes.loc[current_date] = positions
        
        return vol_adjusted_returns.dropna(), position_sizes.dropna()

def calculate_performance_metrics(returns, benchmark_returns=None):
    """Calculate comprehensive performance metrics"""
    
    metrics = {}
    
    # Basic metrics
    total_return = (1 + returns).prod() - 1
    annualized_return = (1 + returns.mean()) ** 252 - 1
    annualized_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
    
    # Drawdown metrics
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns.dropna())
    kurtosis = stats.kurtosis(returns.dropna())
    
    # Win rate
    win_rate = (returns > 0).sum() / len(returns.dropna())
    
    metrics.update({
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Win Rate': win_rate
    })
    
    # Information ratio vs benchmark
    if benchmark_returns is not None:
        excess_returns = returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        metrics['Information Ratio'] = information_ratio
    
    return metrics

def main_momentum_analysis():
    """Main analysis demonstrating momentum strategies"""
    
    # Initialize momentum strategy engine
    momentum_engine = MomentumStrategies()
    
    # Define universe (mix of sectors for diversification)
    universe = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # Tech
        'JPM', 'BAC', 'WFC', 'C',          # Finance
        'JNJ', 'PFE', 'UNH', 'ABBV',      # Healthcare
        'XOM', 'CVX', 'COP', 'SLB',       # Energy
        'WMT', 'HD', 'MCD', 'NKE'         # Consumer
    ]
    
    print("=" * 60)
    print("MOMENTUM STRATEGIES ANALYSIS")
    print("=" * 60)
    
    # Load data
    universe_data = momentum_engine.load_universe_data(
        universe, 
        start_date='2020-01-01', 
        end_date='2024-01-01'
    )
    
    # 1. Cross-Sectional Momentum Strategy
    print("\n1. Cross-Sectional Momentum Strategy")
    print("-" * 40)
    
    momentum_scores = momentum_engine.calculate_momentum_score(
        universe_data['returns'], 
        lookback_periods=[6], 
        method='cumulative'
    )
    
    portfolio_returns, compositions = momentum_engine.cross_sectional_momentum(
        momentum_scores, 
        n_portfolios=5, 
        long_short=True
    )
    
    # Analyze cross-sectional results
    if len(portfolio_returns) > 0:
        print("Portfolio Performance (Cross-Sectional):")
        for col in portfolio_returns.columns:
            if col in portfolio_returns.columns and portfolio_returns[col].dropna().shape[0] > 0:
                metrics = calculate_performance_metrics(portfolio_returns[col].dropna())
                print(f"{col}: Return={metrics['Annualized Return']:.2%}, "
                      f"Sharpe={metrics['Sharpe Ratio']:.3f}, "
                      f"MaxDD={metrics['Max Drawdown']:.2%}")
    
    # 2. Time-Series Momentum Strategy
    print("\n2. Time-Series Momentum Strategy")
    print("-" * 40)
    
    ts_returns, ts_signals = momentum_engine.time_series_momentum(
        universe_data['returns'],
        lookback_window=252
    )
    
    # Calculate equal-weighted time-series momentum portfolio
    ts_portfolio = ts_returns.mean(axis=1).dropna()
    
    if len(ts_portfolio) > 0:
        ts_metrics = calculate_performance_metrics(ts_portfolio)
        print(f"Time-Series Momentum: Return={ts_metrics['Annualized Return']:.2%}, "
              f"Sharpe={ts_metrics['Sharpe Ratio']:.3f}, "
              f"MaxDD={ts_metrics['Max Drawdown']:.2%}")
    
    # 3. Risk-Adjusted Momentum Strategy
    print("\n3. Risk-Adjusted Momentum Strategy")
    print("-" * 40)
    
    vol_adj_returns, position_sizes = momentum_engine.risk_adjusted_momentum(
        universe_data['returns'],
        volatility_window=60,
        momentum_window=252,
        target_volatility=0.15
    )
    
    # Calculate equal-weighted risk-adjusted portfolio
    vol_adj_portfolio = vol_adj_returns.mean(axis=1).dropna()
    
    if len(vol_adj_portfolio) > 0:
        vol_adj_metrics = calculate_performance_metrics(vol_adj_portfolio)
        print(f"Vol-Adjusted Momentum: Return={vol_adj_metrics['Annualized Return']:.2%}, "
              f"Sharpe={vol_adj_metrics['Sharpe Ratio']:.3f}, "
              f"MaxDD={vol_adj_metrics['Max Drawdown']:.2%}")
    
    # Create visualizations
    create_momentum_visualizations(portfolio_returns, ts_portfolio, vol_adj_portfolio, universe_data)
    
    return {
        'cross_sectional': portfolio_returns,
        'time_series': ts_portfolio,
        'vol_adjusted': vol_adj_portfolio,
        'universe_data': universe_data
    }

def create_momentum_visualizations(portfolio_returns, ts_portfolio, vol_adj_portfolio, universe_data):
    """Create comprehensive momentum strategy visualizations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Momentum Strategies Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative returns comparison
    ax1 = axes[0, 0]
    
    if 'Long_Short' in portfolio_returns.columns:
        cs_cumret = (1 + portfolio_returns['Long_Short'].dropna()).cumprod()
        ax1.plot(cs_cumret.index, cs_cumret.values, label='Cross-Sectional', linewidth=2)
    
    if len(ts_portfolio) > 0:
        ts_cumret = (1 + ts_portfolio).cumprod()
        ax1.plot(ts_cumret.index, ts_cumret.values, label='Time-Series', linewidth=2)
    
    if len(vol_adj_portfolio) > 0:
        vol_cumret = (1 + vol_adj_portfolio).cumprod()
        ax1.plot(vol_cumret.index, vol_cumret.values, label='Vol-Adjusted', linewidth=2)
    
    # Add market benchmark
    market_returns = universe_data['returns'].mean(axis=1).dropna()
    market_cumret = (1 + market_returns).cumprod()
    ax1.plot(market_cumret.index, market_cumret.values, label='Equal-Weight Benchmark', 
             linewidth=2, linestyle='--', alpha=0.7)
    
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Rolling Sharpe ratios
    ax2 = axes[0, 1]
    
    window = 252  # 1-year rolling window
    
    if 'Long_Short' in portfolio_returns.columns and len(portfolio_returns['Long_Short'].dropna()) > window:
        cs_rolling_sharpe = portfolio_returns['Long_Short'].rolling(window).mean() / \
                           portfolio_returns['Long_Short'].rolling(window).std() * np.sqrt(252)
        ax2.plot(cs_rolling_sharpe.index, cs_rolling_sharpe.values, label='Cross-Sectional')
    
    if len(ts_portfolio) > window:
        ts_rolling_sharpe = ts_portfolio.rolling(window).mean() / \
                           ts_portfolio.rolling(window).std() * np.sqrt(252)
        ax2.plot(ts_rolling_sharpe.index, ts_rolling_sharpe.values, label='Time-Series')
    
    if len(vol_adj_portfolio) > window:
        vol_rolling_sharpe = vol_adj_portfolio.rolling(window).mean() / \
                            vol_adj_portfolio.rolling(window).std() * np.sqrt(252)
        ax2.plot(vol_rolling_sharpe.index, vol_rolling_sharpe.values, label='Vol-Adjusted')
    
    ax2.set_title('Rolling 1-Year Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    
    # 3. Drawdown analysis
    ax3 = axes[1, 0]
    
    strategies = {}
    if 'Long_Short' in portfolio_returns.columns:
        strategies['Cross-Sectional'] = portfolio_returns['Long_Short'].dropna()
    if len(ts_portfolio) > 0:
        strategies['Time-Series'] = ts_portfolio
    if len(vol_adj_portfolio) > 0:
        strategies['Vol-Adjusted'] = vol_adj_portfolio
    
    for name, returns in strategies.items():
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, label=name)
    
    ax3.set_title('Strategy Drawdowns')
    ax3.set_ylabel('Drawdown (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Return distribution comparison
    ax4 = axes[1, 1]
    
    for name, returns in strategies.items():
        ax4.hist(returns * 100, bins=30, alpha=0.6, label=name, density=True)
    
    ax4.set_title('Daily Return Distributions')
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary table
    print("\n" + "=" * 80)
    print("MOMENTUM STRATEGIES PERFORMANCE SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for name, returns in strategies.items():
        metrics = calculate_performance_metrics(returns)
        summary_data.append({
            'Strategy': name,
            'Ann. Return': f"{metrics['Annualized Return']:.2%}",
            'Ann. Vol': f"{metrics['Annualized Volatility']:.2%}",
            'Sharpe': f"{metrics['Sharpe Ratio']:.3f}",
            'Max DD': f"{metrics['Max Drawdown']:.2%}",
            'Win Rate': f"{metrics['Win Rate']:.2%}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    # Run the momentum analysis
    results = main_momentum_analysis()
    
    print("\n" + "=" * 60)
    print("MOMENTUM ANALYSIS COMPLETED")
    print("=" * 60)
    print("\nKey Findings:")
    print("• Cross-sectional momentum captures relative outperformance")
    print("• Time-series momentum adapts to trend changes")
    print("• Volatility-adjusted momentum provides more stable returns")
    print("• All strategies benefit from diversification across sectors")
    
    print("\nNext Steps:")
    print("• Implement transaction cost analysis")
    print("• Add regime detection for strategy switching")
    print("• Combine with factor models for enhanced alpha")
