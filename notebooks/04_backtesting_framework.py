# Alpha Project: Academic Research Implementation Engine
# Notebook 4: Systematic Backtesting Framework

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class BacktestEngine:
    """
    Comprehensive backtesting framework for academic strategy implementation
    Includes transaction costs, position sizing, risk management, and performance attribution
    """
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001, 
                 max_position_size=0.1, rebalance_frequency='monthly'):
        
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost  # 10 bps default
        self.max_position_size = max_position_size  # 10% max per position
        self.rebalance_frequency = rebalance_frequency
        
        # Results storage
        self.portfolio_returns = None
        self.positions = None
        self.turnover = None
        self.performance_metrics = None
        
    def calculate_transaction_costs(self, old_weights, new_weights, portfolio_value):
        """Calculate transaction costs based on portfolio turnover"""
        
        if old_weights is None:
            # Initial portfolio - only pay costs on initial positions
            turnover = abs(new_weights).sum()
        else:
            # Calculate turnover as sum of absolute weight changes
            weight_changes = abs(new_weights - old_weights)
            turnover = weight_changes.sum()
        
        # Transaction cost = turnover * cost rate * portfolio value
        transaction_cost = turnover * self.transaction_cost * portfolio_value
        
        return transaction_cost, turnover
    
    def apply_position_sizing_rules(self, raw_weights):
        """Apply position sizing constraints"""
        
        # Cap individual positions
        capped_weights = raw_weights.clip(-self.max_position_size, self.max_position_size)
        
        # Normalize to ensure weights sum to target (typically 1.0 for long-only)
        if capped_weights.sum() != 0:
            capped_weights = capped_weights / capped_weights.sum()
        
        return capped_weights
    
    def backtest_strategy(self, returns_data, signals_data, strategy_name="Strategy"):
        """
        Run comprehensive backtest with transaction costs and risk management
        
        Parameters:
        - returns_data: DataFrame of asset returns
        - signals_data: DataFrame of strategy signals/weights
        - strategy_name: Name for the strategy
        """
        
        print(f"Running backtest for {strategy_name}...")
        
        # Align data
        common_dates = returns_data.index.intersection(signals_data.index)
        returns = returns_data.loc[common_dates]
        signals = signals_data.loc[common_dates]
        
        # Initialize tracking variables
        portfolio_values = []
        portfolio_returns = []
        gross_returns = []
        net_returns = []
        positions_over_time = []
        turnover_over_time = []
        
        current_portfolio_value = self.initial_capital
        previous_weights = None
        
        # Determine rebalancing dates
        if self.rebalance_frequency == 'daily':
            rebalance_dates = common_dates
        elif self.rebalance_frequency == 'weekly':
            rebalance_dates = common_dates[::5]  # Every 5 days
        elif self.rebalance_frequency == 'monthly':
            rebalance_dates = returns.resample('M').last().index
        elif self.rebalance_frequency == 'quarterly':
            rebalance_dates = returns.resample('Q').last().index
        else:
            rebalance_dates = common_dates
        
        current_weights = pd.Series(0.0, index=returns.columns)
        
        for i, date in enumerate(common_dates):
            
            # Check if it's a rebalancing date
            if date in rebalance_dates:
                
                # Get target weights from signals
                if date in signals.index:
                    raw_weights = signals.loc[date]
                    
                    # Handle NaN values
                    raw_weights = raw_weights.fillna(0)
                    
                    # Apply position sizing rules
                    target_weights = self.apply_position_sizing_rules(raw_weights)
                    
                    # Calculate transaction costs
                    transaction_cost, turnover = self.calculate_transaction_costs(
                        previous_weights, target_weights, current_portfolio_value
                    )
                    
                    # Update portfolio value after transaction costs
                    current_portfolio_value -= transaction_cost
                    
                    # Update weights
                    previous_weights = current_weights.copy()
                    current_weights = target_weights
                    
                    # Track turnover
                    turnover_over_time.append({
                        'date': date,
                        'turnover': turnover,
                        'transaction_cost': transaction_cost
                    })
            
            # Calculate daily returns
            if date in returns.index:
                daily_returns = returns.loc[date]
                
                # Calculate gross return (before transaction costs)
                gross_return = (current_weights * daily_returns).sum()
                
                # Update portfolio value
                current_portfolio_value *= (1 + gross_return)
                
                # Calculate net return (actual portfolio return)
                if i == 0:
                    net_return = 0.0
                else:
                    net_return = (current_portfolio_value / portfolio_values[-1]) - 1
                
                # Store results
                portfolio_values.append(current_portfolio_value)
                gross_returns.append(gross_return)
                net_returns.append(net_return)
                
                # Store position information
                positions_over_time.append({
                    'date': date,
                    'portfolio_value': current_portfolio_value,
                    **dict(current_weights)
                })
        
        # Create results DataFrames
        self.portfolio_returns = pd.Series(net_returns[1:], index=common_dates[1:])
        self.gross_returns = pd.Series(gross_returns[1:], index=common_dates[1:])
        
        self.positions = pd.DataFrame(positions_over_time).set_index('date')
        self.turnover = pd.DataFrame(turnover_over_time).set_index('date')
        
        print(f"Backtest completed. Final portfolio value: ${current_portfolio_value:,.2f}")
        
        return self.portfolio_returns
    
    def calculate_performance_metrics(self, benchmark_returns=None):
        """Calculate comprehensive performance metrics"""
        
        if self.portfolio_returns is None:
            raise ValueError("No backtest results available. Run backtest first.")
        
        returns = self.portfolio_returns.dropna()
        
        # Basic return metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calculate Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Higher moments
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Hit rate and profit factor
        win_rate = (returns > 0).sum() / len(returns)
        avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).any() else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        # Value at Risk (VaR) and Expected Shortfall (ES)
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        es_95 = returns[returns <= var_95].mean() if (returns <= var_95).any() else var_95
        
        # Maximum consecutive wins/losses
        returns_sign = np.sign(returns)
        consecutive_wins = self._max_consecutive(returns_sign, 1)
        consecutive_losses = self._max_consecutive(returns_sign, -1)
        
        # Trading statistics
        if self.turnover is not None and len(self.turnover) > 0:
            avg_turnover = self.turnover['turnover'].mean()
            total_transaction_costs = self.turnover['transaction_cost'].sum()
            cost_drag = total_transaction_costs / self.initial_capital
        else:
            avg_turnover = 0
            total_transaction_costs = 0
            cost_drag = 0
        
        metrics = {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Annualized Volatility': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Max Drawdown': max_drawdown,
            'VaR 95%': var_95,
            'VaR 99%': var_99,
            'Expected Shortfall 95%': es_95,
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Max Consecutive Wins': consecutive_wins,
            'Max Consecutive Losses': consecutive_losses,
            'Average Turnover': avg_turnover,
            'Total Transaction Costs': total_transaction_costs,
            'Cost Drag': cost_drag
        }
        
        # Benchmark comparison
        if benchmark_returns is not None:
            aligned_benchmark = benchmark_returns.reindex(returns.index).fillna(0)
            excess_returns = returns - aligned_benchmark
            tracking_error = excess_returns.std() * np.sqrt(252)
            information_ratio = (excess_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
            
            # Beta and Alpha
            covariance = np.cov(returns, aligned_benchmark)[0, 1]
            benchmark_variance = np.var(aligned_benchmark)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            alpha = annualized_return - beta * (aligned_benchmark.mean() * 252)
            
            metrics.update({
                'Information Ratio': information_ratio,
                'Tracking Error': tracking_error,
                'Beta': beta,
                'Alpha': alpha
            })
        
        self.performance_metrics = metrics
        return metrics
    
    def _max_consecutive(self, series, value):
        """Calculate maximum consecutive occurrences of a value"""
        consecutive_count = 0
        max_consecutive = 0
        
        for v in series:
            if v == value:
                consecutive_count += 1
                max_consecutive = max(max_consecutive, consecutive_count)
            else:
                consecutive_count = 0
        
        return max_consecutive
    
    def generate_performance_report(self, strategy_name="Strategy", benchmark_name="Benchmark"):
        """Generate a comprehensive performance report"""
        
        if self.performance_metrics is None:
            self.calculate_performance_metrics()
        
        print("=" * 80)
        print(f"PERFORMANCE REPORT: {strategy_name}")
        print("=" * 80)
        
        # Return metrics
        print("\nRETURN METRICS:")
        print("-" * 40)
        print(f"Total Return:               {self.performance_metrics['Total Return']:>10.2%}")
        print(f"Annualized Return:          {self.performance_metrics['Annualized Return']:>10.2%}")
        print(f"Annualized Volatility:      {self.performance_metrics['Annualized Volatility']:>10.2%}")
        
        # Risk-adjusted metrics
        print("\nRISK-ADJUSTED METRICS:")
        print("-" * 40)
        print(f"Sharpe Ratio:               {self.performance_metrics['Sharpe Ratio']:>10.3f}")
        print(f"Sortino Ratio:              {self.performance_metrics['Sortino Ratio']:>10.3f}")
        print(f"Calmar Ratio:               {self.performance_metrics['Calmar Ratio']:>10.3f}")
        
        # Risk metrics
        print("\nRISK METRICS:")
        print("-" * 40)
        print(f"Maximum Drawdown:           {self.performance_metrics['Max Drawdown']:>10.2%}")
        print(f"VaR (95%):                  {self.performance_metrics['VaR 95%']:>10.2%}")
        print(f"VaR (99%):                  {self.performance_metrics['VaR 99%']:>10.2%}")
        print(f"Expected Shortfall (95%):   {self.performance_metrics['Expected Shortfall 95%']:>10.2%}")
        
        # Distribution metrics
        print("\nDISTRIBUTION METRICS:")
        print("-" * 40)
        print(f"Skewness:                   {self.performance_metrics['Skewness']:>10.3f}")
        print(f"Kurtosis:                   {self.performance_metrics['Kurtosis']:>10.3f}")
        print(f"Win Rate:                   {self.performance_metrics['Win Rate']:>10.2%}")
        print(f"Profit Factor:              {self.performance_metrics['Profit Factor']:>10.3f}")
        
        # Trading metrics
        print("\nTRADING METRICS:")
        print("-" * 40)
        print(f"Average Turnover:           {self.performance_metrics['Average Turnover']:>10.2%}")
        print(f"Total Transaction Costs:    ${self.performance_metrics['Total Transaction Costs']:>9,.0f}")
        print(f"Cost Drag:                  {self.performance_metrics['Cost Drag']:>10.2%}")
        
        # Benchmark comparison (if available)
        if 'Information Ratio' in self.performance_metrics:
            print(f"\nBENCHMARK COMPARISON ({benchmark_name}):")
            print("-" * 40)
            print(f"Information Ratio:          {self.performance_metrics['Information Ratio']:>10.3f}")
            print(f"Tracking Error:             {self.performance_metrics['Tracking Error']:>10.2%}")
            print(f"Beta:                       {self.performance_metrics['Beta']:>10.3f}")
            print(f"Alpha:                      {self.performance_metrics['Alpha']:>10.2%}")

def create_comprehensive_visualizations(backtest_engine, strategy_name="Strategy"):
    """Create comprehensive backtest visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle(f'{strategy_name} - Comprehensive Backtest Analysis', fontsize=16, fontweight='bold')
    
    returns = backtest_engine.portfolio_returns.dropna()
    
    # 1. Cumulative Returns
    ax1 = axes[0, 0]
    cumulative_returns = (1 + returns).cumprod()
    ax1.plot(cumulative_returns.index, cumulative_returns.values, linewidth=2, color='navy')
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[0, 1]
    rolling_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.6, color='red')
    ax2.set_title('Drawdown Analysis')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratio
    ax3 = axes[1, 0]
    window = 252  # 1-year rolling
    if len(returns) > window:
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        ax3.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=2, color='green')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('Rolling 1-Year Sharpe Ratio')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.grid(True, alpha=0.3)
    
    # 4. Return Distribution
    ax4 = axes[1, 1]
    ax4.hist(returns * 100, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean()*100:.2f}%')
    ax4.axvline(np.percentile(returns * 100, 5), color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {np.percentile(returns * 100, 5):.2f}%')
    ax4.set_title('Daily Return Distribution')
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Volatility
    ax5 = axes[2, 0]
    if len(returns) > 30:
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
        ax5.plot(rolling_vol.index, rolling_vol.values, linewidth=2, color='purple')
        ax5.set_title('Rolling 30-Day Volatility')
        ax5.set_ylabel('Annualized Volatility (%)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Position Turnover (if available)
    ax6 = axes[2, 1]
    if backtest_engine.turnover is not None and len(backtest_engine.turnover) > 0:
        turnover_data = backtest_engine.turnover['turnover'] * 100
        ax6.bar(turnover_data.index, turnover_data.values, alpha=0.7, color='orange')
        ax6.set_title('Portfolio Turnover')
        ax6.set_ylabel('Turnover (%)')
        ax6.grid(True, alpha=0.3)
    else:
        ax6.text(0.5, 0.5, 'No turnover data available', transform=ax6.transAxes, 
                ha='center', va='center', fontsize=12)
        ax6.set_title('Portfolio Turnover')
    
    plt.tight_layout()
    plt.show()

def main_backtesting_demo():
    """Comprehensive demonstration of the backtesting framework"""
    
    print("=" * 80)
    print("ALPHA PROJECT - SYSTEMATIC BACKTESTING FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    
    # Load market data
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA', 'JPM', 'JNJ']
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    print(f"\nLoading data for {len(universe)} stocks from {start_date} to {end_date}...")
    
    # Download price data
    price_data = yf.download(universe, start=start_date, end=end_date)['Adj Close']
    returns_data = price_data.pct_change().dropna()
    
    # Download market benchmark (S&P 500)
    benchmark_data = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    print("Data loaded successfully!")
    
    # Strategy 1: Momentum Strategy
    print("\n" + "=" * 60)
    print("STRATEGY 1: MOMENTUM STRATEGY")
    print("=" * 60)
    
    # Create momentum signals
    momentum_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns, data=0.0)
    
    lookback_window = 63  # 3 months
    for i in range(lookback_window, len(returns_data)):
        date = returns_data.index[i]
        
        # Calculate momentum scores
        momentum_period = returns_data.iloc[i-lookback_window:i-21]  # Exclude recent month
        momentum_scores = (1 + momentum_period).prod() - 1
        
        # Rank stocks
        valid_scores = momentum_scores.dropna()
        if len(valid_scores) >= 5:
            # Long top 5, short bottom 5
            top_stocks = valid_scores.nlargest(5).index
            bottom_stocks = valid_scores.nsmallest(5).index
            
            # Equal weight positions
            momentum_signals.loc[date, top_stocks] = 0.1  # 10% each in top 5
            momentum_signals.loc[date, bottom_stocks] = -0.1  # -10% each in bottom 5
    
    # Backtest momentum strategy
    momentum_engine = BacktestEngine(
        initial_capital=1000000,
        transaction_cost=0.0015,  # 15 bps
        max_position_size=0.15,   # 15% max position
        rebalance_frequency='monthly'
    )
    
    momentum_returns = momentum_engine.backtest_strategy(
        returns_data, 
        momentum_signals, 
        "Momentum Strategy"
    )
    
    momentum_metrics = momentum_engine.calculate_performance_metrics(benchmark_returns)
    momentum_engine.generate_performance_report("Momentum Strategy", "S&P 500")
    
    # Strategy 2: Mean Reversion Strategy
    print("\n" + "=" * 60)
    print("STRATEGY 2: MEAN REVERSION STRATEGY")
    print("=" * 60)
    
    # Create mean reversion signals
    mean_reversion_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns, data=0.0)
    
    lookback_window = 21  # 1 month
    for i in range(lookback_window, len(returns_data)):
        date = returns_data.index[i]
        
        # Calculate z-scores
        recent_returns = returns_data.iloc[i-lookback_window:i]
        mean_returns = recent_returns.mean()
        std_returns = recent_returns.std()
        
        current_returns = returns_data.iloc[i-1]  # Previous day's return
        z_scores = (current_returns - mean_returns) / (std_returns + 1e-8)
        
        # Create signals (contrarian)
        valid_scores = z_scores.dropna()
        if len(valid_scores) >= 5:
            # Long most oversold, short most overbought
            oversold = valid_scores.nsmallest(5).index  # Most negative z-scores
            overbought = valid_scores.nlargest(5).index  # Most positive z-scores
            
            mean_reversion_signals.loc[date, oversold] = 0.1
            mean_reversion_signals.loc[date, overbought] = -0.1
    
    # Backtest mean reversion strategy
    mean_reversion_engine = BacktestEngine(
        initial_capital=1000000,
        transaction_cost=0.0020,  # 20 bps (higher due to more frequent trading)
        max_position_size=0.15,
        rebalance_frequency='weekly'
    )
    
    mean_reversion_returns = mean_reversion_engine.backtest_strategy(
        returns_data,
        mean_reversion_signals,
        "Mean Reversion Strategy"
    )
    
    mean_reversion_metrics = mean_reversion_engine.calculate_performance_metrics(benchmark_returns)
    mean_reversion_engine.generate_performance_report("Mean Reversion Strategy", "S&P 500")
    
    # Strategy 3: Equal-Weight Portfolio (Baseline)
    print("\n" + "=" * 60)
    print("STRATEGY 3: EQUAL-WEIGHT BASELINE")
    print("=" * 60)
    
    # Create equal-weight signals
    equal_weight_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
    equal_weight_signals[:] = 1.0 / len(returns_data.columns)  # Equal weight
    
    # Backtest equal-weight strategy
    equal_weight_engine = BacktestEngine(
        initial_capital=1000000,
        transaction_cost=0.0005,  # 5 bps (low turnover)
        max_position_size=0.2,
        rebalance_frequency='quarterly'
    )
    
    equal_weight_returns = equal_weight_engine.backtest_strategy(
        returns_data,
        equal_weight_signals,
        "Equal-Weight Portfolio"
    )
    
    equal_weight_metrics = equal_weight_engine.calculate_performance_metrics(benchmark_returns)
    equal_weight_engine.generate_performance_report("Equal-Weight Portfolio", "S&P 500")
    
    # Comparative Analysis
    print("\n" + "=" * 80)
    print("COMPARATIVE STRATEGY ANALYSIS")
    print("=" * 80)
    
    # Create comparison table
    strategies = {
        'Momentum': momentum_metrics,
        'Mean Reversion': mean_reversion_metrics,
        'Equal Weight': equal_weight_metrics
    }
    
    comparison_metrics = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio', 
                         'Max Drawdown', 'Information Ratio', 'Cost Drag']
    
    comparison_data = []
    for strategy_name, metrics in strategies.items():
        row = {'Strategy': strategy_name}
        for metric in comparison_metrics:
            if metric in metrics:
                if 'Return' in metric or 'Volatility' in metric or 'Drawdown' in metric or 'Cost Drag' in metric:
                    row[metric] = f"{metrics[metric]:.2%}"
                else:
                    row[metric] = f"{metrics[metric]:.3f}"
            else:
                row[metric] = "N/A"
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\nStrategy Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Create visualizations for all strategies
    create_strategy_comparison_chart(momentum_returns, mean_reversion_returns, 
                                   equal_weight_returns, benchmark_returns)
    
    # Individual strategy analysis
    print("\n" + "=" * 60)
    print("DETAILED MOMENTUM STRATEGY ANALYSIS")
    print("=" * 60)
    create_comprehensive_visualizations(momentum_engine, "Momentum Strategy")
    
    return {
        'momentum': {'engine': momentum_engine, 'returns': momentum_returns},
        'mean_reversion': {'engine': mean_reversion_engine, 'returns': mean_reversion_returns},
        'equal_weight': {'engine': equal_weight_engine, 'returns': equal_weight_returns},
        'benchmark': benchmark_returns
    }

def create_strategy_comparison_chart(momentum_returns, mean_reversion_returns, 
                                   equal_weight_returns, benchmark_returns):
    """Create comparison chart for all strategies"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strategy Comparison Analysis', fontsize=16, fontweight='bold')
    
    # Align all return series
    common_dates = momentum_returns.index.intersection(mean_reversion_returns.index)
    common_dates = common_dates.intersection(equal_weight_returns.index)
    common_dates = common_dates.intersection(benchmark_returns.index)
    
    mom_aligned = momentum_returns.reindex(common_dates).fillna(0)
    mr_aligned = mean_reversion_returns.reindex(common_dates).fillna(0)
    ew_aligned = equal_weight_returns.reindex(common_dates).fillna(0)
    bench_aligned = benchmark_returns.reindex(common_dates).fillna(0)
    
    # 1. Cumulative Returns Comparison
    ax1 = axes[0, 0]
    
    mom_cum = (1 + mom_aligned).cumprod()
    mr_cum = (1 + mr_aligned).cumprod()
    ew_cum = (1 + ew_aligned).cumprod()
    bench_cum = (1 + bench_aligned).cumprod()
    
    ax1.plot(mom_cum.index, mom_cum.values, label='Momentum', linewidth=2)
    ax1.plot(mr_cum.index, mr_cum.values, label='Mean Reversion', linewidth=2)
    ax1.plot(ew_cum.index, ew_cum.values, label='Equal Weight', linewidth=2)
    ax1.plot(bench_cum.index, bench_cum.values, label='S&P 500', linewidth=2, linestyle='--')
    
    ax1.set_title('Cumulative Returns Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk-Return Scatter
    ax2 = axes[0, 1]
    
    strategies = {
        'Momentum': mom_aligned,
        'Mean Reversion': mr_aligned,
        'Equal Weight': ew_aligned,
        'S&P 500': bench_aligned
    }
    
    for name, returns in strategies.items():
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        ax2.scatter(ann_vol, ann_ret, s=100, label=name, alpha=0.7)
    
    ax2.set_xlabel('Annualized Volatility')
    ax2.set_ylabel('Annualized Return')
    ax2.set_title('Risk-Return Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Rolling Sharpe Ratios
    ax3 = axes[1, 0]
    
    window = 252  # 1-year rolling
    for name, returns in strategies.items():
        if len(returns) > window:
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            ax3.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
    
    ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax3.set_title('Rolling 1-Year Sharpe Ratios')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Return Distributions
    ax4 = axes[1, 1]
    
    for name, returns in strategies.items():
        ax4.hist(returns * 100, bins=30, alpha=0.6, label=name, density=True)
    
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('Return Distributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the comprehensive backtesting demonstration
    results = main_backtesting_demo()
    
    print("\n" + "=" * 80)
    print("BACKTESTING FRAMEWORK COMPLETED")
    print("=" * 80)
    
    print("\nKey Features Demonstrated:")
    print("• Transaction cost modeling")
    print("• Position sizing constraints")
    print("• Multiple rebalancing frequencies")
    print("• Comprehensive risk metrics")
    print("• Benchmark comparison")
    print("• Strategy performance attribution")
    
    print("\nFramework Benefits:")
    print("• Academic rigor with practical implementation")
    print("• Realistic transaction costs and constraints")
    print("• Professional-grade performance metrics")
    print("• Flexible strategy testing capability")
    print("• Publication-ready visualizations")
    
    print(f"\nBacktesting framework ready for deployment!")
    print("Ready to test any academic research implementation!")
