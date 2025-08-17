# Alpha Project: Academic Research Implementation Engine
# Notebook 5: Alternative Risk Premia Strategies

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class AlternativeRiskPremia:
    """
    Implementation of Alternative Risk Premia Strategies
    Based on: Ilmanen, A. (2011) - Expected Returns
             Ang, A. (2014) - Asset Management
             Kolanovic, M. & Krishnamachari, R. (2017) - Big Data and AI Strategies
    """
    
    def __init__(self):
        self.market_data = None
        self.risk_premia_factors = None
        self.portfolio_weights = None
        
    def load_multi_asset_data(self, start_date='2015-01-01', end_date='2024-01-01'):
        """Load multi-asset universe for risk premia strategies"""
        
        print("Loading multi-asset data...")
        
        # Define asset universe
        assets = {
            # Equity Indices
            'SPY': 'S&P 500',
            'EFA': 'EAFE Developed',
            'EEM': 'Emerging Markets',
            'IWM': 'Small Cap',
            'VGK': 'Europe',
            'VPL': 'Asia Pacific',
            
            # Fixed Income
            'TLT': '20Y Treasury',
            'IEF': '7-10Y Treasury', 
            'SHY': '1-3Y Treasury',
            'LQD': 'IG Corporate',
            'HYG': 'High Yield',
            'EMB': 'EM Bonds',
            'TIP': 'TIPS',
            
            # Commodities
            'GLD': 'Gold',
            'SLV': 'Silver',
            'DJP': 'Commodities',
            'USO': 'Oil',
            'UNG': 'Natural Gas',
            
            # Currency
            'UUP': 'USD Index',
            'FXE': 'Euro',
            'FXY': 'Yen',
            
            # Volatility
            'VXX': 'VIX',
            
            # REITs
            'VNQ': 'US REITs',
            'VNQI': 'Intl REITs'
        }
        
        # Download price data
        tickers = list(assets.keys())
        price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate returns
        returns_data = price_data.pct_change().dropna()
        
        # Store data with asset names
        self.market_data = {
            'prices': price_data,
            'returns': returns_data,
            'asset_names': assets
        }
        
        print(f"Loaded data for {len(tickers)} assets from {start_date} to {end_date}")
        return self.market_data
    
    def carry_strategy(self, returns_data, lookback_window=252):
        """
        Implement Carry Strategy
        Based on: Koijen, R. S., Moskowitz, T. J., Pedersen, L. H., & Vrugt, E. B. (2018)
        """
        
        carry_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        # For simplicity, we'll use realized returns as a proxy for carry
        # In practice, you'd use forward rates, dividend yields, etc.
        
        for i in range(lookback_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate carry proxies
            # For bonds: Use yield curve slope (approximated by return differences)
            # For currencies: Use interest rate differentials (approximated)
            # For commodities: Use contango/backwardation (approximated)
            
            period_returns = returns_data.iloc[i-lookback_window:i]
            
            # Carry proxy: negative correlation with past returns (mean reversion)
            carry_scores = -period_returns.mean()
            
            # Adjust for bonds (longer duration = higher carry sensitivity)
            bond_tickers = ['TLT', 'IEF', 'LQD', 'HYG', 'EMB', 'TIP']
            for ticker in bond_tickers:
                if ticker in carry_scores.index:
                    # Longer duration bonds get higher weight in carry
                    if ticker == 'TLT':
                        carry_scores[ticker] *= 2.0  # 20Y bonds
                    elif ticker in ['IEF', 'LQD']:
                        carry_scores[ticker] *= 1.5  # Medium duration
            
            # Currency carry (simplified)
            currency_tickers = ['UUP', 'FXE', 'FXY']
            for ticker in currency_tickers:
                if ticker in carry_scores.index:
                    # Use volatility-adjusted returns as carry proxy
                    vol = period_returns[ticker].std()
                    carry_scores[ticker] = carry_scores[ticker] / (vol + 1e-8)
            
            # Rank and create signals
            valid_scores = carry_scores.dropna()
            if len(valid_scores) >= 10:
                # Long top 1/3, short bottom 1/3
                n_long = len(valid_scores) // 3
                n_short = len(valid_scores) // 3
                
                top_carry = valid_scores.nlargest(n_long).index
                bottom_carry = valid_scores.nsmallest(n_short).index
                
                carry_signals.loc[date, top_carry] = 1.0 / n_long
                carry_signals.loc[date, bottom_carry] = -1.0 / n_short
        
        return carry_signals.fillna(0)
    
    def momentum_strategy(self, returns_data, short_window=21, long_window=252):
        """
        Implement Cross-Asset Momentum Strategy
        Based on: Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012)
        """
        
        momentum_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for i in range(long_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate momentum scores (skip recent period to avoid microstructure noise)
            momentum_period = returns_data.iloc[i-long_window:i-short_window]
            momentum_scores = (1 + momentum_period).prod() - 1
            
            # Risk-adjust momentum scores
            vol_period = returns_data.iloc[i-63:i]  # 3-month volatility
            volatilities = vol_period.std() * np.sqrt(252)
            
            # Volatility-adjusted momentum
            risk_adj_momentum = momentum_scores / (volatilities + 1e-8)
            
            # Create signals based on risk-adjusted momentum
            valid_scores = risk_adj_momentum.dropna()
            if len(valid_scores) >= 10:
                # Use z-score approach
                z_scores = (valid_scores - valid_scores.mean()) / (valid_scores.std() + 1e-8)
                
                # Long assets with z-score > 0.5, short with z-score < -0.5
                long_signals = z_scores > 0.5
                short_signals = z_scores < -0.5
                
                if long_signals.sum() > 0:
                    momentum_signals.loc[date, long_signals] = 1.0 / long_signals.sum()
                if short_signals.sum() > 0:
                    momentum_signals.loc[date, short_signals] = -1.0 / short_signals.sum()
        
        return momentum_signals.fillna(0)
    
    def value_strategy(self, returns_data, lookback_window=756):  # 3 years
        """
        Implement Value Strategy
        Based on: Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013)
        """
        
        value_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for i in range(lookback_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate long-term returns (value reversal)
            long_term_period = returns_data.iloc[i-lookback_window:i-252]  # Skip recent year
            long_term_returns = (1 + long_term_period).prod() - 1
            
            # Value score: negative of long-term returns (contrarian)
            value_scores = -long_term_returns
            
            # Adjust for asset classes
            # Equities: Use traditional value approach
            equity_tickers = ['SPY', 'EFA', 'EEM', 'IWM', 'VGK', 'VPL']
            for ticker in equity_tickers:
                if ticker in value_scores.index:
                    value_scores[ticker] *= 1.2  # Emphasize equity value
            
            # Fixed Income: Use real rates (approximated)
            bond_tickers = ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB']
            for ticker in bond_tickers:
                if ticker in value_scores.index:
                    # For bonds, consider duration and credit risk
                    if ticker in ['TLT', 'IEF']:  # Government bonds
                        value_scores[ticker] *= 0.8  # Lower value sensitivity
                    elif ticker in ['LQD', 'HYG']:  # Corporate bonds
                        value_scores[ticker] *= 1.1  # Higher value sensitivity
            
            # Create signals
            valid_scores = value_scores.dropna()
            if len(valid_scores) >= 8:
                # Long top quartile, short bottom quartile
                n_positions = max(2, len(valid_scores) // 4)
                
                top_value = valid_scores.nlargest(n_positions).index
                bottom_value = valid_scores.nsmallest(n_positions).index
                
                value_signals.loc[date, top_value] = 1.0 / n_positions
                value_signals.loc[date, bottom_value] = -1.0 / n_positions
        
        return value_signals.fillna(0)
    
    def low_volatility_strategy(self, returns_data, lookback_window=252):
        """
        Implement Low Volatility Strategy
        Based on: Baker, M., Bradley, B., & Wurgler, J. (2011)
        """
        
        low_vol_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for i in range(lookback_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate realized volatilities
            vol_period = returns_data.iloc[i-lookback_window:i]
            volatilities = vol_period.std() * np.sqrt(252)
            
            # Calculate risk-adjusted returns
            mean_returns = vol_period.mean() * 252
            sharpe_ratios = mean_returns / (volatilities + 1e-8)
            
            # Low volatility score (inverse volatility weighted by Sharpe)
            low_vol_scores = (1 / (volatilities + 1e-8)) * np.maximum(sharpe_ratios, 0)
            
            # Create signals (long only strategy)
            valid_scores = low_vol_scores.dropna()
            if len(valid_scores) >= 5:
                # Weight by low volatility scores (normalized)
                weights = valid_scores / valid_scores.sum()
                low_vol_signals.loc[date] = weights.reindex(returns_data.columns, fill_value=0)
        
        return low_vol_signals.fillna(0)
    
    def quality_strategy(self, returns_data, lookback_window=756):  # 3 years
        """
        Implement Quality Strategy
        Based on: Asness, C. S., Frazzini, A., & Pedersen, L. H. (2019)
        """
        
        quality_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        for i in range(lookback_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate quality metrics
            period_returns = returns_data.iloc[i-lookback_window:i]
            
            # Quality metrics:
            # 1. Consistency (negative of volatility)
            consistency = -period_returns.std()
            
            # 2. Trend strength (positive momentum with low volatility)
            trend_strength = period_returns.mean() / (period_returns.std() + 1e-8)
            
            # 3. Maximum drawdown (inverse)
            cumulative = (1 + period_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdowns = (cumulative - rolling_max) / rolling_max
            max_dd = drawdowns.min()
            dd_quality = -max_dd  # Lower drawdown = higher quality
            
            # Combine quality metrics
            quality_scores = (consistency + trend_strength + dd_quality) / 3
            
            # Create signals
            valid_scores = quality_scores.dropna()
            if len(valid_scores) >= 5:
                # Long top half of quality assets
                n_positions = len(valid_scores) // 2
                top_quality = valid_scores.nlargest(n_positions).index
                
                quality_signals.loc[date, top_quality] = 1.0 / n_positions
        
        return quality_signals.fillna(0)
    
    def defensive_strategy(self, returns_data, market_proxy='SPY', lookback_window=252):
        """
        Implement Defensive Strategy
        Based on: Frazzini, A., & Pedersen, L. H. (2014) - Betting Against Beta
        """
        
        defensive_signals = pd.DataFrame(index=returns_data.index, columns=returns_data.columns)
        
        if market_proxy not in returns_data.columns:
            print(f"Market proxy {market_proxy} not found. Using equal-weight market.")
            market_returns = returns_data.mean(axis=1)
        else:
            market_returns = returns_data[market_proxy]
        
        for i in range(lookback_window, len(returns_data)):
            date = returns_data.index[i]
            
            # Calculate betas
            period_returns = returns_data.iloc[i-lookback_window:i]
            period_market = market_returns.iloc[i-lookback_window:i]
            
            betas = {}
            for asset in period_returns.columns:
                if asset != market_proxy:
                    asset_returns = period_returns[asset].dropna()
                    aligned_market = period_market.reindex(asset_returns.index).dropna()
                    
                    if len(asset_returns) > 50 and len(aligned_market) > 50:
                        # Calculate beta
                        covariance = np.cov(asset_returns, aligned_market)[0, 1]
                        market_variance = np.var(aligned_market)
                        beta = covariance / market_variance if market_variance > 0 else 1.0
                        betas[asset] = beta
            
            if len(betas) >= 5:
                beta_series = pd.Series(betas)
                
                # Defensive score: inverse of beta
                defensive_scores = 1 / (np.abs(beta_series) + 0.1)  # Add small constant
                
                # Create signals (long low-beta assets)
                n_positions = min(len(defensive_scores) // 2, 10)
                top_defensive = defensive_scores.nlargest(n_positions).index
                
                defensive_signals.loc[date, top_defensive] = 1.0 / n_positions
        
        return defensive_signals.fillna(0)
    
    def combine_risk_premia(self, signals_dict, combination_method='equal_weight'):
        """Combine multiple risk premia strategies"""
        
        # Align all signals
        common_dates = None
        for strategy_name, signals in signals_dict.items():
            if common_dates is None:
                common_dates = signals.index
            else:
                common_dates = common_dates.intersection(signals.index)
        
        aligned_signals = {}
        for strategy_name, signals in signals_dict.items():
            aligned_signals[strategy_name] = signals.reindex(common_dates, fill_value=0)
        
        if combination_method == 'equal_weight':
            # Equal weight combination
            combined_signals = pd.DataFrame(0.0, index=common_dates, 
                                          columns=list(signals_dict.values())[0].columns)
            
            for signals in aligned_signals.values():
                combined_signals += signals / len(aligned_signals)
                
        elif combination_method == 'vol_target':
            # Volatility-targeted combination
            combined_signals = pd.DataFrame(0.0, index=common_dates,
                                          columns=list(signals_dict.values())[0].columns)
            
            # Calculate strategy volatilities (simplified)
            weights = {}
            for strategy_name, signals in aligned_signals.items():
                # Approximate volatility from signal changes
                signal_vol = signals.diff().std().mean()
                weights[strategy_name] = 1.0 / (signal_vol + 1e-8)
            
            # Normalize weights
            total_weight = sum(weights.values())
            for strategy_name in weights:
                weights[strategy_name] /= total_weight
            
            # Combine with volatility weights
            for strategy_name, signals in aligned_signals.items():
                combined_signals += signals * weights[strategy_name]
        
        return combined_signals

def analyze_risk_premia_performance(signals_dict, returns_data, transaction_cost=0.001):
    """Analyze performance of risk premia strategies"""
    
    performance_results = {}
    
    # Align data
    common_dates = returns_data.index
    for strategy_name, signals in signals_dict.items():
        common_dates = common_dates.intersection(signals.index)
    
    aligned_returns = returns_data.reindex(common_dates)
    
    for strategy_name, signals in signals_dict.items():
        aligned_signals = signals.reindex(common_dates, fill_value=0)
        
        # Calculate strategy returns
        strategy_returns = []
        portfolio_value = 1.0
        previous_weights = pd.Series(0.0, index=aligned_returns.columns)
        
        for i, date in enumerate(common_dates[1:], 1):
            # Get current weights and returns
            current_weights = aligned_signals.iloc[i-1]  # Previous day's signals
            current_returns = aligned_returns.iloc[i]
            
            # Calculate turnover and transaction costs
            turnover = np.abs(current_weights - previous_weights).sum()
            transaction_cost_amount = turnover * transaction_cost
            
            # Calculate gross return
            gross_return = (current_weights * current_returns).sum()
            
            # Calculate net return (after transaction costs)
            net_return = gross_return - transaction_cost_amount
            
            strategy_returns.append(net_return)
            portfolio_value *= (1 + net_return)
            previous_weights = current_weights
        
        # Calculate performance metrics
        strategy_returns = pd.Series(strategy_returns, index=common_dates[1:])
        
        total_return = (1 + strategy_returns).prod() - 1
        annualized_return = (1 + strategy_returns.mean()) ** 252 - 1
        annualized_volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        performance_results[strategy_name] = {
            'returns': strategy_returns,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_value
        }
    
    return performance_results

def main_risk_premia_analysis():
    """Main analysis of alternative risk premia strategies"""
    
    print("=" * 80)
    print("ALTERNATIVE RISK PREMIA STRATEGIES ANALYSIS")
    print("=" * 80)
    
    # Initialize the risk premia engine
    arp_engine = AlternativeRiskPremia()
    
    # Load multi-asset data
    market_data = arp_engine.load_multi_asset_data(start_date='2016-01-01', end_date='2024-01-01')
    returns_data = market_data['returns']
    
    print(f"\nImplementing risk premia strategies...")
    
    # Implement individual strategies
    strategies = {}
    
    print("1. Carry Strategy...")
    strategies['Carry'] = arp_engine.carry_strategy(returns_data)
    
    print("2. Momentum Strategy...")
    strategies['Momentum'] = arp_engine.momentum_strategy(returns_data)
    
    print("3. Value Strategy...")
    strategies['Value'] = arp_engine.value_strategy(returns_data)
    
    print("4. Low Volatility Strategy...")
    strategies['Low_Vol'] = arp_engine.low_volatility_strategy(returns_data)
    
    print("5. Quality Strategy...")
    strategies['Quality'] = arp_engine.quality_strategy(returns_data)
    
    print("6. Defensive Strategy...")
    strategies['Defensive'] = arp_engine.defensive_strategy(returns_data)
    
    # Create combined strategies
    print("7. Creating Combined Strategies...")
    
    # Equal-weight combination
    strategies['Equal_Weight_Combo'] = arp_engine.combine_risk_premia(
        strategies, combination_method='equal_weight'
    )
    
    # Volatility-targeted combination
    strategies['Vol_Target_Combo'] = arp_engine.combine_risk_premia(
        strategies, combination_method='vol_target'
    )
    
    # Analyze performance
    print("\nAnalyzing strategy performance...")
    performance_results = analyze_risk_premia_performance(strategies, returns_data)
    
    # Display results
    print("\n" + "=" * 60)
    print("RISK PREMIA STRATEGY PERFORMANCE")
    print("=" * 60)
    
    # Create performance summary
    summary_data = []
    for strategy_name, results in performance_results.items():
        summary_data.append({
            'Strategy': strategy_name,
            'Total Return': f"{results['total_return']:.2%}",
            'Ann. Return': f"{results['annualized_return']:.2%}",
            'Ann. Vol': f"{results['annualized_volatility']:.2%}",
            'Sharpe': f"{results['sharpe_ratio']:.3f}",
            'Max DD': f"{results['max_drawdown']:.2%}",
            'Final Value': f"${results['final_value']:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Create visualizations
    create_risk_premia_visualizations(performance_results, market_data)
    
    return performance_results, strategies, market_data

def create_risk_premia_visualizations(performance_results, market_data):
    """Create comprehensive visualizations for risk premia analysis"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Alternative Risk Premia Strategies Analysis', fontsize=16, fontweight='bold')
    
    # 1. Cumulative Performance
    ax1 = axes[0, 0]
    
    for strategy_name, results in performance_results.items():
        returns = results['returns']
        cumulative = (1 + returns).cumprod()
        ax1.plot(cumulative.index, cumulative.values, label=strategy_name, linewidth=2)
    
    ax1.set_title('Cumulative Performance Comparison')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Risk-Return Scatter
    ax2 = axes[0, 1]
    
    for strategy_name, results in performance_results.items():
        ax2.scatter(results['annualized_volatility'], results['annualized_return'], 
                   s=100, label=strategy_name, alpha=0.7)
    
    ax2.set_xlabel('Annualized Volatility')
    ax2.set_ylabel('Annualized Return')
    ax2.set_title('Risk-Return Profile')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Sharpe Ratios
    ax3 = axes[1, 0]
    
    sharpe_ratios = {name: results['sharpe_ratio'] for name, results in performance_results.items()}
    strategies = list(sharpe_ratios.keys())
    ratios = list(sharpe_ratios.values())
    
    bars = ax3.bar(strategies, ratios, alpha=0.7)
    ax3.set_title('Sharpe Ratios Comparison')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Color bars based on performance
    for i, bar in enumerate(bars):
        if ratios[i] > 0.5:
            bar.set_color('green')
        elif ratios[i] > 0:
            bar.set_color('yellow')
        else:
            bar.set_color('red')
    
    # 4. Maximum Drawdowns
    ax4 = axes[1, 1]
    
    max_drawdowns = {name: results['max_drawdown'] for name, results in performance_results.items()}
    strategies = list(max_drawdowns.keys())
    drawdowns = [abs(dd) for dd in max_drawdowns.values()]
    
    bars = ax4.bar(strategies, drawdowns, alpha=0.7, color='red')
    ax4.set_title('Maximum Drawdowns')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Correlations (sample strategies)
    ax5 = axes[2, 0]
    
    sample_strategies = ['Momentum', 'Value', 'Low_Vol', 'Quality']
    sample_strategies = [s for s in sample_strategies if s in performance_results]
    
    if len(sample_strategies) >= 2:
        returns_df = pd.DataFrame({
            name: performance_results[name]['returns'] 
            for name in sample_strategies
        })
        
        # Calculate rolling correlation matrix
        window = 126  # 6 months
        if len(returns_df) > window:
            rolling_corr = returns_df[sample_strategies[0]].rolling(window).corr(
                returns_df[sample_strategies[1]]
            )
            ax5.plot(rolling_corr.index, rolling_corr.values, linewidth=2)
            ax5.set_title(f'Rolling Correlation: {sample_strategies[0]} vs {sample_strategies[1]}')
            ax5.set_ylabel('Correlation')
            ax5.grid(True, alpha=0.3)
            ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 6. Asset Class Exposure (for combined strategy)
    ax6 = axes[2, 1]
    
    if 'Equal_Weight_Combo' in performance_results:
        # Show asset class breakdown for the combined strategy
        # This is a simplified version - in practice you'd track actual exposures
        
        asset_classes = {
            'Equity': ['SPY', 'EFA', 'EEM', 'IWM', 'VGK', 'VPL'],
            'Fixed Income': ['TLT', 'IEF', 'SHY', 'LQD', 'HYG', 'EMB', 'TIP'],
            'Commodities': ['GLD', 'SLV', 'DJP', 'USO', 'UNG'],
            'Other': ['UUP', 'FXE', 'FXY', 'VXX', 'VNQ', 'VNQI']
        }
        
        # Create pie chart of asset class exposure
        exposure_counts = {}
        total_assets = len(market_data['returns'].columns)
        
        for asset_class, tickers in asset_classes.items():
            count = len([t for t in tickers if t in market_data['returns'].columns])
            exposure_counts[asset_class] = count
        
        ax6.pie(exposure_counts.values(), labels=exposure_counts.keys(), autopct='%1.1f%%')
        ax6.set_title('Asset Class Universe Breakdown')
    
    plt.tight_layout()
    plt.show()
    
    # Strategy-specific insights
    print("\n" + "=" * 60)
    print("RISK PREMIA STRATEGY INSIGHTS")
    print("=" * 60)
    
    best_sharpe = max(performance_results.items(), key=lambda x: x[1]['sharpe_ratio'])
    lowest_vol = min(performance_results.items(), key=lambda x: x[1]['annualized_volatility'])
    best_return = max(performance_results.items(), key=lambda x: x[1]['annualized_return'])
    
    print(f"Best Sharpe Ratio: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
    print(f"Lowest Volatility: {lowest_vol[0]} ({lowest_vol[1]['annualized_volatility']:.2%})")
    print(f"Highest Return: {best_return[0]} ({best_return[1]['annualized_return']:.2%})")
    
    # Correlation analysis
    if len(performance_results) >= 3:
        print(f"\nStrategy Correlation Analysis:")
        
        strategy_returns = pd.DataFrame({
            name: results['returns'] 
            for name, results in performance_results.items()
        })
        
        correlation_matrix = strategy_returns.corr()
        
        print("Average inter-strategy correlation:")
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        avg_correlation = upper_triangle.stack().mean()
        print(f"  {avg_correlation:.3f}")
        
        print("\nMost diversifying strategies:")
        avg_correlations = correlation_matrix.mean().sort_values()
        for strategy in avg_correlations.head(3).index:
            print(f"  {strategy}: {avg_correlations[strategy]:.3f}")

if __name__ == "__main__":
    # Run the risk premia analysis
    performance_results, strategies, market_data = main_risk_premia_analysis()
    
    print("\n" + "=" * 80)
    print("ALTERNATIVE RISK PREMIA ANALYSIS COMPLETED")
    print("=" * 80)
    
    print("\nKey Achievements:")
    print("• Implemented 6 core risk premia strategies")
    print("• Created multi-asset universe spanning equity, fixed income, commodities")
    print("• Developed strategy combination frameworks")
    print("• Analyzed performance with transaction costs")
    print("• Generated comprehensive risk-return analysis")
    
    print("\nImplementation Notes:")
    print("• Strategies are simplified for demonstration")
    print("• In practice, use more sophisticated factor construction")
    print("• Consider regime-dependent strategy weights")
    print("• Implement more granular transaction cost models")
    print("• Add capacity constraints for institutional implementation")
    
    print(f"\nRisk premia framework tested on {len(market_data['returns'].columns)} assets")
    print("Ready for institutional deployment!")
