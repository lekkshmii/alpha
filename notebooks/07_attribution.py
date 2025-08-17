# Alpha Project: Academic Research Implementation Engine
# Notebook 7: Performance Attribution and Risk Decomposition

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class PerformanceAttribution:
    """
    Comprehensive Performance Attribution and Risk Decomposition Framework
    Based on: Grinold, R. C., & Kahn, R. N. (1999) - Active Portfolio Management
             Litterman, R. (2003) - Modern Investment Management
             Meucci, A. (2005) - Risk and Asset Allocation
    """
    
    def __init__(self):
        self.portfolio_returns = None
        self.benchmark_returns = None
        self.factor_returns = None
        self.factor_loadings = None
        self.attribution_results = None
        
    def load_factor_data(self, start_date='2020-01-01', end_date='2024-01-01'):
        """Load factor data for attribution analysis"""
        
        print("Loading factor data...")
        
        # Market factors
        market_data = {
            '^GSPC': 'Market',  # S&P 500
            '^IRX': 'RiskFree',  # 3-Month Treasury
        }
        
        # Style factors (using ETFs as proxies)
        style_factors = {
            'IWD': 'Value',      # Value
            'IWF': 'Growth',     # Growth
            'IWM': 'SmallCap',   # Small Cap
            'IWB': 'LargeCap',   # Large Cap
            'MTUM': 'Momentum',  # Momentum
            'QUAL': 'Quality',   # Quality
            'USMV': 'LowVol',    # Low Volatility
        }
        
        # Sector factors
        sector_factors = {
            'XLK': 'Technology',
            'XLF': 'Financials', 
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrials',
            'XLP': 'ConsumerStaples',
            'XLY': 'ConsumerDiscretionary',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'RealEstate',
            'XLC': 'Communication'
        }
        
        # Combine all factors
        all_factors = {**market_data, **style_factors, **sector_factors}
        
        # Download factor data
        factor_prices = yf.download(list(all_factors.keys()), start=start_date, end=end_date)['Adj Close']
        factor_returns = factor_prices.pct_change().dropna()
        
        # Rename columns
        factor_returns = factor_returns.rename(columns=all_factors)
        
        # Convert risk-free rate
        if 'RiskFree' in factor_returns.columns:
            factor_returns['RiskFree'] = factor_returns['RiskFree'] / 100 / 252  # Convert to daily decimal
        
        # Calculate excess returns
        if 'Market' in factor_returns.columns and 'RiskFree' in factor_returns.columns:
            factor_returns['Market_Excess'] = factor_returns['Market'] - factor_returns['RiskFree']
        
        self.factor_returns = factor_returns
        
        print(f"Loaded {len(factor_returns.columns)} factors from {start_date} to {end_date}")
        return factor_returns
    
    def calculate_factor_loadings(self, portfolio_returns, method='regression', 
                                 rolling_window=252):
        """Calculate factor loadings/exposures"""
        
        print("Calculating factor loadings...")
        
        if self.factor_returns is None:
            raise ValueError("Factor returns not loaded. Call load_factor_data() first.")
        
        # Align data
        common_dates = portfolio_returns.index.intersection(self.factor_returns.index)
        port_returns = portfolio_returns.reindex(common_dates).dropna()
        factors = self.factor_returns.reindex(common_dates)
        
        # Select relevant factors for regression
        factor_cols = [col for col in factors.columns 
                      if col not in ['RiskFree', 'Market'] and not factors[col].isna().all()]
        
        if 'Market_Excess' in factors.columns:
            factor_cols = ['Market_Excess'] + factor_cols
        
        factor_subset = factors[factor_cols].dropna()
        
        # Align portfolio returns with factor data
        common_dates = port_returns.index.intersection(factor_subset.index)
        port_returns = port_returns.reindex(common_dates)
        factor_subset = factor_subset.reindex(common_dates)
        
        if method == 'regression':
            # Static factor loadings using full sample regression
            
            # Calculate excess returns
            if 'RiskFree' in factors.columns:
                excess_returns = port_returns - factors['RiskFree'].reindex(port_returns.index).fillna(0)
            else:
                excess_returns = port_returns
            
            # Run regression
            X = factor_subset.values
            y = excess_returns.values
            
            # Add constant for alpha
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            # OLS regression
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                
                # Create loadings dataframe
                loadings = pd.Series(beta[1:], index=factor_cols, name='Loading')
                alpha = beta[0]
                
                # Calculate R-squared
                y_pred = X_with_const @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                self.factor_loadings = {
                    'loadings': loadings,
                    'alpha': alpha,
                    'r_squared': r_squared
                }
                
            except np.linalg.LinAlgError:
                print("Regression failed - using PCA fallback")
                self.factor_loadings = self._pca_factor_loadings(port_returns, factor_subset)
        
        elif method == 'rolling':
            # Rolling factor loadings
            
            loadings_over_time = pd.DataFrame(index=common_dates, columns=factor_cols)
            alphas_over_time = pd.Series(index=common_dates)
            
            for i in range(rolling_window, len(common_dates)):
                end_date = common_dates[i]
                start_idx = i - rolling_window
                
                # Rolling window data
                window_returns = port_returns.iloc[start_idx:i]
                window_factors = factor_subset.iloc[start_idx:i]
                
                if 'RiskFree' in factors.columns:
                    window_rf = factors['RiskFree'].iloc[start_idx:i]
                    window_excess = window_returns - window_rf
                else:
                    window_excess = window_returns
                
                # Regression for this window
                X = window_factors.values
                y = window_excess.values
                
                if len(X) > 0 and len(y) > 0:
                    X_with_const = np.column_stack([np.ones(len(X)), X])
                    
                    try:
                        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                        
                        loadings_over_time.loc[end_date] = beta[1:]
                        alphas_over_time.loc[end_date] = beta[0]
                        
                    except np.linalg.LinAlgError:
                        continue
            
            self.factor_loadings = {
                'loadings_timeseries': loadings_over_time.dropna(),
                'alphas_timeseries': alphas_over_time.dropna()
            }
        
        return self.factor_loadings
    
    def _pca_factor_loadings(self, returns, factors):
        """Fallback PCA-based factor loadings"""
        
        # Use PCA to extract main factors
        pca = PCA(n_components=min(5, len(factors.columns)))
        pca_factors = pca.fit_transform(factors.fillna(0))
        
        # Regress returns on PCA factors
        X = pca_factors
        y = returns.values
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        # Create synthetic loadings
        loadings = pd.Series(beta, index=[f'PC{i+1}' for i in range(len(beta))], name='Loading')
        
        return {
            'loadings': loadings,
            'alpha': 0.0,
            'r_squared': pca.explained_variance_ratio_.sum(),
            'method': 'PCA'
        }
    
    def brinson_attribution(self, portfolio_weights, benchmark_weights, 
                          portfolio_returns, benchmark_returns, sector_returns):
        """
        Brinson Performance Attribution
        Decomposes active return into allocation and selection effects
        """
        
        print("Performing Brinson attribution...")
        
        # Ensure all inputs are aligned
        common_sectors = portfolio_weights.index.intersection(benchmark_weights.index)
        common_sectors = common_sectors.intersection(sector_returns.index)
        
        # Align weights and returns
        pw = portfolio_weights.reindex(common_sectors, fill_value=0)
        bw = benchmark_weights.reindex(common_sectors, fill_value=0)
        sr = sector_returns.reindex(common_sectors, fill_value=0)
        
        # Portfolio and benchmark total returns
        portfolio_return = (pw * sr).sum()
        benchmark_return = (bw * sr).sum()
        
        # Active return
        active_return = portfolio_return - benchmark_return
        
        # Allocation effect: (wp - wb) * rb
        allocation_effect = ((pw - bw) * sr).sum()
        
        # Selection effect: wb * (rp - rb)
        # For simplicity, assume portfolio sector returns = sector returns
        # In practice, you'd have portfolio-specific sector returns
        selection_effect = active_return - allocation_effect
        
        # Interaction effect (usually small)
        interaction_effect = ((pw - bw) * (sr - sr)).sum()  # Simplified
        
        attribution_results = {
            'active_return': active_return,
            'allocation_effect': allocation_effect,
            'selection_effect': selection_effect,
            'interaction_effect': interaction_effect,
            'sector_breakdown': pd.DataFrame({
                'portfolio_weight': pw,
                'benchmark_weight': bw,
                'sector_return': sr,
                'weight_difference': pw - bw,
                'allocation_contribution': (pw - bw) * sr
            })
        }
        
        return attribution_results
    
    def factor_attribution(self, portfolio_returns, benchmark_returns=None):
        """
        Factor-based performance attribution
        Attributes returns to common risk factors
        """
        
        print("Performing factor attribution...")
        
        if self.factor_loadings is None:
            self.calculate_factor_loadings(portfolio_returns)
        
        if self.factor_returns is None:
            raise ValueError("Factor returns not available")
        
        # Align data
        common_dates = portfolio_returns.index.intersection(self.factor_returns.index)
        port_returns = portfolio_returns.reindex(common_dates)
        factor_rets = self.factor_returns.reindex(common_dates)
        
        # Use static loadings for attribution
        if 'loadings' in self.factor_loadings:
            loadings = self.factor_loadings['loadings']
            alpha = self.factor_loadings['alpha']
            
            # Calculate factor contributions
            factor_contributions = {}
            
            for factor in loadings.index:
                if factor in factor_rets.columns:
                    factor_contrib = loadings[factor] * factor_rets[factor]
                    factor_contributions[factor] = factor_contrib
            
            # Create attribution DataFrame
            attribution_df = pd.DataFrame(factor_contributions)
            attribution_df.index = common_dates
            
            # Calculate residual returns (specific return)
            total_factor_return = attribution_df.sum(axis=1)
            residual_returns = port_returns - total_factor_return - alpha
            
            # Summary statistics
            factor_summary = {}
            
            for factor in attribution_df.columns:
                contrib = attribution_df[factor]
                factor_summary[factor] = {
                    'average_contribution': contrib.mean(),
                    'volatility_contribution': contrib.std(),
                    'total_contribution': contrib.sum(),
                    'loading': loadings[factor]
                }
            
            # Alpha and residual summary
            factor_summary['Alpha'] = {
                'average_contribution': alpha,
                'volatility_contribution': 0,
                'total_contribution': alpha * len(common_dates),
                'loading': 1.0
            }
            
            factor_summary['Residual'] = {
                'average_contribution': residual_returns.mean(),
                'volatility_contribution': residual_returns.std(),
                'total_contribution': residual_returns.sum(),
                'loading': 1.0
            }
            
            self.attribution_results = {
                'factor_contributions': attribution_df,
                'residual_returns': residual_returns,
                'factor_summary': pd.DataFrame(factor_summary).T,
                'loadings': loadings,
                'alpha': alpha
            }
        
        return self.attribution_results
    
    def risk_decomposition(self, portfolio_returns, decomposition_method='factor'):
        """
        Risk decomposition analysis
        Decomposes portfolio risk into systematic and idiosyncratic components
        """
        
        print("Performing risk decomposition...")
        
        if decomposition_method == 'factor':
            # Factor-based risk decomposition
            
            if self.factor_loadings is None:
                self.calculate_factor_loadings(portfolio_returns)
            
            if 'loadings' in self.factor_loadings:
                loadings = self.factor_loadings['loadings']
                
                # Align factor returns
                common_dates = portfolio_returns.index.intersection(self.factor_returns.index)
                factor_rets = self.factor_returns.reindex(common_dates)
                
                # Calculate factor covariance matrix
                factor_cols = loadings.index
                factor_subset = factor_rets[factor_cols].dropna()
                factor_cov = factor_subset.cov() * 252  # Annualized
                
                # Portfolio factor risk
                factor_risk = np.sqrt(loadings.T @ factor_cov @ loadings)
                
                # Total portfolio risk
                total_risk = portfolio_returns.std() * np.sqrt(252)  # Annualized
                
                # Idiosyncratic risk
                idiosyncratic_risk = np.sqrt(max(0, total_risk**2 - factor_risk**2))
                
                # Risk contributions by factor
                risk_contributions = {}
                for i, factor in enumerate(factor_cols):
                    marginal_contrib = 2 * loadings[factor] * (factor_cov.iloc[i] @ loadings)
                    risk_contributions[factor] = marginal_contrib / (2 * factor_risk) * factor_risk
                
                risk_decomp = {
                    'total_risk': total_risk,
                    'factor_risk': factor_risk,
                    'idiosyncratic_risk': idiosyncratic_risk,
                    'factor_risk_percentage': factor_risk / total_risk * 100,
                    'idiosyncratic_risk_percentage': idiosyncratic_risk / total_risk * 100,
                    'risk_contributions': pd.Series(risk_contributions),
                    'factor_covariance': factor_cov
                }
                
        elif decomposition_method == 'principal_components':
            # PCA-based risk decomposition
            
            # Use all available factor returns for PCA
            factor_data = self.factor_returns.dropna()
            
            # Align with portfolio returns
            common_dates = portfolio_returns.index.intersection(factor_data.index)
            factor_aligned = factor_data.reindex(common_dates)
            
            # Perform PCA
            pca = PCA()
            pca_factors = pca.fit_transform(factor_aligned.fillna(0))
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Number of components explaining 90% of variance
            n_components_90 = np.argmax(cumulative_variance >= 0.9) + 1
            
            risk_decomp = {
                'explained_variance_ratio': explained_variance,
                'cumulative_variance': cumulative_variance,
                'n_components_90_percent': n_components_90,
                'total_components': len(explained_variance),
                'principal_components': pca_factors[:, :n_components_90]
            }
        
        return risk_decomp
    
    def style_analysis(self, portfolio_returns, style_indices=None):
        """
        Style analysis to determine portfolio's style exposures
        Based on Sharpe's Return-Based Style Analysis
        """
        
        print("Performing style analysis...")
        
        if style_indices is None:
            # Default style indices
            style_indices = {
                'Growth': 'IWF',
                'Value': 'IWD', 
                'Large Cap': 'IWB',
                'Small Cap': 'IWM',
                'International': 'EFA',
                'Emerging Markets': 'EEM',
                'Bonds': 'AGG',
                'REITs': 'VNQ'
            }
        
        # Download style index data
        style_data = yf.download(list(style_indices.values()), 
                               start=portfolio_returns.index[0], 
                               end=portfolio_returns.index[-1])['Adj Close']
        
        style_returns = style_data.pct_change().dropna()
        style_returns.columns = list(style_indices.keys())
        
        # Align data
        common_dates = portfolio_returns.index.intersection(style_returns.index)
        port_ret = portfolio_returns.reindex(common_dates)
        style_ret = style_returns.reindex(common_dates)
        
        # Constrained regression (weights sum to 1, all non-negative)
        def objective(weights):
            predicted_returns = (style_ret * weights).sum(axis=1)
            tracking_error = ((port_ret - predicted_returns) ** 2).sum()
            return tracking_error
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: w.sum() - 1},  # Weights sum to 1
        ]
        
        bounds = [(0, 1) for _ in range(len(style_indices))]  # Non-negative weights
        
        # Initial guess
        x0 = np.ones(len(style_indices)) / len(style_indices)
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            if result.success:
                style_weights = pd.Series(result.x, index=style_returns.columns)
                
                # Calculate R-squared
                predicted_returns = (style_ret * style_weights).sum(axis=1)
                ss_res = ((port_ret - predicted_returns) ** 2).sum()
                ss_tot = ((port_ret - port_ret.mean()) ** 2).sum()
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Tracking error
                tracking_error = np.sqrt(ss_res / len(common_dates)) * np.sqrt(252)  # Annualized
                
                style_analysis_results = {
                    'style_weights': style_weights,
                    'r_squared': r_squared,
                    'tracking_error': tracking_error,
                    'predicted_returns': predicted_returns,
                    'residual_returns': port_ret - predicted_returns
                }
                
            else:
                print("Style analysis optimization failed")
                style_analysis_results = None
                
        except Exception as e:
            print(f"Style analysis failed: {e}")
            style_analysis_results = None
        
        return style_analysis_results

def create_attribution_visualizations(attribution_engine, portfolio_returns):
    """Create comprehensive attribution visualizations"""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Performance Attribution & Risk Decomposition Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Factor Loadings
    ax1 = axes[0, 0]
    
    if attribution_engine.factor_loadings and 'loadings' in attribution_engine.factor_loadings:
        loadings = attribution_engine.factor_loadings['loadings']
        
        # Sort by absolute value
        loadings_sorted = loadings.reindex(loadings.abs().sort_values(ascending=False).index)
        
        colors = ['green' if x > 0 else 'red' for x in loadings_sorted.values]
        bars = ax1.barh(range(len(loadings_sorted)), loadings_sorted.values, color=colors, alpha=0.7)
        
        ax1.set_yticks(range(len(loadings_sorted)))
        ax1.set_yticklabels(loadings_sorted.index)
        ax1.set_xlabel('Factor Loading')
        ax1.set_title('Factor Loadings (Beta Exposures)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    
    # 2. Factor Contributions Over Time
    ax2 = axes[0, 1]
    
    if attribution_engine.attribution_results and 'factor_contributions' in attribution_engine.attribution_results:
        factor_contrib = attribution_engine.attribution_results['factor_contributions']
        
        # Plot cumulative contributions for top factors
        top_factors = factor_contrib.abs().mean().nlargest(5).index
        
        for factor in top_factors:
            cumulative_contrib = factor_contrib[factor].cumsum()
            ax2.plot(cumulative_contrib.index, cumulative_contrib.values, 
                    label=factor, linewidth=2)
        
        ax2.set_title('Cumulative Factor Contributions')
        ax2.set_ylabel('Cumulative Contribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Risk Decomposition
    ax3 = axes[1, 0]
    
    risk_decomp = attribution_engine.risk_decomposition(portfolio_returns)
    
    if 'factor_risk' in risk_decomp:
        risk_components = [
            risk_decomp['factor_risk'],
            risk_decomp['idiosyncratic_risk']
        ]
        labels = ['Factor Risk', 'Idiosyncratic Risk']
        colors = ['skyblue', 'lightcoral']
        
        ax3.pie(risk_components, labels=labels, colors=colors, autopct='%1.1f%%')
        ax3.set_title('Risk Decomposition')
    
    # 4. Factor Risk Contributions
    ax4 = axes[1, 1]
    
    if 'risk_contributions' in risk_decomp:
        risk_contrib = risk_decomp['risk_contributions']
        
        # Sort by contribution size
        risk_contrib_sorted = risk_contrib.reindex(risk_contrib.abs().sort_values(ascending=False).index)
        
        colors = ['green' if x > 0 else 'red' for x in risk_contrib_sorted.values]
        ax4.bar(range(len(risk_contrib_sorted)), risk_contrib_sorted.values, 
               color=colors, alpha=0.7)
        
        ax4.set_xticks(range(len(risk_contrib_sorted)))
        ax4.set_xticklabels(risk_contrib_sorted.index, rotation=45)
        ax4.set_ylabel('Risk Contribution')
        ax4.set_title('Factor Risk Contributions')
        ax4.grid(True, alpha=0.3)
    
    # 5. Rolling Alpha
    ax5 = axes[2, 0]
    
    if attribution_engine.factor_loadings and 'alphas_timeseries' in attribution_engine.factor_loadings:
        rolling_alpha = attribution_engine.factor_loadings['alphas_timeseries']
        
        # Convert to annualized percentage
        rolling_alpha_annual = rolling_alpha * 252 * 100
        
        ax5.plot(rolling_alpha_annual.index, rolling_alpha_annual.values, 
                linewidth=2, color='purple')
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('Rolling Alpha (1-Year Window)')
        ax5.set_ylabel('Alpha (%)')
        ax5.grid(True, alpha=0.3)
    
    # 6. Performance Attribution Summary
    ax6 = axes[2, 1]
    
    if attribution_engine.attribution_results and 'factor_summary' in attribution_engine.attribution_results:
        factor_summary = attribution_engine.attribution_results['factor_summary']
        
        # Select top contributors
        top_contributors = factor_summary['total_contribution'].abs().nlargest(8)
        
        colors = ['green' if x > 0 else 'red' for x in top_contributors.values]
        bars = ax6.bar(range(len(top_contributors)), top_contributors.values, 
                      color=colors, alpha=0.7)
        
        ax6.set_xticks(range(len(top_contributors)))
        ax6.set_xticklabels(top_contributors.index, rotation=45)
        ax6.set_ylabel('Total Contribution')
        ax6.set_title('Top Factor Contributors')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main_attribution_analysis():
    """Main performance attribution analysis"""
    
    print("=" * 80)
    print("PERFORMANCE ATTRIBUTION & RISK DECOMPOSITION ANALYSIS")
    print("=" * 80)
    
    # Create sample portfolio
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'JNJ', 'XOM', 'WMT', 'HD']
    
    print("Loading portfolio and benchmark data...")
    
    # Download portfolio data
    portfolio_data = yf.download(universe, start='2020-01-01', end='2024-01-01')['Adj Close']
    portfolio_returns_individual = portfolio_data.pct_change().dropna()
    
    # Create equal-weight portfolio
    portfolio_returns = portfolio_returns_individual.mean(axis=1)
    
    # Download benchmark (S&P 500)
    benchmark_data = yf.download('^GSPC', start='2020-01-01', end='2024-01-01')['Adj Close']
    benchmark_returns = benchmark_data.pct_change().dropna()
    
    # Initialize attribution engine
    attribution_engine = PerformanceAttribution()
    
    # Load factor data
    factor_returns = attribution_engine.load_factor_data(start_date='2020-01-01', end_date='2024-01-01')
    
    print("\n" + "=" * 60)
    print("FACTOR LOADING ANALYSIS")
    print("=" * 60)
    
    # Calculate factor loadings
    factor_loadings = attribution_engine.calculate_factor_loadings(portfolio_returns, method='regression')
    
    if factor_loadings and 'loadings' in factor_loadings:
        print("\nFactor Loadings (Betas):")
        loadings_df = factor_loadings['loadings'].sort_values(key=abs, ascending=False)
        
        for factor, loading in loadings_df.head(10).items():
            print(f"  {factor:<20}: {loading:>8.3f}")
        
        print(f"\nAlpha: {factor_loadings['alpha']*252:.4f} (annualized)")
        print(f"R-squared: {factor_loadings['r_squared']:.3f}")
    
    print("\n" + "=" * 60)
    print("FACTOR ATTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Perform factor attribution
    attribution_results = attribution_engine.factor_attribution(portfolio_returns, benchmark_returns)
    
    if attribution_results and 'factor_summary' in attribution_results:
        print("\nFactor Attribution Summary:")
        
        factor_summary = attribution_results['factor_summary']
        
        # Convert to more readable format
        summary_display = factor_summary.copy()
        summary_display['average_contribution'] *= 252 * 100  # Annualized percentage
        summary_display['volatility_contribution'] *= np.sqrt(252) * 100  # Annualized percentage
        
        # Sort by absolute total contribution
        summary_display = summary_display.reindex(
            summary_display['total_contribution'].abs().sort_values(ascending=False).index
        )
        
        print(f"{'Factor':<20} {'Avg Contrib (%)':<15} {'Vol Contrib (%)':<15} {'Loading':<10}")
        print("-" * 65)
        
        for factor in summary_display.head(10).index:
            row = summary_display.loc[factor]
            print(f"{factor:<20} {row['average_contribution']:>12.2f} "
                  f"{row['volatility_contribution']:>12.2f} {row['loading']:>8.3f}")
    
    print("\n" + "=" * 60)
    print("RISK DECOMPOSITION ANALYSIS")
    print("=" * 60)
    
    # Perform risk decomposition
    risk_decomp = attribution_engine.risk_decomposition(portfolio_returns, method='factor')
    
    if 'total_risk' in risk_decomp:
        print(f"\nRisk Decomposition:")
        print(f"  Total Risk:           {risk_decomp['total_risk']:.2%}")
        print(f"  Factor Risk:          {risk_decomp['factor_risk']:.2%} "
              f"({risk_decomp['factor_risk_percentage']:.1f}%)")
        print(f"  Idiosyncratic Risk:   {risk_decomp['idiosyncratic_risk']:.2%} "
              f"({risk_decomp['idiosyncratic_risk_percentage']:.1f}%)")
        
        if 'risk_contributions' in risk_decomp:
            print(f"\nTop Risk Contributors:")
            risk_contrib = risk_decomp['risk_contributions'].sort_values(key=abs, ascending=False)
            
            for factor, contrib in risk_contrib.head(5).items():
                print(f"  {factor:<20}: {contrib:.4f}")
    
    print("\n" + "=" * 60)
    print("STYLE ANALYSIS")
    print("=" * 60)
    
    # Perform style analysis
    style_results = attribution_engine.style_analysis(portfolio_returns)
    
    if style_results:
        print(f"\nStyle Analysis Results:")
        print(f"  R-squared:        {style_results['r_squared']:.3f}")
        print(f"  Tracking Error:   {style_results['tracking_error']:.2%}")
        
        print(f"\nStyle Exposures:")
        style_weights = style_results['style_weights'].sort_values(ascending=False)
        
        for style, weight in style_weights.items():
            if weight > 0.01:  # Only show weights > 1%
                print(f"  {style:<20}: {weight:.1%}")
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Calculate performance metrics
    portfolio_total_return = (1 + portfolio_returns).prod() - 1
    portfolio_annual_return = (1 + portfolio_returns.mean()) ** 252 - 1
    portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
    portfolio_sharpe = portfolio_annual_return / portfolio_volatility
    
    benchmark_total_return = (1 + benchmark_returns).prod() - 1
    benchmark_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
    benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
    benchmark_sharpe = benchmark_annual_return / benchmark_volatility
    
    # Active return metrics
    active_returns = portfolio_returns - benchmark_returns.reindex(portfolio_returns.index, method='ffill')
    information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252)
    
    print(f"Portfolio Performance:")
    print(f"  Total Return:     {portfolio_total_return:.2%}")
    print(f"  Annual Return:    {portfolio_annual_return:.2%}")
    print(f"  Volatility:       {portfolio_volatility:.2%}")
    print(f"  Sharpe Ratio:     {portfolio_sharpe:.3f}")
    
    print(f"\nBenchmark Performance:")
    print(f"  Total Return:     {benchmark_total_return:.2%}")
    print(f"  Annual Return:    {benchmark_annual_return:.2%}")
    print(f"  Volatility:       {benchmark_volatility:.2%}")
    print(f"  Sharpe Ratio:     {benchmark_sharpe:.3f}")
    
    print(f"\nActive Performance:")
    print(f"  Active Return:    {(portfolio_annual_return - benchmark_annual_return):.2%}")
    print(f"  Information Ratio: {information_ratio:.3f}")
    
    # Create visualizations
    create_attribution_visualizations(attribution_engine, portfolio_returns)
    
    return {
        'attribution_engine': attribution_engine,
        'portfolio_returns': portfolio_returns,
        'benchmark_returns': benchmark_returns,
        'factor_loadings': factor_loadings,
        'attribution_results': attribution_results,
        'risk_decomposition': risk_decomp,
        'style_analysis': style_results
    }

if __name__ == "__main__":
    # Run the attribution analysis
    results = main_attribution_analysis()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE ATTRIBUTION ANALYSIS COMPLETED")
    print("=" * 80)
    
    print("\nKey Insights:")
    print("• Factor loadings reveal systematic risk exposures")
    print("• Attribution identifies sources of active return")
    print("• Risk decomposition separates systematic vs idiosyncratic risk")
    print("• Style analysis provides investor-friendly exposure breakdown")
    
    print("\nPractical Applications:")
    print("• Portfolio risk management and monitoring")
    print("• Performance evaluation and manager selection")
    print("• Factor timing and tactical allocation decisions")
    print("• Client reporting and transparency")
    print("• Regulatory compliance and risk reporting")
    
    print("\nFramework Benefits:")
    print("• Comprehensive multi-factor analysis")
    print("• Industry-standard attribution methodologies")
    print("• Professional-grade risk decomposition")
    print("• Actionable insights for portfolio management")
    print("• Institutional-quality reporting capabilities")
    
    print(f"\nAttribution framework ready for deployment!")
    print("Complete Alpha project implementation finished!")
