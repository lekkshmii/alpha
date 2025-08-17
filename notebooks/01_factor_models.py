# Notebook 1: Fama-French Factor Models Implementation

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class FamaFrenchFactors:
    """
    Implementation of Fama-French Three and Five Factor Models
    Based on: Fama, E. F., & French, K. R. (1993, 2015)
    """
    
    def __init__(self):
        self.factors_3f = None
        self.factors_5f = None
        self.universe = None
        
    def load_market_data(self, tickers, start_date='2020-01-01', end_date='2024-01-01'):
        """Load stock price data and calculate returns"""
        print("Loading market data...")
        
        # Download price data
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        
        # Calculate returns
        returns = data.pct_change().dropna()
        
        # Load risk-free rate (using 3M Treasury as proxy)
        rf_data = yf.download('^IRX', start=start_date, end=end_date)['Adj Close']
        rf_rate = (rf_data / 100) / 252  # Convert to daily decimal
        
        # Align dates
        common_dates = returns.index.intersection(rf_rate.index)
        returns = returns.loc[common_dates]
        rf_rate = rf_rate.loc[common_dates]
        
        self.returns = returns
        self.rf_rate = rf_rate
        
        print(f"Loaded data for {len(tickers)} stocks from {start_date} to {end_date}")
        return returns, rf_rate
    
    def get_market_data(self, start_date='2020-01-01', end_date='2024-01-01'):
        """Get market and size/value proxy data"""
        
        # Market portfolio (S&P 500)
        market = yf.download('^GSPC', start=start_date, end=end_date)['Adj Close']
        market_returns = market.pct_change().dropna()
        
        # Size factor proxies
        small_cap = yf.download('IWM', start=start_date, end=end_date)['Adj Close']  # Russell 2000
        large_cap = yf.download('IWB', start=start_date, end=end_date)['Adj Close']  # Russell 1000
        
        small_returns = small_cap.pct_change().dropna()
        large_returns = large_cap.pct_change().dropna()
        
        # Value factor proxies
        value = yf.download('IWD', start=start_date, end=end_date)['Adj Close']  # Value ETF
        growth = yf.download('IWF', start=start_date, end=end_date)['Adj Close']  # Growth ETF
        
        value_returns = value.pct_change().dropna()
        growth_returns = growth.pct_change().dropna()
        
        # Profitability and Investment proxies (for 5-factor)
        profit_proxy = yf.download('QUAL', start=start_date, end=end_date)['Adj Close']  # Quality ETF
        investment_proxy = yf.download('MTUM', start=start_date, end=end_date)['Adj Close']  # Momentum ETF
        
        profit_returns = profit_proxy.pct_change().dropna()
        investment_returns = investment_proxy.pct_change().dropna()
        
        return {
            'market': market_returns,
            'small': small_returns,
            'large': large_returns,
            'value': value_returns,
            'growth': growth_returns,
            'profit': profit_returns,
            'investment': investment_returns
        }
    
    def construct_factors(self, start_date='2020-01-01', end_date='2024-01-01'):
        """Construct Fama-French factors"""
        
        factor_data = self.get_market_data(start_date, end_date)
        
        # Align all data to common dates
        common_dates = None
        for key, data in factor_data.items():
            if common_dates is None:
                common_dates = data.index
            else:
                common_dates = common_dates.intersection(data.index)
        
        # Market factor (excess return)
        market_excess = factor_data['market'].loc[common_dates] - self.rf_rate.reindex(common_dates).fillna(0)
        
        # Size factor (SMB - Small Minus Big)
        smb = factor_data['small'].loc[common_dates] - factor_data['large'].loc[common_dates]
        
        # Value factor (HML - High Minus Low)
        hml = factor_data['value'].loc[common_dates] - factor_data['growth'].loc[common_dates]
        
        # Profitability factor (RMW - Robust Minus Weak)
        rmw = factor_data['profit'].loc[common_dates] - factor_data['investment'].loc[common_dates]
        
        # Investment factor (CMA - Conservative Minus Aggressive)
        cma = -factor_data['investment'].loc[common_dates]  # Simplified proxy
        
        # Three-factor model
        self.factors_3f = pd.DataFrame({
            'Mkt-RF': market_excess,
            'SMB': smb,
            'HML': hml
        })
        
        # Five-factor model
        self.factors_5f = pd.DataFrame({
            'Mkt-RF': market_excess,
            'SMB': smb,
            'HML': hml,
            'RMW': rmw,
            'CMA': cma
        })
        
        print("Factors constructed successfully!")
        return self.factors_3f, self.factors_5f
    
    def run_factor_regression(self, stock_returns, factors, stock_name):
        """Run factor regression for a single stock"""
        
        # Align data
        common_dates = stock_returns.dropna().index.intersection(factors.dropna().index)
        y = stock_returns.loc[common_dates] - self.rf_rate.reindex(common_dates).fillna(0)
        X = factors.loc[common_dates]
        
        # Run regression
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        
        # Create results dictionary
        results = {
            'stock': stock_name,
            'alpha': model.intercept_,
            'r2': r2,
            'coefficients': dict(zip(X.columns, model.coef_))
        }
        
        return results
    
    def analyze_portfolio(self, tickers, start_date='2020-01-01', end_date='2024-01-01'):
        """Analyze a portfolio of stocks using factor models"""
        
        # Load data
        returns, rf_rate = self.load_market_data(tickers, start_date, end_date)
        
        # Construct factors
        factors_3f, factors_5f = self.construct_factors(start_date, end_date)
        
        results_3f = []
        results_5f = []
        
        print("Running factor regressions...")
        
        for ticker in tickers:
            if ticker in returns.columns:
                # Three-factor analysis
                result_3f = self.run_factor_regression(returns[ticker], factors_3f, ticker)
                results_3f.append(result_3f)
                
                # Five-factor analysis
                result_5f = self.run_factor_regression(returns[ticker], factors_5f, ticker)
                results_5f.append(result_5f)
        
        return pd.DataFrame(results_3f), pd.DataFrame(results_5f)

# Example Usage and Demonstration
def main_analysis():
    """Main analysis demonstrating the Fama-French implementation"""
    
    # Initialize the factor model
    ff_model = FamaFrenchFactors()
    
    # Define a universe of stocks for analysis
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA']
    
    print("=" * 60)
    print("FAMA-FRENCH FACTOR MODEL ANALYSIS")
    print("=" * 60)
    
    # Run the analysis
    results_3f, results_5f = ff_model.analyze_portfolio(
        tech_stocks, 
        start_date='2020-01-01', 
        end_date='2024-01-01'
    )
    
    # Display results
    print("\n" + "=" * 40)
    print("THREE-FACTOR MODEL RESULTS")
    print("=" * 40)
    
    # Create summary table
    summary_3f = results_3f.copy()
    
    # Extract coefficient columns
    coef_cols = ['Mkt-RF', 'SMB', 'HML']
    for col in coef_cols:
        summary_3f[col] = summary_3f['coefficients'].apply(lambda x: x.get(col, np.nan))
    
    # Display key metrics
    display_cols = ['stock', 'alpha', 'r2'] + coef_cols
    print(summary_3f[display_cols].round(4))
    
    print("\n" + "=" * 40)
    print("FIVE-FACTOR MODEL RESULTS") 
    print("=" * 40)
    
    # Create summary table for 5-factor
    summary_5f = results_5f.copy()
    
    # Extract coefficient columns
    coef_cols_5f = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    for col in coef_cols_5f:
        summary_5f[col] = summary_5f['coefficients'].apply(lambda x: x.get(col, np.nan))
    
    # Display key metrics
    display_cols_5f = ['stock', 'alpha', 'r2'] + coef_cols_5f
    print(summary_5f[display_cols_5f].round(4))
    
    # Visualization
    create_factor_visualizations(summary_3f, summary_5f)
    
    return summary_3f, summary_5f

def create_factor_visualizations(results_3f, results_5f):
    """Create professional visualizations of factor analysis results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Fama-French Factor Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Alpha comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(results_3f))
    ax1.bar(x_pos - 0.2, results_3f['alpha'], 0.4, label='3-Factor', alpha=0.7)
    ax1.bar(x_pos + 0.2, results_5f['alpha'], 0.4, label='5-Factor', alpha=0.7)
    ax1.set_xlabel('Stocks')
    ax1.set_ylabel('Alpha')
    ax1.set_title('Alpha Comparison: 3-Factor vs 5-Factor')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(results_3f['stock'], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. R-squared comparison
    ax2 = axes[0, 1]
    ax2.bar(x_pos - 0.2, results_3f['r2'], 0.4, label='3-Factor', alpha=0.7)
    ax2.bar(x_pos + 0.2, results_5f['r2'], 0.4, label='5-Factor', alpha=0.7)
    ax2.set_xlabel('Stocks')
    ax2.set_ylabel('R-squared')
    ax2.set_title('Model Fit: R-squared Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_3f['stock'], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Market beta distribution
    ax3 = axes[1, 0]
    market_betas = results_3f['Mkt-RF']
    ax3.hist(market_betas, bins=8, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Market Beta')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Market Betas')
    ax3.grid(True, alpha=0.3)
    
    # 4. Factor loadings heatmap
    ax4 = axes[1, 1]
    factor_matrix = results_3f[['Mkt-RF', 'SMB', 'HML']].T
    factor_matrix.columns = results_3f['stock']
    
    im = ax4.imshow(factor_matrix, cmap='RdBu_r', aspect='auto')
    ax4.set_xticks(range(len(results_3f['stock'])))
    ax4.set_xticklabels(results_3f['stock'], rotation=45)
    ax4.set_yticks(range(3))
    ax4.set_yticklabels(['Market', 'Size', 'Value'])
    ax4.set_title('Factor Loadings Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Run the main analysis
    summary_3f, summary_5f = main_analysis()
    
    print("\n" + "=" * 60)
    print("FACTOR MODEL INSIGHTS")
    print("=" * 60)
    
    # Key insights
    avg_alpha_3f = summary_3f['alpha'].mean()
    avg_alpha_5f = summary_5f['alpha'].mean()
    avg_r2_3f = summary_3f['r2'].mean()
    avg_r2_5f = summary_5f['r2'].mean()
    
    print(f"Average Alpha (3-Factor): {avg_alpha_3f:.4f}")
    print(f"Average Alpha (5-Factor): {avg_alpha_5f:.4f}")
    print(f"Average R² (3-Factor): {avg_r2_3f:.4f}")
    print(f"Average R² (5-Factor): {avg_r2_5f:.4f}")
    
    # Identify stocks with significant alpha
    significant_alpha = summary_5f[abs(summary_5f['alpha']) > 0.001]
    
    if len(significant_alpha) > 0:
        print(f"\nStocks with significant alpha (>0.1% daily):")
        for _, row in significant_alpha.iterrows():
            print(f"{row['stock']}: {row['alpha']:.4f} ({row['alpha']*252:.2%} annualized)")
    
    print("\nAnalysis completed successfully!")
