"""
Configuration settings for the MPT Portfolio Optimizer.
Streamlined configuration focused on portfolio optimization.
"""

# App configuration
APP_CONFIG = {
    'page_title': "MPT Portfolio Optimizer",
    'page_icon': "📊",
    'layout': "wide"
}

# Default portfolio configuration
DEFAULT_PORTFOLIO_CONFIG = {
    'monthly_investment': 5000,
    'transaction_cost_rate': 0.0025,  # 0.25% transaction costs
    'risk_free_rate': 0.04  # 4% risk-free rate
}

# Comprehensive asset universe for portfolio optimization
ASSET_UNIVERSE = {
    # US Equity ETFs
    'VOO': 'S&P 500 ETF',
    'QQQ': 'Nasdaq 100 ETF', 
    'VTI': 'Total US Market ETF',
    'IWM': 'Russell 2000 Small Cap ETF',
    'ARKK': 'Innovation ETF',
    
    # International ETFs
    'VWO': 'Emerging Markets ETF',
    'VEA': 'Developed Markets ETF',
    'VXUS': 'Total International Stock ETF',
    'EFA': 'EAFE ETF',
    'VGK': 'European ETF',
    
    # Bonds & Fixed Income
    'BND': 'Total Bond Market ETF',
    'TLT': 'Long-Term Treasury ETF',
    'SHY': 'Short-Term Treasury ETF',
    'LQD': 'Investment Grade Corporate Bonds',
    'HYG': 'High Yield Corporate Bonds',
    
    # Cryptocurrency
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'ADA-USD': 'Cardano',
    'SOL-USD': 'Solana',
    
    # Sector ETFs
    'XLK': 'Technology Sector ETF',
    'XLF': 'Financial Sector ETF',
    'XLV': 'Healthcare Sector ETF',
    'XLE': 'Energy Sector ETF',
    'XLI': 'Industrial Sector ETF',
    
    # Commodities & REITs
    'GLD': 'Gold ETF',
    'SLV': 'Silver ETF',
    'VNQ': 'Real Estate ETF',
    'USO': 'Oil ETF',
    'DBA': 'Agriculture ETF'
}

# Asset categories for UI organization
ASSET_CATEGORIES = {
    "🇺🇸 US Equity ETFs": {
        'VOO': 'S&P 500 ETF',
        'QQQ': 'Nasdaq 100 ETF', 
        'VTI': 'Total US Market ETF',
        'IWM': 'Russell 2000 ETF',
        'ARKK': 'Innovation ETF'
    },
    "🌍 International ETFs": {
        'VWO': 'Emerging Markets ETF',
        'VEA': 'Developed Markets ETF',
        'VXUS': 'Total International ETF',
        'EFA': 'EAFE ETF',
        'VGK': 'European ETF'
    },
    "🏛️ Bonds & Fixed Income": {
        'BND': 'Total Bond Market ETF',
        'TLT': 'Long-Term Treasury ETF',
        'SHY': 'Short-Term Treasury ETF',
        'LQD': 'Investment Grade Corporate',
        'HYG': 'High Yield Corporate'
    },
    "₿ Cryptocurrency": {
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        'ADA-USD': 'Cardano',
        'SOL-USD': 'Solana'
    },
    "🏭 Sector ETFs": {
        'XLK': 'Technology Sector',
        'XLF': 'Financial Sector',
        'XLV': 'Healthcare Sector',
        'XLE': 'Energy Sector',
        'XLI': 'Industrial Sector'
    },
    "🥇 Commodities & REITs": {
        'GLD': 'Gold ETF',
        'SLV': 'Silver ETF',
        'VNQ': 'Real Estate ETF',
        'USO': 'Oil ETF',
        'DBA': 'Agriculture ETF'
    }
}

# Optimization methods available
OPTIMIZATION_METHODS = [
    "Max Sharpe Ratio",
    "Min Variance", 
    "Equal Weight"
]

# Rebalancing frequencies
REBALANCING_FREQUENCIES = [
    "3 Months",
    "6 Months", 
    "12 Months",
    "Static"
]

# Analysis periods
ANALYSIS_PERIODS = {
    "Last 1 Year": 1,
    "Last 2 Years": 2,
    "Last 3 Years": 3,
    "Last 5 Years": 5,
    "Last 10 Years": 10,
    "Maximum Available": 15
}

# Market regime parameters for optimization
REGIME_PARAMETERS = {
    "bull_low_vol": {"max_weight": 0.50, "min_weight": 0.02, "rf": 0.04},
    "bull_high_vol": {"max_weight": 0.40, "min_weight": 0.03, "rf": 0.04},
    "bear_high_vol": {"max_weight": 0.30, "min_weight": 0.05, "rf": 0.03},
    "neutral": {"max_weight": 0.40, "min_weight": 0.02, "rf": 0.04}
}

# Performance thresholds for regime detection
REGIME_THRESHOLDS = {
    "high_return": 0.15,
    "medium_return": 0.10,
    "low_return": 0.05,
    "low_volatility": 0.25,
    "high_volatility": 0.30
}

# Export settings
EXPORT_CONFIG = {
    'include_charts': True,
    'include_allocations': True,
    'include_performance_metrics': True,
    'date_format': '%Y-%m-%d'
}