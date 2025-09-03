"""
Configuration settings for the MPT Portfolio Optimizer app.
Contains all default values, constants, and configuration data.
"""

import yfinance as yf

# Exchange rates (cached data)
def get_live_rates():
    """Get live exchange rates from Yahoo Finance."""
    try:
        return {
            'usd_pkr_rate': yf.Ticker("PKR=X").info['regularMarketPrice'],
            'usdt_usd_rate': yf.Ticker("USDT-USD").info['regularMarketPrice'],
            'gbp_usd_rate': yf.Ticker("GBP=X").info['regularMarketPrice']
        }
    except:
        # Fallback rates if API fails
        return {
            'usd_pkr_rate': 280.0,
            'usdt_usd_rate': 1.0,
            'gbp_usd_rate': 1.25
        }

def get_cached_rates():
    """Get cached exchange rates, fetch new ones only if not cached."""
    import streamlit as st
    from datetime import datetime
    
    # Initialize session state for exchange rates if not exists
    if 'exchange_rates' not in st.session_state:
        st.session_state.exchange_rates = get_live_rates()
        st.session_state.rates_last_updated = datetime.now()
    
    return st.session_state.exchange_rates

def refresh_exchange_rates():
    """Force refresh exchange rates."""
    import streamlit as st
    from datetime import datetime
    
    with st.spinner("Fetching latest exchange rates..."):
        st.session_state.exchange_rates = get_live_rates()
        st.session_state.rates_last_updated = datetime.now()
    
    return st.session_state.exchange_rates

# Default configuration
DEFAULT_CONFIG = {
    # Income defaults
    'monthly_income': 1800,
    
    # Fixed PKR allocations (configurable in app)
    'pkr_allocations': {
        'Dependents Allowance': 0,
        'My Allowance': 0,
        'Family Fund': 0,
        'Emergency Fund': 0
    }
}

# App configuration
APP_CONFIG = {
    'page_title': "MPT-Based SIP Portfolio Optimizer",
    'page_icon': "📈",
    'layout': "wide"
}

# Ticker mapping for yfinance (comprehensive)
TICKER_MAPPING = {
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
