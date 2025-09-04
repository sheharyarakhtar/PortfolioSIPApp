"""
Modern Portfolio Theory (MPT) Optimizer - Commercial PoC
A professional portfolio optimization tool using Modern Portfolio Theory
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import modules
from models.sip_strategy import SIPStrategyAnalyzer
from ui.mpt_components import MPTInterface
from utils.excel_export import ExcelExportManager
from utils.database_client import local_data_client

def main():
    """Main application function for MPT portfolio optimization."""
    
    # Page configuration
    st.set_page_config(
        page_title="MPT Portfolio Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("📊 Modern Portfolio Theory Optimizer")
    st.markdown("""
    **Professional Portfolio Optimization Tool**
    
    Optimize your investment portfolio using advanced mathematical models based on Modern Portfolio Theory.
    Build efficient portfolios that maximize returns for a given level of risk.
    """)
    
    # Tab selection using radio buttons for better control
    st.markdown("### Choose Your Analysis Type:")
    
    selected_tab = st.radio(
        "Select analysis type:",
        options=["🎯 Portfolio Optimization", "📊 Portfolio Composition Analysis"],
        index=0,
        horizontal=True,
        key="selected_tab",
        label_visibility="collapsed"
    )
    
    # Update active tab in session state
    if "🎯 Portfolio Optimization" in selected_tab:
        st.session_state.active_tab = 'optimization'
    else:
        st.session_state.active_tab = 'composition'
    
    # Create dynamic sidebar based on active tab
    render_dynamic_sidebar()
    
    st.markdown("---")
    
    # Render content based on selected tab
    if st.session_state.active_tab == 'optimization':
        render_portfolio_optimization_content()
    else:
        render_portfolio_composition_content()

def render_dynamic_sidebar():
    """Render sidebar content based on active tab."""
    
    # Determine which tab content to show
    active_tab = st.session_state.get('active_tab', 'optimization')
    
    with st.sidebar:
        if active_tab == 'optimization':
            render_optimization_sidebar()
        else:
            render_composition_sidebar()

def render_optimization_sidebar():
    """Render sidebar for portfolio optimization."""
    
    st.header("🎯 Portfolio Configuration")
    st.caption("⚡ Active: Portfolio Optimization")
    
    # Investment amount
    investment_amount = st.number_input(
        "Monthly Investment Amount ($)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=100,
        key="opt_investment_amount",
        help="Amount to invest monthly in your portfolio"
    )
    
    # Asset selection
    st.subheader("📈 Asset Selection")
    
    # Asset categories
    asset_categories = get_asset_categories()
    
    # Initialize session state for asset selections if not exists
    if 'asset_selections' not in st.session_state:
        st.session_state.asset_selections = {}
    
    # Asset selection interface with select all buttons
    for category, assets in asset_categories.items():
        with st.expander(category):
            # Select All / Deselect All buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button(f"✅ Select All", key=f"opt_select_all_{category}"):
                    for ticker in assets.keys():
                        st.session_state.asset_selections[ticker] = True
                    st.rerun()
            
            with col2:
                if st.button(f"❌ Deselect All", key=f"opt_deselect_all_{category}"):
                    for ticker in assets.keys():
                        st.session_state.asset_selections[ticker] = False
                    st.rerun()
            
            st.markdown("---")
            
            # Individual asset checkboxes
            for ticker, name in assets.items():
                current_state = st.session_state.asset_selections.get(ticker, False)
                if st.checkbox(f"{ticker} - {name}", value=current_state, key=f"opt_checkbox_{ticker}"):
                    st.session_state.asset_selections[ticker] = True
                else:
                    st.session_state.asset_selections[ticker] = False
    
    # Custom ticker search and selection
    st.subheader("🔍 Add Custom Tickers")
    
    # Get all available tickers from CSV
    all_available_tickers = local_data_client.get_available_tickers()
    
    # Get currently selected tickers
    currently_selected = [ticker for ticker, selected in st.session_state.asset_selections.items() if selected]
    
    # Filter out already selected tickers
    available_for_selection = [ticker for ticker in all_available_tickers if ticker not in currently_selected]
    
    if available_for_selection:
        # Search and select dropdown
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_ticker = st.selectbox(
                "Search and select a ticker:",
                options=[""] + available_for_selection,
                index=0,
                key="custom_ticker_select",
                help=f"Choose from {len(available_for_selection)} available tickers"
            )
        
        with col2:
            if st.button("➕ Add", disabled=not selected_ticker, key="add_custom_ticker"):
                if selected_ticker:
                    st.session_state.asset_selections[selected_ticker] = True
                    st.rerun()
        
        # Show currently selected custom tickers (not in predefined categories)
        asset_categories = get_asset_categories()
        predefined_tickers = set()
        for category_tickers in asset_categories.values():
            predefined_tickers.update(category_tickers.keys())
        
        custom_selected = [ticker for ticker in currently_selected if ticker not in predefined_tickers]
        
        if custom_selected:
            st.markdown("**Selected Custom Tickers:**")
            for ticker in custom_selected:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {ticker}")
                with col2:
                    if st.button("❌", key=f"remove_{ticker}", help=f"Remove {ticker}"):
                        st.session_state.asset_selections[ticker] = False
                        st.rerun()
    else:
        st.info("All available tickers are already selected!")
    
    # Optimization settings
    st.subheader("⚙️ Optimization Settings")
    
    optimization_methods = st.multiselect(
        "Optimization Methods",
        options=["Max Sharpe Ratio", "Min Variance", "Equal Weight"],
        default=["Max Sharpe Ratio", "Min Variance"],
        key="opt_methods",
        help="Select optimization strategies to compare"
    )
    
    rebalancing_frequencies = st.multiselect(
        "Rebalancing Frequencies",
        options=["3 Months", "6 Months", "12 Months", "Static"],
        default=["6 Months", "12 Months"],
        key="opt_frequencies",
        help="How often to rebalance the portfolio"
    )
    
    # Rebalancing strategy selection
    st.subheader("🔄 Rebalancing Strategy")
    
    rebalancing_strategy = st.radio(
        "Choose your rebalancing approach:",
        options=[
            "Both Strategies (Recommended)", 
            "Keep Holdings + Add New (Conservative)",
            "Sell & Reinvest All (Aggressive)"
        ],
        index=0,
        key="opt_rebalancing_strategy",
        help="Conservative: Keep existing holdings, only optimize new contributions. Aggressive: Sell everything and reinvest optimally."
    )
    
    st.caption("💡 **Conservative**: Lower transaction costs, tax-efficient. **Aggressive**: Higher transaction costs but fully optimal allocation.")
    
    # Analysis period
    st.subheader("📅 Analysis Period")
    
    period_options = {
        "Last 1 Year": 1,
        "Last 2 Years": 2,
        "Last 3 Years": 3,
        "Last 5 Years": 5,
        "Last 10 Years": 10,
        "Maximum Available": 15
    }
    
    selected_period = st.selectbox(
        "Time Period",
        options=list(period_options.keys()),
        index=2,  # Default to 3 years
        key="opt_period",
        help="Historical period for backtesting"
    )

def render_composition_sidebar():
    """Render sidebar for portfolio composition analysis."""
    
    st.header("📊 Composition Analysis Settings")
    st.caption("⚡ Active: Portfolio Composition Analysis")
    
    # Investment amount for composition analysis
    composition_amount = st.number_input(
        "Monthly Investment Amount ($)",
        min_value=100,
        max_value=100000,
        value=5000,
        step=100,
        key="comp_investment_amount",
        help="Amount invested monthly in the analysis"
    )
    
    # Asset selection for composition analysis
    st.subheader("📈 Select Assets for Analysis")
    
    # Same asset categories but with different keys
    asset_categories = get_asset_categories()
    
    # Initialize session state for composition asset selections
    if 'composition_asset_selections' not in st.session_state:
        st.session_state.composition_asset_selections = {}
    
    # Asset selection interface for composition analysis
    for category, assets in asset_categories.items():
        with st.expander(category):
            # Select All / Deselect All buttons for composition
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if st.button(f"✅ Select All", key=f"comp_select_all_{category}"):
                    for ticker in assets.keys():
                        st.session_state.composition_asset_selections[ticker] = True
                    st.rerun()
            
            with col2:
                if st.button(f"❌ Deselect All", key=f"comp_deselect_all_{category}"):
                    for ticker in assets.keys():
                        st.session_state.composition_asset_selections[ticker] = False
                    st.rerun()
            
            st.markdown("---")
            
            # Individual asset checkboxes for composition
            for ticker, name in assets.items():
                current_state = st.session_state.composition_asset_selections.get(ticker, False)
                if st.checkbox(f"{ticker} - {name}", value=current_state, key=f"comp_checkbox_{ticker}"):
                    st.session_state.composition_asset_selections[ticker] = True
                else:
                    st.session_state.composition_asset_selections[ticker] = False
    
    # Custom ticker search and selection for composition analysis
    st.subheader("🔍 Add Custom Tickers")
    
    # Get all available tickers from CSV
    all_available_tickers = local_data_client.get_available_tickers()
    
    # Get currently selected tickers for composition
    currently_selected = [ticker for ticker, selected in st.session_state.composition_asset_selections.items() if selected]
    
    # Filter out already selected tickers
    available_for_selection = [ticker for ticker in all_available_tickers if ticker not in currently_selected]
    
    if available_for_selection:
        # Search and select dropdown
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_ticker = st.selectbox(
                "Search and select a ticker:",
                options=[""] + available_for_selection,
                index=0,
                key="comp_custom_ticker_select",
                help=f"Choose from {len(available_for_selection)} available tickers"
            )
        
        with col2:
            if st.button("➕ Add", disabled=not selected_ticker, key="add_comp_custom_ticker"):
                if selected_ticker:
                    st.session_state.composition_asset_selections[selected_ticker] = True
                    st.rerun()
        
        # Show currently selected custom tickers (not in predefined categories)
        asset_categories = get_asset_categories()
        predefined_tickers = set()
        for category_tickers in asset_categories.values():
            predefined_tickers.update(category_tickers.keys())
        
        custom_selected = [ticker for ticker in currently_selected if ticker not in predefined_tickers]
        
        if custom_selected:
            st.markdown("**Selected Custom Tickers:**")
            for ticker in custom_selected:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"• {ticker}")
                with col2:
                    if st.button("❌", key=f"comp_remove_{ticker}", help=f"Remove {ticker}"):
                        st.session_state.composition_asset_selections[ticker] = False
                        st.rerun()
    else:
        st.info("All available tickers are already selected!")
    
    # Note: Strategy selection moved to main content area for better UX

def get_asset_categories():
    """Get the standard asset categories."""
    return {
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

def render_portfolio_optimization_content():
    """Render the main content for portfolio optimization tab."""
    
    # Get selected assets
    selected_assets = []
    for ticker, is_selected in st.session_state.get('asset_selections', {}).items():
        if is_selected:
            selected_assets.append(ticker)
    
    # Get settings from sidebar
    investment_amount = st.session_state.get('opt_investment_amount', 5000)
    optimization_methods = st.session_state.get('opt_methods', [])
    rebalancing_frequencies = st.session_state.get('opt_frequencies', [])
    rebalancing_strategy = st.session_state.get('opt_rebalancing_strategy', "Both Strategies (Recommended)")
    selected_period = st.session_state.get('opt_period', "Last 3 Years")
    
    # Calculate dates
    end_date = datetime.now()
    period_options = {
        "Last 1 Year": 1, "Last 2 Years": 2, "Last 3 Years": 3,
        "Last 5 Years": 5, "Last 10 Years": 10, "Maximum Available": 15
    }
    years_back = period_options.get(selected_period, 3)
    if selected_period == "Maximum Available":
                    start_date = datetime(2015, 1, 1)
    else:
        start_date = end_date - timedelta(days=years_back * 365)
    
    # Main content area
    if len(selected_assets) < 2:
        st.warning("⚠️ Please select at least 2 assets from the sidebar to begin optimization.")
        
        # Show sample portfolio for demonstration
        st.subheader("📊 Sample Portfolio Analysis")
        st.info("Here's what the analysis would look like with a sample portfolio:")
        
        demo_assets = ['VOO', 'VWO', 'BND', 'GLD']
        demo_config = {
            'tickers': demo_assets,
            'sip_amount': investment_amount,
            'transaction_cost_rate': 0.0025
        }
        
        if st.button("🚀 Run Demo Analysis"):
            # Use default settings for demo if none selected
            demo_methods = optimization_methods if optimization_methods else ["Max Sharpe Ratio", "Min Variance"]
            demo_frequencies = rebalancing_frequencies if rebalancing_frequencies else ["6 Months"]
            
            run_portfolio_analysis(demo_config, start_date.strftime('%Y-%m-%d'), 
                                 end_date.strftime('%Y-%m-%d'), demo_methods, 
                                 demo_frequencies, "Both Strategies (Recommended)", is_demo=True)
        
        # Display results if available (for both demo and regular analysis)
        if st.session_state.get('analysis_results'):
            display_analysis_results()
    
    elif not optimization_methods or not rebalancing_frequencies:
        st.warning("⚠️ Please select at least one optimization method and rebalancing frequency.")
    
    else:
        # Show selected configuration
        st.subheader("🎯 Portfolio Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Investment", f"${investment_amount:,}")
            
        with col2:
            st.metric("Selected Assets", len(selected_assets))
        
        with col3:
            st.metric("Analysis Period", selected_period)
        
        # Display selected assets
        with st.expander("📋 Selected Assets", expanded=True):
            asset_categories = get_asset_categories()
            asset_df = pd.DataFrame({
                'Ticker': selected_assets,
                'Asset Type': [get_asset_type(asset, asset_categories) for asset in selected_assets]
            })
            st.dataframe(asset_df, hide_index=True)
        
        # Run analysis button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
                config = {
                    'tickers': selected_assets,
                    'sip_amount': investment_amount,
                    'transaction_cost_rate': 0.0025
                }
                
                run_portfolio_analysis(
                    config, 
                    start_date.strftime('%Y-%m-%d'), 
                    end_date.strftime('%Y-%m-%d'),
                    optimization_methods,
                    rebalancing_frequencies,
                    rebalancing_strategy
                    )
        
        # Display results if available (for both demo and regular analysis)
        if st.session_state.get('analysis_results'):
            display_analysis_results()

def render_portfolio_composition_content():
    """Render the main content for portfolio composition analysis tab."""
    
    st.markdown("### 📊 Portfolio Composition Analysis")
    st.markdown("Analyze how your portfolio would have grown with different strategies and time periods.")
    
    # Get selected assets for composition
    composition_assets = []
    for ticker, is_selected in st.session_state.get('composition_asset_selections', {}).items():
        if is_selected:
            composition_assets.append(ticker)
    
    # Get settings from sidebar
    composition_amount = st.session_state.get('comp_investment_amount', 5000)
    
    # Main composition analysis interface
    if len(composition_assets) < 1:
        st.warning("⚠️ Please select at least 1 asset from the sidebar to begin analysis.")
        st.info("💡 You can select entire categories using the 'Select All' buttons.")
        
        # Show popular combinations as suggestions
        st.subheader("🎯 Popular Portfolio Combinations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Growth Portfolio", key="growth_portfolio"):
                growth_assets = ['VOO', 'QQQ', 'VWO', 'ARKK']
                for asset in growth_assets:
                    st.session_state.composition_asset_selections[asset] = True
                st.rerun()
            st.caption("VOO, QQQ, VWO, ARKK")
        
        with col2:
            if st.button("⚖️ Balanced Portfolio", key="balanced_portfolio"):
                balanced_assets = ['VOO', 'VWO', 'BND', 'GLD']
                for asset in balanced_assets:
                    st.session_state.composition_asset_selections[asset] = True
                st.rerun()
            st.caption("VOO, VWO, BND, GLD")
        
        with col3:
            if st.button("🛡️ Conservative Portfolio", key="conservative_portfolio"):
                conservative_assets = ['VOO', 'BND', 'SHY', 'GLD']
                for asset in conservative_assets:
                    st.session_state.composition_asset_selections[asset] = True
                st.rerun()
            st.caption("VOO, BND, SHY, GLD")
    
    else:
        # Show selected configuration
        st.subheader("🎯 Analysis Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Investment", f"${composition_amount:,}")
        
        with col2:
            st.metric("Selected Assets", len(composition_assets))
        
        with col3:
            st.metric("Available Strategies", "12 Total")
        
        # Display selected assets
        with st.expander("📋 Selected Assets", expanded=False):
            asset_categories = get_asset_categories()
            asset_df = pd.DataFrame({
                'Ticker': composition_assets,
                'Asset Type': [get_asset_type(asset, asset_categories) for asset in composition_assets]
            })
            st.dataframe(asset_df, hide_index=True)
        
        # Time period and strategy selection for analysis
        st.subheader("📅 Analysis Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            time_periods = {
                "6 Months": 0.5,
                "1 Year": 1,
                "2 Years": 2,
                "3 Years": 3,
                "5 Years": 5
            }
            
            selected_period = st.selectbox(
                "SIP Start Period",
                options=list(time_periods.keys()),
                index=2,
                key="comp_period",
                help="How long ago you would have started SIP"
            )
        
        with col2:
            # Build all available strategies (no need for sidebar selection)
            all_strategies = []
            
            methods = ["Max Sharpe Ratio", "Min Variance", "Equal Weight"]
            frequencies = ["6 Months", "12 Months"]
            rebalancing_approaches = ["Conservative", "Aggressive"]
            
            for method in methods:
                for freq in frequencies:
                    for rebal in rebalancing_approaches:
                        if method == "Equal Weight":
                            all_strategies.append(f"Equal Weight ({freq}) - {rebal}")
                        elif method == "Max Sharpe Ratio":
                            all_strategies.append(f"Max Sharpe ({freq}) - {rebal}")
                        elif method == "Min Variance":
                            all_strategies.append(f"Min Variance ({freq}) - {rebal}")
            
            selected_strategy = st.selectbox(
                "Strategy to Analyze",
                options=all_strategies,
                index=0,
                key="comp_strategy",
                help="Choose which strategy to analyze in detail"
            )
        
        with col3:
            st.write("") # Spacer
            if st.button("🔍 Analyze Portfolio Composition", type="primary", use_container_width=True):
                # Create independent analysis
                composition_config = {
                    'tickers': composition_assets,
                    'sip_amount': composition_amount,
                    'transaction_cost_rate': 0.0025
                }
                
                # Create mock analysis results for composition analysis
                MPTInterface._generate_composition_analysis(
                    {'config': composition_config}, 
                    selected_strategy, 
                    time_periods[selected_period]
                )
        
        # Display composition results if available
        if st.session_state.get('composition_analysis'):
            st.markdown("---")
            MPTInterface._display_composition_results()

def get_asset_type(ticker: str, asset_categories: Dict) -> str:
    """Get the category/type of an asset."""
    for category, assets in asset_categories.items():
        if ticker in assets:
            return category.replace("🇺🇸 ", "").replace("🌍 ", "").replace("🏛️ ", "").replace("₿ ", "").replace("🏭 ", "").replace("🥇 ", "")
    return "Custom"

def run_portfolio_analysis(config: Dict, start_date: str, end_date: str, 
                         optimization_methods: List[str], rebalancing_frequencies: List[str],
                         rebalancing_strategy: str = "Both Strategies (Recommended)", is_demo: bool = False):
    """Run comprehensive portfolio analysis."""
    
    analyzer = SIPStrategyAnalyzer(config)
    
    with st.spinner("🔄 Optimizing portfolio... This may take a few minutes."):
        try:
            progress_bar = st.progress(0)
            
            if is_demo:
                st.info("🎭 Running demo analysis with sample portfolio (VOO, VWO, BND, GLD)")
            
            # Validate tickers
            valid_tickers, invalid_tickers = analyzer.validate_tickers(config['tickers'])
            
            if invalid_tickers:
                st.warning(f"⚠️ {len(invalid_tickers)} assets unavailable: {', '.join(invalid_tickers)}")
            
            if len(valid_tickers) < 2:
                st.error("❌ Need at least 2 valid assets for optimization.")
                return
            
            # Update analyzer with valid tickers
            analyzer.available_tickers = valid_tickers
            progress_bar.progress(20)
            
            # Download data
            data = analyzer.download_data(start_date, end_date)
            progress_bar.progress(40)
            
            # Build strategies
            strategies = build_strategies(optimization_methods, rebalancing_frequencies, rebalancing_strategy)
            progress_bar.progress(60)
            
            # Run backtests
            results = {}
            for strategy in strategies:
                if strategy['method'] == 'equal_weight':
                    results[strategy['name']] = analyzer.backtest_equal_weight(data)
                else:
                    results[strategy['name']] = analyzer.backtest_strategy(data, strategy)
            
            # Add benchmark
            results['S&P 500 Benchmark'] = analyzer.backtest_benchmark(data)
            progress_bar.progress(80)
            
            # Calculate performance metrics
            performance_summary = {}
            for name, result in results.items():
                metrics = analyzer.calculate_performance_metrics(
                    result['portfolio_values'], 
                    result['monthly_contributions']
                )
                performance_summary[name] = metrics
            
            # Create analysis results
            analysis_results = {
                'results': results,
                'performance_summary': performance_summary,
                'data': data,
                'config': config
            }
            
            # Create charts
            charts = analyzer.create_interactive_charts(analysis_results)
            progress_bar.progress(100)
            
            # Store results
            st.session_state.analysis_results = analysis_results
            st.session_state.charts = charts
            
            progress_bar.empty()
            st.success("✅ Portfolio optimization completed!")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            st.error(f"❌ Error during analysis: {str(e)}")
            
            # Show more detailed error info in expander for debugging
            with st.expander("🔍 Error Details (for debugging)"):
                st.code(error_details)
                
            # Provide helpful suggestions based on error type
            if "array element with a sequence" in str(e):
                st.info("💡 **Tip:** This error often occurs with data formatting issues. Try selecting different assets or time periods.")
            elif "KeyError" in str(e):
                st.info("💡 **Tip:** This might be a configuration issue. Try refreshing the page and selecting assets again.")
            elif "optimization" in str(e).lower():
                st.info("💡 **Tip:** Optimization failed. Try selecting more diverse assets or a different time period.")

def build_strategies(optimization_methods: List[str], rebalancing_frequencies: List[str], 
                   rebalancing_strategy: str = "Both Strategies (Recommended)") -> List[Dict]:
    """Build strategy configurations from user selections."""
    
    strategies = []
    
    # Map frequency names to months
    freq_mapping = {
        "3 Months": 3,
        "6 Months": 6, 
        "12 Months": 12,
        "Static": 12
    }
    
    # Determine which rebalancing approaches to test
    if rebalancing_strategy == "Both Strategies (Recommended)":
        rebalancing_approaches = [
            ("Conservative", False),  # Keep holdings + add new
            ("Aggressive", True)      # Sell & reinvest all
        ]
    elif rebalancing_strategy == "Keep Holdings + Add New (Conservative)":
        rebalancing_approaches = [("Conservative", False)]
    else:  # Sell & Reinvest All (Aggressive)
        rebalancing_approaches = [("Aggressive", True)]
    
    for method in optimization_methods:
        for freq in rebalancing_frequencies:
            months = freq_mapping[freq]
            
            for approach_name, full_rebalancing in rebalancing_approaches:
                if method == "Equal Weight":
                    name = f'Equal Weight ({freq}) - {approach_name}'
                    strategies.append({
                        'name': name,
                        'rebalance_months': months,
                        'method': 'equal_weight',
                        'full_rebalancing': full_rebalancing
                    })
                elif method == "Max Sharpe Ratio":
                    name = f'Max Sharpe ({freq}) - {approach_name}'
                    strategies.append({
                        'name': name,
                        'rebalance_months': months,
                        'method': 'max_sharpe',
                        'full_rebalancing': full_rebalancing
                    })
                elif method == "Min Variance":
                    name = f'Min Variance ({freq}) - {approach_name}'
                    strategies.append({
                        'name': name,
                        'rebalance_months': months,
                        'method': 'min_variance',
                        'full_rebalancing': full_rebalancing
                    })
    
    return strategies

def display_analysis_results():
    """Display comprehensive analysis results."""
    
    analysis_results = st.session_state.analysis_results
    charts = st.session_state.charts
    
    st.markdown("---")
    st.header("📊 Portfolio Optimization Results")
    
    # Use the MPT interface to display results
    MPTInterface.render_performance_summary(analysis_results)
    MPTInterface.render_interactive_charts(charts)
    MPTInterface.render_optimal_allocation(analysis_results)
    
    st.markdown("---")
    MPTInterface.render_export_options(analysis_results)

if __name__ == "__main__":
    main()