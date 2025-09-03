"""
MPT-Based Portfolio Optimizer - Standalone Application
Modern Portfolio Theory meets Systematic Investment Plans
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

# Import modules with corrected paths
from config.settings import get_cached_rates, refresh_exchange_rates, APP_CONFIG
from models.sip_strategy import SIPStrategyAnalyzer
from ui.simplified_components import SimplifiedInputManager, SimplifiedDisplayManager
from ui.sip_components import SIPAnalysisManager
from utils.excel_export import ExcelExportManager, create_financial_summary_from_session

def main():
    """Main application function focused on MPT-based SIP optimization."""
    
    # Page configuration
    st.set_page_config(
        page_title="MPT-Based SIP Optimizer",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 MPT-Based SIP Portfolio Optimizer")
    st.markdown("""
    **Modern Portfolio Theory meets Systematic Investment Plans**
    
    This app helps you optimize your monthly investments using advanced mathematical models while keeping local investments separate.
    """)
    
    # Exchange rates section
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💱 Exchange Rates")
    with col2:
        if st.button("🔄 Refresh Rates"):
            exchange_rates = refresh_exchange_rates()
            st.success("Rates updated!")
        else:
            exchange_rates = get_cached_rates()
    
    # Store exchange rates in session state for export
    st.session_state.exchange_rates = exchange_rates
    
    # Display current rates
    if hasattr(st.session_state, 'rates_last_updated'):
        st.caption(f"Last updated: {st.session_state.rates_last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("USD/PKR", f"{exchange_rates['usd_pkr_rate']:.2f}")
    with col2:
        st.metric("USDT/USD", f"{exchange_rates['usdt_usd_rate']:.4f}")
    with col3:
        st.metric("GBP/USD", f"{exchange_rates['gbp_usd_rate']:.4f}")
    
    st.markdown("---")
    
    # Step 1: Income Setup
    total_income, usd_pkr_rate = SimplifiedInputManager.render_income_and_pkr_expenses(exchange_rates)
    
    # Store in session state for export
    st.session_state.total_income = total_income
    st.session_state.usd_pkr_rate = usd_pkr_rate
    
    st.markdown("---")
    
    # Step 2: Configure PKR Expenses
    pkr_expenses_usd = SimplifiedInputManager.render_configurable_pkr_expenses(usd_pkr_rate)
    
    # Store in session state for export
    st.session_state.pkr_expenses_usd = pkr_expenses_usd
    
    st.markdown("---")
    
    # Step 3: Investment Allocation
    available_usd, international_amount, local_amount, cash_amount = SimplifiedInputManager.render_investment_allocation_inputs(total_income, pkr_expenses_usd)
    
    # Store in session state for export
    st.session_state.available_usd = available_usd
    st.session_state.international_amount = international_amount
    st.session_state.local_amount = local_amount
    st.session_state.cash_amount = cash_amount
    
    if available_usd <= 0:
        st.error("Cannot proceed without available investment funds.")
        return
    
    st.markdown("---")
    
    # Step 4: Local allocation guide
    SimplifiedDisplayManager.render_local_allocation_guide(local_amount)
    
    st.markdown("---")
    
    # Step 5: MPT Optimization (only if international amount > 0)
    if international_amount > 0:
        st.header("🧠 Modern Portfolio Theory Optimization")
        st.markdown(f"""
        **Monthly Amount for MPT Optimization: ${international_amount:,.0f}**
        
        This section will optimize your international investments using advanced mathematical models:
        - **Max Sharpe Ratio**: Maximizes return per unit of risk
        - **Min Variance**: Minimizes portfolio volatility
        - **Dynamic Rebalancing**: Tests different rebalancing frequencies
        - **Market Regime Awareness**: Adapts to bull/bear markets
        """)
        
        # MPT Configuration
        selected_assets, rebalancing_strategies = SimplifiedInputManager.render_mpt_configuration()
        
        # Custom allocation input
        st.markdown("---")
        custom_allocation = SimplifiedInputManager.render_custom_allocation_input(selected_assets)
        
        if len(selected_assets) >= 2 and rebalancing_strategies:
            # SIP Analysis Configuration
            st.subheader("📅 Analysis Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 Analysis Period**")
                
                # Date range presets
                preset_option = st.selectbox(
                    "Choose time period:",
                    options=[
                        "Custom Range",
                        "Last 1 Year", 
                        "Last 2 Years",
                        "Last 3 Years", 
                        "Last 5 Years",
                        "Last 10 Years",
                        "Maximum Available (2015-Present)"
                    ],
                    index=2  # Default to "Last 3 Years"
                )
                
                # Calculate dates based on preset
                end_date = datetime.now()
                if preset_option == "Last 1 Year":
                    start_date = end_date - pd.DateOffset(years=1)
                elif preset_option == "Last 2 Years":
                    start_date = end_date - pd.DateOffset(years=2)
                elif preset_option == "Last 3 Years":
                    start_date = end_date - pd.DateOffset(years=3)
                elif preset_option == "Last 5 Years":
                    start_date = end_date - pd.DateOffset(years=5)
                elif preset_option == "Last 10 Years":
                    start_date = end_date - pd.DateOffset(years=10)
                elif preset_option == "Maximum Available (2015-Present)":
                    start_date = datetime(2015, 1, 1)
                else:  # Custom Range
                    start_date = datetime(2020, 1, 1)
                
                # Show custom date picker only if "Custom Range" is selected
                if preset_option == "Custom Range":
                    date_range = st.date_input(
                        "Select custom date range:",
                        value=(start_date.date(), end_date.date()),
                        min_value=datetime(2015, 1, 1).date(),
                        max_value=end_date.date()
                    )
                    
                    if len(date_range) == 2:
                        start_date_str = date_range[0].strftime('%Y-%m-%d')
                        end_date_str = date_range[1].strftime('%Y-%m-%d')
                    else:
                        start_date_str = start_date.strftime('%Y-%m-%d')
                        end_date_str = end_date.strftime('%Y-%m-%d')
                else:
                    start_date_str = start_date.strftime('%Y-%m-%d')
                    end_date_str = end_date.strftime('%Y-%m-%d')
                    st.info(f"📊 **Analysis Period**: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            with col2:
                st.markdown("**📊 Analysis Features**")
                st.info("**🚀 Enhanced Analysis**: Both rebalancing approaches will be tested simultaneously for complete comparison!")
                
                st.markdown("**Rebalancing Strategies Included:**")
                st.markdown("• **🟢 New Contrib**: Existing holdings stay, only new contributions optimized")
                st.markdown("• **🔴 Full Rebal**: Entire portfolio rebalanced (higher transaction costs)")
                
                st.markdown("**Other Features:**")
                st.markdown("• Historical backtesting")
                st.markdown("• Risk-adjusted optimization") 
                st.markdown("• Transaction cost modeling (0.25% per rebalance)")
                st.markdown("• Interactive filtering in results")
                
                st.caption("💡 **Transaction Costs**: Includes bid-ask spreads, market impact, and rebalancing overhead. Most major brokerages offer commission-free ETF trading.")
            
            # Analysis Button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run MPT Analysis", type="primary"):
                    run_mpt_analysis(
                        selected_assets, 
                        rebalancing_strategies, 
                        international_amount, 
                        start_date_str, 
                        end_date_str,
                        custom_allocation
                    )
            
            # Display results if available
            if st.session_state.get('mpt_analysis_results'):
                display_mpt_results(international_amount)
        
        else:
            st.warning("⚠️ Please select at least 2 assets and 1 rebalancing strategy to run analysis.")
    
    else:
        st.info("💡 No funds allocated to international markets. Increase international allocation to run MPT analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **📝 Key Features:**
    - ✅ **Configurable PKR expenses** - Perfect for sharing with different users
    - ✅ **Ticker validation** - Automatically handles unavailable securities
    - ✅ **MPT optimization** - Mathematically optimal allocations
    - ✅ **Custom allocation testing** - Compare your strategy vs optimal
    - ✅ **Interactive charts** - Comprehensive performance visualization
    - ✅ **Export functionality** - Download results for implementation
    """)

def run_mpt_analysis(selected_assets: List[str], rebalancing_strategies: Dict[str, int], 
                    sip_amount: float, start_date: str, end_date: str, custom_allocation: Optional[Dict[str, float]] = None):
    """Run MPT analysis on selected assets with optional custom allocation."""
    
    # Configure analyzer with dynamic asset selection
    sip_config = {
        'tickers': selected_assets,  # Use dynamically selected assets
        'sip_amount': sip_amount,  # Use international amount only
        'transaction_cost_rate': 0.0025  # 0.25% - more realistic for bid-ask spreads + rebalancing costs
    }
    
    # Always create new analyzer with current asset selection
    # (since assets can change dynamically)
    analyzer = SIPStrategyAnalyzer(sip_config)
    
    # Cache the analyzer for the session
    st.session_state.mpt_analyzer = analyzer
    
    with st.spinner("🔄 Running MPT analysis... This may take a few minutes."):
        try:
            progress_bar = st.progress(0)
            st.caption(f"🎯 Analyzing {len(selected_assets)} assets with ${sip_amount:,.0f}/month")
            
            # Validate tickers first
            st.caption("🔍 Validating ticker availability...")
            valid_tickers, invalid_tickers = analyzer.validate_tickers(selected_assets)
            
            if invalid_tickers:
                st.warning(f"⚠️ {len(invalid_tickers)} tickers are not available: {', '.join(invalid_tickers)}")
                st.info(f"✅ Continuing analysis with {len(valid_tickers)} valid tickers: {', '.join(valid_tickers)}")
            
            if len(valid_tickers) < 2:
                st.error("❌ Need at least 2 valid tickers for analysis. Please select different assets.")
                return
            
            # Update analyzer with valid tickers only
            analyzer.available_tickers = valid_tickers
            selected_assets = valid_tickers  # Use only valid tickers
            
            progress_bar.progress(20)
            
            # Build strategies based on user selection - ALWAYS test both rebalancing approaches
            strategies_to_test = []
            
            for strategy_name, months in rebalancing_strategies.items():
                if "Static" in strategy_name:
                    # Add both rebalancing approaches for Static Equal Weight
                    strategies_to_test.extend([
                        {
                            'name': f'Static Equal Weight (New Contrib)',
                            'rebalance_months': months,
                            'method': 'equal_weight',
                            'full_rebalancing': False
                        },
                        {
                            'name': f'Static Equal Weight (Full Rebal)',
                            'rebalance_months': months,
                            'method': 'equal_weight',
                            'full_rebalancing': True
                        }
                    ])
                else:
                    # Add both Max Sharpe and Min Variance for each frequency, with both rebalancing approaches
                    strategies_to_test.extend([
                        # New Contrib versions
                        {
                            'name': f'Max Sharpe ({months}M) (New Contrib)',
                            'rebalance_months': months,
                            'method': 'max_sharpe',
                            'full_rebalancing': False
                        },
                        {
                            'name': f'Min Variance ({months}M) (New Contrib)',
                            'rebalance_months': months,
                            'method': 'min_variance',
                            'full_rebalancing': False
                        },
                        # Full Rebal versions
                        {
                            'name': f'Max Sharpe ({months}M) (Full Rebal)',
                            'rebalance_months': months,
                            'method': 'max_sharpe',
                            'full_rebalancing': True
                        },
                        {
                            'name': f'Min Variance ({months}M) (Full Rebal)',
                            'rebalance_months': months,
                            'method': 'min_variance',
                            'full_rebalancing': True
                        }
                    ])
            
            st.caption(f"Testing {len(strategies_to_test)} strategies...")
            progress_bar.progress(40)
            
            # Run analysis
            data = analyzer.download_data(start_date, end_date)
            progress_bar.progress(60)
            
            results = {}
            
            # Backtest strategies
            for strategy in strategies_to_test:
                if strategy['method'] == 'equal_weight':
                    results[strategy['name']] = analyzer.backtest_equal_weight(data)
                else:
                    results[strategy['name']] = analyzer.backtest_strategy(data, strategy)
            
            # Add custom allocation if provided
            if custom_allocation:
                # Convert custom allocation to format expected by backtest_user_allocation
                custom_formatted = {}
                asset_names = {
                    # US Equity ETFs
                    'VOO': 'VOO (S&P 500)', 'QQQ': 'QQQ (Nasdaq 100)', 'VTI': 'VTI (Total US Market)',
                    'IWM': 'IWM (Russell 2000)', 'ARKK': 'ARKK (Innovation)',
                    # International ETFs
                    'VWO': 'VWO (Emerging Markets)', 'VEA': 'VEA (Developed Markets)', 
                    'VXUS': 'VXUS (Total International)', 'EFA': 'EFA (EAFE)', 'VGK': 'VGK (European)',
                    # Bonds & Fixed Income
                    'BND': 'BND (Total Bond Market)', 'TLT': 'TLT (Long-Term Treasury)',
                    'SHY': 'SHY (Short-Term Treasury)', 'LQD': 'LQD (Investment Grade Corp)',
                    'HYG': 'HYG (High Yield Corp)',
                    # Cryptocurrency
                    'BTC-USD': 'Bitcoin (BTC)', 'ETH-USD': 'Ethereum (ETH)',
                    'ADA-USD': 'Cardano (ADA)', 'SOL-USD': 'Solana (SOL)',
                    # Sector ETFs
                    'XLK': 'XLK (Technology)', 'XLF': 'XLF (Financial)', 'XLV': 'XLV (Healthcare)',
                    'XLE': 'XLE (Energy)', 'XLI': 'XLI (Industrial)',
                    # Commodities & REITs
                    'GLD': 'GLD (Gold)', 'SLV': 'SLV (Silver)', 'VNQ': 'VNQ (Real Estate)',
                    'USO': 'USO (Oil)', 'DBA': 'DBA (Agriculture)'
                }
                
                # Add custom assets if they exist
                if 'custom_assets' in st.session_state:
                    for ticker, name in st.session_state.custom_assets.items():
                        asset_names[ticker] = f"{ticker} ({name})"
                
                for asset, percentage in custom_allocation.items():
                    formatted_name = asset_names.get(asset, asset)
                    custom_formatted[formatted_name] = percentage
                
                results['Your Custom Allocation'] = analyzer.backtest_user_allocation(data, custom_formatted)
            
            # Add S&P 500 benchmark
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
            
            progress_bar.progress(90)
            
            # Create charts
            analysis_results = {
                'results': results,
                'performance_summary': performance_summary,
                'data': data
            }
            
            charts = analyzer.create_interactive_charts(analysis_results)
            
            progress_bar.progress(100)
            
            # Store results
            st.session_state.mpt_analysis_results = analysis_results
            st.session_state.mpt_charts = charts
            st.session_state.mpt_sip_amount = sip_amount
            
            progress_bar.empty()
            st.success("✅ MPT Analysis completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Error running analysis: {str(e)}")
            st.info("Please check your inputs and try again.")

def display_mpt_results(sip_amount: float):
    """Display MPT analysis results."""
    
    analysis_results = st.session_state.mpt_analysis_results
    charts = st.session_state.mpt_charts
    
    st.markdown("---")
    st.header("📊 MPT Analysis Results")
    
    # Performance summary
    SIPAnalysisManager.render_analysis_results(analysis_results, sip_amount)
    
    # Interactive charts
    SIPAnalysisManager.render_interactive_charts(charts)
    
    # Strategy recommendations
    SIPAnalysisManager.render_strategy_recommendations(analysis_results)
    
    # Optimized allocation (only for international portion)
    st.subheader("🎯 Optimal International Allocation")
    
    performance_summary = analysis_results['performance_summary']
    valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                       if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                       and 'Benchmark' not in name and 'Custom' not in name}
    
    if valid_strategies:
        best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
        best_strategy_name = best_strategy[0]
        
        st.success(f"**🏆 Best Strategy: {best_strategy_name}**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected XIRR", f"{best_strategy[1]['XIRR']:.1%}")
        with col2:
            st.metric("Volatility", f"{best_strategy[1]['Volatility']:.1%}")
        with col3:
            st.metric("Sharpe Ratio", f"{best_strategy[1]['Sharpe Ratio']:.2f}")
        
        # Get the actual optimal allocation weights
        analyzer = st.session_state.mpt_analyzer
        data = analysis_results['data']
        
        # Determine the method based on strategy name
        if "Min Variance" in best_strategy_name:
            method = "min_variance"
        elif "Max Sharpe" in best_strategy_name:
            method = "max_sharpe"
        else:
            method = "min_variance"  # Default
        
        optimal_weights = analyzer.get_optimal_allocation_weights(data, method)
        
        if optimal_weights:
            st.markdown("### 💰 **EXACT ALLOCATION RECOMMENDATIONS**")
            st.markdown(f"*Based on ${sip_amount:,.0f}/month for international investments:*")
            
            # Create allocation table
            allocation_data = []
            asset_names = {
                # US Equity ETFs
                'VOO': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'VTI': 'Total US Market ETF',
                'IWM': 'Russell 2000 ETF', 'ARKK': 'Innovation ETF',
                # International ETFs
                'VWO': 'Emerging Markets ETF', 'VEA': 'Developed Markets ETF', 
                'VXUS': 'Total International ETF', 'EFA': 'EAFE ETF', 'VGK': 'European ETF',
                # Bonds & Fixed Income
                'BND': 'Total Bond Market ETF', 'TLT': 'Long-Term Treasury ETF',
                'SHY': 'Short-Term Treasury ETF', 'LQD': 'Investment Grade Corporate Bonds',
                'HYG': 'High Yield Corporate Bonds',
                # Cryptocurrency
                'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
                'ADA-USD': 'Cardano', 'SOL-USD': 'Solana',
                # Sector ETFs
                'XLK': 'Technology Sector ETF', 'XLF': 'Financial Sector ETF', 'XLV': 'Healthcare Sector ETF',
                'XLE': 'Energy Sector ETF', 'XLI': 'Industrial Sector ETF',
                # Commodities & REITs
                'GLD': 'Gold ETF', 'SLV': 'Silver ETF', 'VNQ': 'Real Estate ETF',
                'USO': 'Oil ETF', 'DBA': 'Agriculture ETF'
            }
            
            # Add custom assets if they exist
            if 'custom_assets' in st.session_state:
                for ticker, name in st.session_state.custom_assets.items():
                    asset_names[ticker] = name
            
            for asset, weight in optimal_weights.items():
                if weight > 0.001:  # Only show meaningful allocations
                    monthly_amount = sip_amount * weight
                    allocation_data.append({
                        "Asset": f"{asset} ({asset_names.get(asset, asset)})",
                        "Allocation %": f"{weight*100:.1f}%",
                        "Monthly Amount": f"${monthly_amount:.0f}",
                        "Annual Amount": f"${monthly_amount*12:,.0f}"
                    })
            
            # Sort by allocation percentage (descending)
            allocation_data.sort(key=lambda x: float(x["Allocation %"].replace('%', '')), reverse=True)
            
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, width='stretch', hide_index=True)
            
            # Summary
            st.success(f"✅ **Total Monthly International Investment: ${sip_amount:,.0f}**")
            
            # Show comparison with custom allocation if provided
            if 'Your Custom Allocation' in performance_summary:
                custom_metrics = performance_summary['Your Custom Allocation']
                best_metrics = best_strategy[1]
                
                st.markdown("### 📊 Your Custom vs Optimal Comparison")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Your Custom Allocation:**")
                    st.metric("XIRR", f"{custom_metrics['XIRR']:.1%}")
                    st.metric("Volatility", f"{custom_metrics['Volatility']:.1%}")
                    st.metric("Final Value", f"${custom_metrics['Final Value']:,.0f}")
                
                with col2:
                    st.markdown("**Optimal MPT Allocation:**")
                    st.metric("XIRR", f"{best_metrics['XIRR']:.1%}")
                    st.metric("Volatility", f"{best_metrics['Volatility']:.1%}")
                    st.metric("Final Value", f"${best_metrics['Final Value']:,.0f}")
                
                # Calculate improvement
                improvement = best_metrics['Final Value'] - custom_metrics['Final Value']
                improvement_pct = (improvement / custom_metrics['Final Value']) * 100
                
                if improvement > 0:
                    st.success(f"🚀 **Potential Improvement: ${improvement:,.0f} ({improvement_pct:.1f}% better)**")
                elif improvement < 0:
                    st.info(f"💡 Your custom allocation is actually better by ${abs(improvement):,.0f}!")
                else:
                    st.info("🤝 Both strategies perform similarly!")
        
        else:
            st.warning("⚠️ Could not calculate specific allocation weights. Using general recommendations.")
        
        # Show implementation guide
        st.markdown("### 📋 Implementation Guide")
        
        if "Min Variance" in best_strategy_name:
            st.info("""
            **Min Variance Strategy Focus:**
            - Emphasizes risk reduction while maintaining returns
            - Balanced allocation across selected assets
            - Lower volatility, steady growth approach
            """)
        elif "Max Sharpe" in best_strategy_name:
            st.info("""
            **Max Sharpe Strategy Focus:**
            - Maximizes risk-adjusted returns
            - May have higher concentration in top performers
            - Optimal balance of risk and return
            """)
        else:
            st.info("""
            **Equal Weight Strategy:**
            - Simple, balanced approach
            - Equal exposure to all selected assets
            - Easy to implement and maintain
            """)
        
        # Rebalancing guidance
        if "3M" in best_strategy_name:
            st.warning("🔄 **Recommended**: Rebalance every 3 months for optimal performance")
        elif "6M" in best_strategy_name:
            st.success("🔄 **Recommended**: Rebalance every 6 months (balanced approach)")
        elif "12M" in best_strategy_name:
            st.info("🔄 **Recommended**: Rebalance annually (low maintenance)")
        else:
            st.info("🔄 **Static allocation**: No rebalancing needed")
    
    # Export functionality
    if st.button("📥 Export Comprehensive Analysis (Excel)"):
        # Get financial data from session state or calculate from current inputs
        exchange_rates = st.session_state.get('exchange_rates', get_cached_rates())
        
        # Create financial summary (you might want to store these in session state)
        financial_summary = create_financial_summary_from_session(
            total_income=st.session_state.get('total_income', 0),
            pkr_expenses_usd=st.session_state.get('pkr_expenses_usd', 0),
            international_amount=st.session_state.get('international_amount', 0),
            local_amount=st.session_state.get('local_amount', 0),
            cash_amount=st.session_state.get('cash_amount', 0),
            exchange_rates=exchange_rates
        )
        
        # Get optimal allocation if available
        optimal_allocation = None
        if st.session_state.get('mpt_analyzer') and analysis_results:
            analyzer = st.session_state.mpt_analyzer
            data = analysis_results['data']
            
            # Find best strategy
            performance_summary = analysis_results['performance_summary']
            valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                               if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                               and 'Benchmark' not in name and 'Custom' not in name}
            
            if valid_strategies:
                best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
                best_strategy_name = best_strategy[0]
                
                # Determine method
                if "Min Variance" in best_strategy_name:
                    method = "min_variance"
                elif "Max Sharpe" in best_strategy_name:
                    method = "max_sharpe"
                else:
                    method = "min_variance"
                
                optimal_allocation = analyzer.get_optimal_allocation_weights(data, method)
        
        # Create comprehensive report
        excel_manager = ExcelExportManager()
        file_data, filename, mime_type = excel_manager.create_comprehensive_report(
            analysis_results=analysis_results,
            financial_summary=financial_summary,
            optimal_allocation=optimal_allocation,
            sip_amount=sip_amount
        )
        
        # Determine button label based on file type
        if filename.endswith('.xlsx'):
            button_label = "📊 Download Comprehensive Report (Excel)"
            success_message = "✅ Excel report created with 4 detailed sheets!"
        else:
            button_label = "📊 Download Comprehensive Report (CSV)"
            success_message = "✅ CSV report created (Excel libraries not available)"
        
        st.info(success_message)
        
        # Download button
        st.download_button(
            label=button_label,
            data=file_data,
            file_name=filename,
            mime=mime_type
        )

if __name__ == "__main__":
    main()
