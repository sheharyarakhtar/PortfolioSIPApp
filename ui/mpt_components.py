"""
MPT-focused UI components for the portfolio optimization interface.
Clean, professional components focused on Modern Portfolio Theory.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from typing import Dict, List, Optional
from utils.excel_export import ExcelExportManager

class MPTInterface:
    """Professional MPT interface components."""
    
    @staticmethod
    def render_performance_summary(analysis_results: Dict):
        """Render performance summary with professional styling."""
        
        performance_summary = analysis_results['performance_summary']
        
        st.subheader("📊 Performance Summary")
        
        # Create performance DataFrame
        perf_data = []
        for strategy, metrics in performance_summary.items():
            if not pd.isna(metrics['XIRR']):  # Only include valid strategies
                perf_data.append({
                    'Strategy': strategy,
                    'Annual Return (XIRR)': f"{metrics['XIRR']:.2%}",
                    'Volatility': f"{metrics['Volatility']:.2%}",
                    'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.2f}",
                    'Max Drawdown': f"{metrics['Max Drawdown']:.2%}",
                    'Final Value': f"${metrics['Final Value']:,.0f}"
                })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # Sort by XIRR for better display
            perf_df['XIRR_numeric'] = [metrics['XIRR'] for strategy, metrics in performance_summary.items() 
                                      if not pd.isna(metrics['XIRR'])]
            perf_df = perf_df.sort_values('XIRR_numeric', ascending=False).drop('XIRR_numeric', axis=1)
            
            st.dataframe(perf_df, hide_index=True, use_container_width=True)
            
            # Key insights
            MPTInterface._render_key_insights(performance_summary)
        else:
            st.warning("No valid performance data available.")
    
    @staticmethod
    def _render_key_insights(performance_summary: Dict):
        """Render key performance insights."""
        
        valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                           if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                           and 'Benchmark' not in name}
        
        if valid_strategies:
            st.subheader("🎯 Key Insights")
            
            best_return = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['Sharpe Ratio'])
            lowest_risk = min(valid_strategies.items(), key=lambda x: x[1]['Volatility'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "🏆 Best Return",
                    best_return[0].replace(" (6 Months)", "").replace(" (12 Months)", ""),
                    f"{best_return[1]['XIRR']:.2%} annually"
                )
            
            with col2:
                st.metric(
                    "⚖️ Best Risk-Adjusted",
                    best_sharpe[0].replace(" (6 Months)", "").replace(" (12 Months)", ""),
                    f"Sharpe: {best_sharpe[1]['Sharpe Ratio']:.2f}"
                )
            
            with col3:
                st.metric(
                    "🛡️ Lowest Risk",
                    lowest_risk[0].replace(" (6 Months)", "").replace(" (12 Months)", ""),
                    f"{lowest_risk[1]['Volatility']:.2%} volatility"
                )
    
    @staticmethod
    def render_interactive_charts(charts: Dict[str, go.Figure]):
        """Render interactive charts with professional styling."""
        
        st.subheader("📈 Performance Analysis")
        
        # Portfolio growth chart
        st.markdown("#### Portfolio Growth Over Time")
        st.plotly_chart(charts['growth'], use_container_width=True)
        st.caption("💡 The gap between strategy lines and contributions shows investment gains")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Annual Returns Comparison")
            st.plotly_chart(charts['returns'], use_container_width=True)
        
        with col2:
            st.markdown("#### Risk vs Return Analysis")
            st.plotly_chart(charts['risk_return'], use_container_width=True)
    
    @staticmethod
    def render_optimal_allocation(analysis_results: Dict):
        """Render optimal allocation recommendations with detailed breakdown."""
        
        st.subheader("🎯 Optimal Portfolio Allocation")
        
        performance_summary = analysis_results['performance_summary']
        config = analysis_results['config']
        
        # Find best strategy (excluding benchmark initially)
        valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                           if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                           and 'Benchmark' not in name}
        
        # Get S&P 500 benchmark performance
        sp500_metrics = None
        for name, metrics in performance_summary.items():
            if 'Benchmark' in name and not pd.isna(metrics['XIRR']):
                sp500_metrics = metrics
                break
        
        # Compare best strategy with S&P 500
        if valid_strategies and sp500_metrics:
            best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            best_strategy_name = best_strategy[0]
            best_metrics = best_strategy[1]
            
            # Check if S&P 500 beats the best strategy
            if sp500_metrics['XIRR'] > best_metrics['XIRR']:
                st.warning("🚨 **Important Recommendation:**")
                st.error(f"**S&P 500 outperforms all optimized strategies!**")
                st.info(f"**S&P 500**: {sp500_metrics['XIRR']:.2%} XIRR vs **Best Strategy**: {best_metrics['XIRR']:.2%} XIRR")
                st.success("💡 **Recommendation: Just invest in S&P 500 (VOO) for better returns with lower complexity!**")
                
                # Override recommendation with S&P 500
                best_strategy_name = "S&P 500 (Simple & Better)"
                best_metrics = sp500_metrics
                is_sp500_better = True
            else:
                is_sp500_better = False
        elif valid_strategies:
            best_strategy = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            best_strategy_name = best_strategy[0]
            best_metrics = best_strategy[1]
            is_sp500_better = False
        else:
            st.warning("No valid strategies found for allocation recommendations.")
            return
        
        # Shorten strategy name for better display
        display_name = best_strategy_name.replace(' (6 Months)', ' (6M)').replace(' (12 Months)', ' (12M)').replace(' (3 Months)', ' (3M)')
        display_name = display_name.replace('Max Sharpe', 'MaxSharpe').replace('Min Variance', 'MinVar')
        
        st.success(f"**🏆 Recommended Strategy: {display_name}**")
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Expected Annual Return", f"{best_metrics['XIRR']:.2%}")
        
        with col2:
            st.metric("Volatility", f"{best_metrics['Volatility']:.2%}")
        
        with col3:
            st.metric("Sharpe Ratio", f"{best_metrics['Sharpe Ratio']:.2f}")
        
        with col4:
            st.metric("Max Drawdown", f"{best_metrics['Max Drawdown']:.2%}")
        
        # Show allocation based on whether S&P 500 is better
        if is_sp500_better:
            MPTInterface._render_sp500_allocation(config['sip_amount'])
        else:
            # Get and display detailed allocation weights
            MPTInterface._render_detailed_allocation(analysis_results, best_strategy_name)
        
        # Implementation guidance
        MPTInterface._render_implementation_guide(best_strategy_name)
    
    @staticmethod
    def _render_sp500_allocation(monthly_amount: float):
        """Render S&P 500 allocation recommendation."""
        
        st.markdown("### 💰 Optimal Allocation: S&P 500 Only")
        
        allocation_data = [{
            "Asset": "VOO",
            "Name": "S&P 500 ETF (Vanguard)",
            "Allocation %": "100.0%",
            "Monthly Amount": f"${monthly_amount:.0f}",
            "Annual Amount": f"${monthly_amount * 12:,.0f}"
        }]
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, hide_index=True, use_container_width=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Monthly Investment", f"${monthly_amount:,}")
        
        with col2:
            st.metric("Total Annual Investment", f"${monthly_amount * 12:,}")
        
        with col3:
            st.metric("Portfolio Simplicity", "1 asset")
        
        # Implementation instructions
        st.markdown("### 📋 Implementation Instructions")
        
        st.success("**Simple Monthly Investment Plan:**")
        st.write(f"• **VOO** (S&P 500 ETF): ${monthly_amount:.0f} (100%)")
        
        st.info(f"💡 **Total**: ${monthly_amount:,}/month in a single, diversified ETF")
        
        st.markdown("### 🎯 Why S&P 500?")
        st.info("""
        **Advantages of S&P 500 investment:**
        - ✅ **Better returns** than complex optimization
        - ✅ **Lower costs** (single ETF, no rebalancing)
        - ✅ **Instant diversification** (500 companies)
        - ✅ **Simple to manage** (set and forget)
        - ✅ **Tax efficient** (no frequent trading)
        - ✅ **Lower fees** (VOO expense ratio: 0.03%)
        """)
    
    @staticmethod
    def _render_detailed_allocation(analysis_results: Dict, strategy_name: str):
        """Render detailed allocation breakdown with exact amounts."""
        
        st.markdown("### 💰 Detailed Asset Allocation")
        
        config = analysis_results['config']
        monthly_amount = config['sip_amount']
        
        # Try to get optimal weights from the analyzer
        try:
            # Import here to avoid circular imports
            from models.sip_strategy import SIPStrategyAnalyzer
            
            # Create analyzer instance
            analyzer = SIPStrategyAnalyzer(config)
            analyzer.available_tickers = config['tickers']
            
            # Get data for optimization
            data = analysis_results.get('data')
            
            if data is not None:
                # Determine optimization method from strategy name
                if "Max Sharpe" in strategy_name:
                    method = "max_sharpe"
                elif "Min Variance" in strategy_name:
                    method = "min_variance"
                elif "Equal Weight" in strategy_name:
                    method = "equal_weight"
                else:
                    # Default to max_sharpe if method unclear
                    method = "max_sharpe"
                
                if method == "equal_weight":
                    # Equal weight allocation
                    num_assets = len(config['tickers'])
                    optimal_weights = {ticker: 1.0/num_assets for ticker in config['tickers']}
                else:
                    # Get optimal weights from MPT optimization
                    optimal_weights = analyzer.get_optimal_allocation_weights(data, method)
                
                if optimal_weights:
                    MPTInterface._display_allocation_table(optimal_weights, monthly_amount, config['tickers'])
                else:
                    st.warning("⚠️ Could not calculate optimal weights for this combination of assets and time period.")
                    st.info("💡 Try selecting different assets or a longer time period for better optimization results.")
                    MPTInterface._display_fallback_allocation(config['tickers'], monthly_amount)
            else:
                MPTInterface._display_fallback_allocation(config['tickers'], monthly_amount)
                
        except Exception as e:
            st.warning(f"Could not calculate optimal weights: {str(e)}")
            MPTInterface._display_fallback_allocation(config['tickers'], monthly_amount)
    
    @staticmethod
    def _display_allocation_table(optimal_weights: Dict[str, float], monthly_amount: float, tickers: List[str]):
        """Display detailed allocation table with exact amounts."""
        
        # Asset name mapping for better display
        asset_names = {
            'VOO': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'VTI': 'Total US Market ETF',
            'IWM': 'Russell 2000 ETF', 'ARKK': 'Innovation ETF',
            'VWO': 'Emerging Markets ETF', 'VEA': 'Developed Markets ETF', 
            'VXUS': 'Total International ETF', 'EFA': 'EAFE ETF', 'VGK': 'European ETF',
            'BND': 'Total Bond Market ETF', 'TLT': 'Long-Term Treasury ETF',
            'SHY': 'Short-Term Treasury ETF', 'LQD': 'Investment Grade Corporate',
            'HYG': 'High Yield Corporate',
            'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
            'ADA-USD': 'Cardano', 'SOL-USD': 'Solana',
            'XLK': 'Technology Sector ETF', 'XLF': 'Financial Sector ETF', 
            'XLV': 'Healthcare Sector ETF', 'XLE': 'Energy Sector ETF', 'XLI': 'Industrial Sector ETF',
            'GLD': 'Gold ETF', 'SLV': 'Silver ETF', 'VNQ': 'Real Estate ETF',
            'USO': 'Oil ETF', 'DBA': 'Agriculture ETF'
        }
        
        # Create allocation data
        allocation_data = []
        total_allocated = 0
        
        for ticker in tickers:
            weight = optimal_weights.get(ticker, 0)
            if weight > 0.001:  # Only show meaningful allocations (>0.1%)
                monthly_allocation = monthly_amount * weight
                annual_allocation = monthly_allocation * 12
                
                allocation_data.append({
                    "Asset": ticker,
                    "Name": asset_names.get(ticker, "Unknown Asset"),
                    "Allocation %": f"{weight*100:.1f}%",
                    "Monthly Amount": f"${monthly_allocation:.0f}",
                    "Annual Amount": f"${annual_allocation:,.0f}"
                })
                total_allocated += weight
        
        # Sort by allocation percentage (descending)
        allocation_data.sort(key=lambda x: float(x["Allocation %"].replace('%', '')), reverse=True)
        
        if allocation_data:
            # Display allocation table
            allocation_df = pd.DataFrame(allocation_data)
            st.dataframe(allocation_df, hide_index=True, use_container_width=True)
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Monthly Investment", f"${monthly_amount:,}")
            
            with col2:
                st.metric("Total Annual Investment", f"${monthly_amount * 12:,}")
            
            with col3:
                st.metric("Portfolio Diversification", f"{len(allocation_data)} assets")
            
            # Implementation instructions
            st.markdown("### 📋 Implementation Instructions")
            
            st.success("**Monthly Investment Plan:**")
            for item in allocation_data:
                st.write(f"• **{item['Asset']}** ({item['Name']}): {item['Monthly Amount']} ({item['Allocation %']})")
            
            st.info(f"💡 **Total**: ${monthly_amount:,}/month across {len(allocation_data)} assets")
            
        else:
            st.warning("No significant allocations found. Using equal weight fallback.")
            MPTInterface._display_fallback_allocation(tickers, monthly_amount)
    
    @staticmethod
    def _display_fallback_allocation(tickers: List[str], monthly_amount: float):
        """Display equal weight allocation as fallback."""
        
        st.markdown("### 💰 Equal Weight Allocation")
        
        num_assets = len(tickers)
        allocation_per_asset = monthly_amount / num_assets
        
        allocation_data = []
        for ticker in tickers:
            allocation_data.append({
                "Asset": ticker,
                "Monthly Amount": f"${allocation_per_asset:.0f}",
                "Allocation %": f"{100/num_assets:.1f}%",
                "Annual Amount": f"${allocation_per_asset * 12:,.0f}"
            })
        
        allocation_df = pd.DataFrame(allocation_data)
        st.dataframe(allocation_df, hide_index=True, use_container_width=True)
        
        st.info(f"💡 **Equal Weight Strategy**: ${allocation_per_asset:.0f} per asset, {100/num_assets:.1f}% each")
    
    @staticmethod
    def _render_implementation_guide(strategy_name: str):
        """Render implementation guidance based on strategy."""
        
        st.markdown("### 📋 Implementation Guide")
        
        if "Max Sharpe" in strategy_name:
            st.info("""
            **Max Sharpe Ratio Strategy:**
            - Focuses on maximizing risk-adjusted returns
            - May concentrate in higher-performing assets
            - Suitable for growth-oriented investors
            - Rebalance according to selected frequency
            """)
        elif "Min Variance" in strategy_name:
            st.info("""
            **Minimum Variance Strategy:**
            - Emphasizes risk reduction
            - More balanced allocation across assets
            - Suitable for conservative investors
            - Lower volatility with steady growth
            """)
        elif "Equal Weight" in strategy_name:
            st.info("""
            **Equal Weight Strategy:**
            - Simple, balanced approach
            - Equal allocation to all selected assets
            - Easy to implement and maintain
            - Good baseline strategy
            """)
        
        # Rebalancing guidance
        if "3 Months" in strategy_name:
            st.warning("🔄 **Rebalancing**: Every 3 months (higher maintenance, potentially better performance)")
        elif "6 Months" in strategy_name:
            st.success("🔄 **Rebalancing**: Every 6 months (balanced approach)")
        elif "12 Months" in strategy_name:
            st.info("🔄 **Rebalancing**: Annually (lower maintenance)")
        else:
            st.info("🔄 **Static allocation**: No rebalancing required")
        
        # Rebalancing strategy explanation
        if "Conservative" in strategy_name:
            st.info("""
            **🟢 Conservative Approach (Keep Holdings + Add New):**
            - Keep existing holdings unchanged
            - Only optimize allocation for new monthly contributions
            - Lower transaction costs and tax implications
            - More tax-efficient for taxable accounts
            """)
        elif "Aggressive" in strategy_name:
            st.warning("""
            **🔴 Aggressive Approach (Sell & Reinvest All):**
            - Sell all holdings and reinvest optimally each rebalancing period
            - Achieves perfect optimal allocation every time
            - Higher transaction costs and potential tax implications
            - Best for tax-advantaged accounts (401k, IRA)
            """)
        else:
            st.info("""
            **📊 Mixed Strategy Analysis:**
            - Both conservative and aggressive approaches were tested
            - Choose based on your account type and tax situation
            - Conservative for taxable accounts, Aggressive for retirement accounts
            """)
    
    @staticmethod
    def render_portfolio_composition_analysis(analysis_results: Dict):
        """Render interactive portfolio composition analysis table."""
        
        st.subheader("📊 Portfolio Composition Analysis")
        st.markdown("See how your portfolio would have grown if you started SIP at different time periods")
        
        config = analysis_results['config']
        performance_summary = analysis_results['performance_summary']
        
        # Strategy selector
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Get available strategies (exclude benchmark)
            available_strategies = [name for name in performance_summary.keys() 
                                  if 'Benchmark' not in name and not pd.isna(performance_summary[name]['XIRR'])]
            
            selected_strategy = st.selectbox(
                "Select Strategy",
                options=available_strategies,
                index=0 if available_strategies else 0,
                help="Choose which strategy to analyze"
            )
        
        with col2:
            # Time period selector
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
                index=2,  # Default to 2 years
                help="How long ago you would have started SIP"
            )
        
        with col3:
            # Rebalancing approach (if both are available)
            rebalancing_options = []
            if any("Conservative" in strategy for strategy in available_strategies):
                rebalancing_options.append("Conservative")
            if any("Aggressive" in strategy for strategy in available_strategies):
                rebalancing_options.append("Aggressive")
            
            if len(rebalancing_options) > 1:
                selected_rebalancing = st.selectbox(
                    "Rebalancing Approach",
                    options=rebalancing_options,
                    help="Conservative: Keep + Add New, Aggressive: Sell & Reinvest"
                )
                # Update selected strategy to match rebalancing choice
                for strategy in available_strategies:
                    if selected_rebalancing in strategy and selected_strategy.split(" - ")[0] in strategy:
                        selected_strategy = strategy
                        break
        
        if st.button("🔍 Analyze Portfolio Composition", type="primary"):
            MPTInterface._generate_composition_analysis(
                analysis_results, selected_strategy, time_periods[selected_period]
            )
        
        # Display results if available
        if st.session_state.get('composition_analysis'):
            MPTInterface._display_composition_results()
    
    @staticmethod
    def _generate_composition_analysis(analysis_results: Dict, strategy_name: str, years_back: float):
        """Generate detailed portfolio composition analysis."""
        
        try:
            from models.sip_strategy import SIPStrategyAnalyzer
            from datetime import datetime, timedelta
            
            config = analysis_results['config']
            
            # Calculate analysis period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=int(years_back * 365))
            
            # Create analyzer
            analyzer = SIPStrategyAnalyzer(config)
            analyzer.available_tickers = config['tickers']
            
            # Validate tickers for the specific period
            valid_tickers, invalid_tickers = analyzer.validate_tickers(config['tickers'])
            
            if invalid_tickers:
                st.warning(f"⚠️ {len(invalid_tickers)} assets unavailable for {years_back} year period: {', '.join(invalid_tickers)}")
                st.info(f"✅ Continuing analysis with {len(valid_tickers)} available assets: {', '.join(valid_tickers)}")
            
            if len(valid_tickers) == 0:
                st.error("❌ No assets have data for the selected time period.")
                st.info("💡 Try selecting a shorter time period or different assets.")
                return
            elif len(valid_tickers) == 1:
                st.info(f"ℹ️ Only {valid_tickers[0]} has data for this period. Showing single-asset analysis.")
            
            # Update config with valid tickers
            config = config.copy()
            config['tickers'] = valid_tickers
            analyzer.available_tickers = valid_tickers
            
            # Get data for the specific period
            data = analyzer.download_data(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            
            # Determine strategy parameters
            if "Max Sharpe" in strategy_name:
                method = "max_sharpe"
            elif "Min Variance" in strategy_name:
                method = "min_variance"
            else:
                method = "equal_weight"
            
            full_rebalancing = "Aggressive" in strategy_name
            
            # Extract rebalancing frequency
            rebalance_months = 6  # Default
            if "3 Months" in strategy_name:
                rebalance_months = 3
            elif "12 Months" in strategy_name:
                rebalance_months = 12
            
            # Build strategy config
            strategy_config = {
                'name': strategy_name,
                'rebalance_months': rebalance_months,
                'method': method,
                'full_rebalancing': full_rebalancing
            }
            
            # Run backtest
            if method == 'equal_weight':
                result = analyzer.backtest_equal_weight(data)
            else:
                result = analyzer.backtest_strategy(data, strategy_config)
            
            # Calculate individual asset performance
            asset_performance = MPTInterface._calculate_asset_performance(data, config['tickers'])
            
            # Calculate final portfolio composition
            final_composition = MPTInterface._calculate_final_composition(
                result, data, config, analyzer, method
            )
            
            # Store results
            st.session_state.composition_analysis = {
                'strategy_name': strategy_name,
                'period': f"{years_back} years",
                'final_composition': final_composition,
                'asset_performance': asset_performance,
                'portfolio_metrics': analyzer.calculate_performance_metrics(
                    result['portfolio_values'], 
                    result['monthly_contributions']
                ),
                'total_invested': result['monthly_contributions'].sum(),
                'final_value': result['portfolio_values'].iloc[-1] if len(result['portfolio_values']) > 0 else 0
            }
            
            st.success(f"✅ Analysis completed for {strategy_name} over {years_back} years!")
            
        except Exception as e:
            st.error(f"❌ Error generating analysis: {str(e)}")
            st.info("This might be due to insufficient data for the selected period.")
    
    @staticmethod
    def _calculate_asset_performance(data: pd.DataFrame, tickers: List[str]) -> Dict:
        """Calculate individual asset performance metrics."""
        
        asset_performance = {}
        
        for ticker in tickers:
            if ticker in data.columns:
                prices = data[ticker].dropna()
                if len(prices) > 1:
                    # Calculate returns
                    returns = prices.pct_change().dropna()
                    
                    # CAGR calculation
                    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
                    years = len(prices) / 252  # Approximate trading days per year
                    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
                    
                    # Volatility (annualized)
                    volatility = returns.std() * np.sqrt(252)
                    
                    # Sharpe ratio (assuming 4% risk-free rate)
                    sharpe = (cagr - 0.04) / volatility if volatility > 0 else 0
                    
                    # Max drawdown
                    rolling_max = prices.expanding().max()
                    drawdown = (prices - rolling_max) / rolling_max
                    max_drawdown = drawdown.min()
                    
                    asset_performance[ticker] = {
                        'start_price': prices.iloc[0],
                        'end_price': prices.iloc[-1],
                        'total_return': total_return,
                        'cagr': cagr,
                        'volatility': volatility,
                        'sharpe_ratio': sharpe,
                        'max_drawdown': max_drawdown
                    }
        
        return asset_performance
    
    @staticmethod
    def _calculate_final_composition(result: Dict, data: pd.DataFrame, config: Dict, 
                                   analyzer, method: str) -> Dict:
        """Calculate final portfolio composition in terms of shares and values."""
        
        final_composition = {}
        
        # Get the final date and prices
        final_date = data.index[-1]
        monthly_contributions = result['monthly_contributions']
        
        # Simulate the SIP to get final share counts
        # This is a simplified calculation - in practice, you'd track shares through the backtest
        
        if method == "equal_weight":
            # Equal weight: distribute equally
            num_assets = len(config['tickers'])
            weight_per_asset = 1.0 / num_assets
            
            for ticker in config['tickers']:
                if ticker in data.columns:
                    ticker_data = data[ticker].dropna()
                    if len(ticker_data) > 0:
                        # Calculate total investment in this asset
                        total_invested_in_asset = monthly_contributions.sum() * weight_per_asset
                        
                        # Calculate average price (more accurate)
                        avg_price = ticker_data.mean()
                        shares_owned = total_invested_in_asset / avg_price if avg_price > 0 else 0
                        
                        final_price = ticker_data.iloc[-1]
                        final_value = shares_owned * final_price
                        
                        final_composition[ticker] = {
                            'shares_owned': shares_owned,
                            'avg_price': avg_price,
                            'final_price': final_price,
                            'final_value': final_value,
                            'total_invested': total_invested_in_asset,
                            'gain_loss': final_value - total_invested_in_asset,
                            'return_pct': (final_value / total_invested_in_asset - 1) if total_invested_in_asset > 0 else 0
                        }
        else:
            # For optimized strategies, use the optimal weights
            optimal_weights = analyzer.get_optimal_allocation_weights(data, method)
            
            if optimal_weights:
                for ticker in config['tickers']:
                    if ticker in optimal_weights and ticker in data.columns:
                        weight = optimal_weights[ticker]
                        if weight > 0:  # Only process assets with positive weights
                            ticker_data = data[ticker].dropna()
                            if len(ticker_data) > 0:
                                total_invested_in_asset = monthly_contributions.sum() * weight
                                
                                avg_price = ticker_data.mean()
                                shares_owned = total_invested_in_asset / avg_price if avg_price > 0 else 0
                                
                                final_price = ticker_data.iloc[-1]
                                final_value = shares_owned * final_price
                                
                                final_composition[ticker] = {
                                    'shares_owned': shares_owned,
                                    'avg_price': avg_price,
                                    'final_price': final_price,
                                    'final_value': final_value,
                                    'total_invested': total_invested_in_asset,
                                    'gain_loss': final_value - total_invested_in_asset,
                                    'return_pct': (final_value / total_invested_in_asset - 1) if total_invested_in_asset > 0 else 0,
                                    'weight': weight
                                }
            else:
                # If optimization fails, fall back to equal weight (silently)
                num_assets = len(config['tickers'])
                weight_per_asset = 1.0 / num_assets
                
                for ticker in config['tickers']:
                    if ticker in data.columns:
                        ticker_data = data[ticker].dropna()
                        if len(ticker_data) > 0:
                            total_invested_in_asset = monthly_contributions.sum() * weight_per_asset
                            
                            avg_price = ticker_data.mean()
                            shares_owned = total_invested_in_asset / avg_price if avg_price > 0 else 0
                            
                            final_price = ticker_data.iloc[-1]
                            final_value = shares_owned * final_price
                            
                            final_composition[ticker] = {
                                'shares_owned': shares_owned,
                                'avg_price': avg_price,
                                'final_price': final_price,
                                'final_value': final_value,
                                'total_invested': total_invested_in_asset,
                                'gain_loss': final_value - total_invested_in_asset,
                                'return_pct': (final_value / total_invested_in_asset - 1) if total_invested_in_asset > 0 else 0,
                                'weight': weight_per_asset
                            }
        
        return final_composition
    
    @staticmethod
    def _display_composition_results():
        """Display the portfolio composition analysis results."""
        
        analysis = st.session_state.composition_analysis
        
        st.markdown("### 📊 Portfolio Composition Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Shorten strategy name to prevent overflow
            strategy_short = analysis['strategy_name'].replace(' (6 Months)', ' (6M)').replace(' (12 Months)', ' (12M)').replace(' (3 Months)', ' (3M)')
            strategy_short = strategy_short.replace('Max Sharpe', 'MaxSharpe').replace('Min Variance', 'MinVar')
            strategy_short = strategy_short.split(' - ')[0]  # Remove Conservative/Aggressive part
            st.metric("Strategy", strategy_short)
        
        with col2:
            st.metric("Period", analysis['period'])
        
        with col3:
            st.metric("Total Invested", f"${analysis['total_invested']:,.0f}")
        
        with col4:
            st.metric("Final Value", f"${analysis['final_value']:,.0f}")
        
        # Portfolio composition table
        st.markdown("#### 🎯 Final Portfolio Composition")
        
        composition_data = []
        asset_names = {
            'VOO': 'S&P 500 ETF', 'QQQ': 'Nasdaq 100 ETF', 'VTI': 'Total US Market ETF',
            'IWM': 'Russell 2000 ETF', 'ARKK': 'Innovation ETF',
            'VWO': 'Emerging Markets ETF', 'VEA': 'Developed Markets ETF', 
            'VXUS': 'Total International ETF', 'EFA': 'EAFE ETF', 'VGK': 'European ETF',
            'BND': 'Total Bond Market ETF', 'TLT': 'Long-Term Treasury ETF',
            'SHY': 'Short-Term Treasury ETF', 'LQD': 'Investment Grade Corporate',
            'HYG': 'High Yield Corporate',
            'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
            'ADA-USD': 'Cardano', 'SOL-USD': 'Solana',
            'XLK': 'Technology Sector ETF', 'XLF': 'Financial Sector ETF', 
            'XLV': 'Healthcare Sector ETF', 'XLE': 'Energy Sector ETF', 'XLI': 'Industrial Sector ETF',
            'GLD': 'Gold ETF', 'SLV': 'Silver ETF', 'VNQ': 'Real Estate ETF',
            'USO': 'Oil ETF', 'DBA': 'Agriculture ETF'
        }
        
        # Debug: Check if we have composition data
        if not analysis['final_composition']:
            st.error("❌ No portfolio composition data available. This might be due to:")
            st.info("• Insufficient historical data for the selected period")
            st.info("• Assets not available in the selected timeframe")
            st.info("• Try selecting a shorter time period or different assets")
            return
        
        for ticker, comp in analysis['final_composition'].items():
            asset_perf = analysis['asset_performance'].get(ticker, {})
            
            # Ensure all values are valid numbers
            shares_owned = comp.get('shares_owned', 0)
            avg_price = comp.get('avg_price', 0)
            final_price = comp.get('final_price', 0)
            total_invested = comp.get('total_invested', 0)
            final_value = comp.get('final_value', 0)
            gain_loss = comp.get('gain_loss', 0)
            return_pct = comp.get('return_pct', 0)
            
            if shares_owned > 0:  # Only show assets with actual holdings
                composition_data.append({
                    "Asset": ticker,
                    "Name": asset_names.get(ticker, "Unknown"),
                    "Shares Owned": f"{shares_owned:.2f}",
                    "Avg Price": f"${avg_price:.2f}",
                    "Current Price": f"${final_price:.2f}",
                    "Total Invested": f"${total_invested:,.0f}",
                    "Current Value": f"${final_value:,.0f}",
                    "Gain/Loss": f"${gain_loss:,.0f}",
                    "Return %": f"{return_pct:.1%}",
                    "Asset CAGR": f"{asset_perf.get('cagr', 0):.1%}",
                    "Asset Volatility": f"{asset_perf.get('volatility', 0):.1%}",
                    "Asset Sharpe": f"{asset_perf.get('sharpe_ratio', 0):.2f}"
                })
        
        if composition_data:
            # Sort by current value (descending)
            composition_data.sort(key=lambda x: float(x["Current Value"].replace('$', '').replace(',', '')), reverse=True)
            
            composition_df = pd.DataFrame(composition_data)
            st.dataframe(composition_df, hide_index=True, use_container_width=True)
        else:
            st.warning("⚠️ No portfolio holdings found. This could be due to:")
            st.info("• Very short time period with insufficient SIP contributions")
            st.info("• Assets with zero or negative weights in optimization")
            st.info("• Data availability issues for the selected period")
        
        # Individual asset performance
        st.markdown("#### 📈 Individual Asset Performance")
        
        asset_perf_data = []
        for ticker, perf in analysis['asset_performance'].items():
            asset_perf_data.append({
                "Asset": ticker,
                "Name": asset_names.get(ticker, "Unknown"),
                "Start Price": f"${perf['start_price']:.2f}",
                "End Price": f"${perf['end_price']:.2f}",
                "Total Return": f"{perf['total_return']:.1%}",
                "CAGR": f"{perf['cagr']:.1%}",
                "Volatility": f"{perf['volatility']:.1%}",
                "Sharpe Ratio": f"{perf['sharpe_ratio']:.2f}",
                "Max Drawdown": f"{perf['max_drawdown']:.1%}"
            })
        
        # Sort by CAGR (descending)
        asset_perf_data.sort(key=lambda x: float(x["CAGR"].replace('%', '')), reverse=True)
        
        asset_perf_df = pd.DataFrame(asset_perf_data)
        st.dataframe(asset_perf_df, hide_index=True, use_container_width=True)
        
        # Portfolio summary
        portfolio_metrics = analysis['portfolio_metrics']
        
        st.markdown("#### 🏆 Portfolio Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio CAGR", f"{portfolio_metrics['XIRR']:.2%}")
        
        with col2:
            st.metric("Portfolio Volatility", f"{portfolio_metrics['Volatility']:.2%}")
        
        with col3:
            st.metric("Portfolio Sharpe", f"{portfolio_metrics['Sharpe Ratio']:.2f}")
        
        with col4:
            total_gain = analysis['final_value'] - analysis['total_invested']
            st.metric("Total Gain/Loss", f"${total_gain:,.0f}")
    
    @staticmethod
    def render_export_options(analysis_results: Dict):
        """Render export options for analysis results."""
        
        st.subheader("📥 Export Comprehensive Report")
        
        st.markdown("""
        **📋 Excel Report includes:**
        - Executive Summary with key insights
        - Strategy Performance Comparison
        - Risk Analysis & Metrics
        - Portfolio Composition Analysis
        - Asset Correlation Matrix
        - Portfolio Growth Timeline
        - Monthly Returns Analysis
        """)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📋 Download Comprehensive Excel Report", use_container_width=True, type="primary"):
                MPTInterface._export_enhanced_excel(analysis_results)
    
    @staticmethod
    def _export_enhanced_excel(analysis_results: Dict):
        """Export comprehensive Excel analysis report."""
        
        try:
            excel_manager = ExcelExportManager()
            
            # Get SIP amount from config
            config = analysis_results.get('config', {})
            sip_amount = config.get('sip_amount', 0)
            
            file_data, filename, mime_type = excel_manager.create_comprehensive_report(
                analysis_results=analysis_results,
                sip_amount=sip_amount
            )
            
            st.download_button(
                label="📋 Download Comprehensive Excel Report",
                data=file_data,
                file_name=filename,
                mime=mime_type,
                help="Download detailed portfolio analysis with multiple sheets",
                use_container_width=True
            )
            
            st.success("✅ Excel report generated successfully! The report contains 7 comprehensive analysis sheets.")
            
        except Exception as e:
            st.error(f"Error creating Excel export: {str(e)}")
            
            # Show error details in expander for debugging
            with st.expander("🔍 Error Details"):
                import traceback
                st.code(traceback.format_exc())
