"""
SIP Strategy UI components for the Streamlit interface.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from models.sip_strategy import SIPStrategyAnalyzer

class SIPAnalysisManager:
    """Manages SIP analysis interface components."""
    
    @staticmethod
    def render_analysis_results(analysis_results: Dict, sip_amount: int):
        """Render comprehensive analysis results"""
        
        performance_summary = analysis_results['performance_summary']
        
        # Performance Summary Table
        st.subheader("📊 Performance Summary")
        
        # Add filtering controls
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            rebalancing_filter = st.selectbox(
                "🔄 Rebalancing Approach:",
                options=["All Strategies", "New Contrib Only", "Full Rebal Only"],
                index=0,
                help="Filter strategies by rebalancing approach"
            )
        
        with col2:
            method_filter = st.selectbox(
                "🎯 Optimization Method:",
                options=["All Methods", "Max Sharpe", "Min Variance", "Equal Weight"],
                index=0,
                help="Filter strategies by optimization method"
            )
            
        with col3:
            frequency_filter = st.selectbox(
                "📅 Rebalancing Frequency:",
                options=["All Frequencies", "3M", "6M", "12M", "Static"],
                index=0,
                help="Filter strategies by rebalancing frequency"
            )
        
        # Apply filters to performance summary
        filtered_performance = {}
        for strategy, metrics in performance_summary.items():
            # Apply rebalancing filter
            if rebalancing_filter == "New Contrib Only" and "New Contrib" not in strategy:
                continue
            elif rebalancing_filter == "Full Rebal Only" and "Full Rebal" not in strategy:
                continue
            
            # Apply method filter
            if method_filter == "Max Sharpe" and "Max Sharpe" not in strategy:
                continue
            elif method_filter == "Min Variance" and "Min Variance" not in strategy:
                continue
            elif method_filter == "Equal Weight" and "Equal Weight" not in strategy:
                continue
            
            # Apply frequency filter
            if frequency_filter != "All Frequencies":
                if frequency_filter == "Static" and "Static" not in strategy:
                    continue
                elif frequency_filter != "Static" and f"({frequency_filter})" not in strategy:
                    continue
            
            filtered_performance[strategy] = metrics
        
        # Create performance DataFrame with filtered data
        perf_data = []
        for strategy, metrics in filtered_performance.items():
            perf_data.append({
                'Strategy': strategy,
                'Final Value': f"${metrics['Final Value']:,.0f}",
                'Total Return': f"{metrics['Total Return']:.1%}",
                'XIRR (Annual)': f"{metrics['XIRR']:.1%}",
                'Volatility': f"{metrics['Volatility']:.1%}",
                'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.2f}",
                'Max Drawdown': f"{metrics['Max Drawdown']:.1%}"
            })
        
        if perf_data:
            perf_df = pd.DataFrame(perf_data)
            
            # Sort by XIRR for better display
            perf_df['XIRR_numeric'] = [metrics['XIRR'] for metrics in filtered_performance.values()]
            perf_df = perf_df.sort_values('XIRR_numeric', ascending=False).drop('XIRR_numeric', axis=1)
            
            st.dataframe(perf_df, width='stretch', hide_index=True)
            
            st.caption(f"📋 Showing {len(perf_data)} of {len(performance_summary)} strategies")
        else:
            st.warning("⚠️ No strategies match the selected filters. Please adjust your filter criteria.")
        
        # Store filtered performance for charts
        st.session_state.filtered_performance = filtered_performance
        
        # Key Metrics Cards (based on filtered data)
        if filtered_performance:
            st.subheader("🎯 Key Insights")
            
            # Find best strategies from filtered data
            valid_strategies = {name: metrics for name, metrics in filtered_performance.items() 
                               if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])}
            
            if valid_strategies:
                best_return = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
                best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['Sharpe Ratio'])
                lowest_risk = min(valid_strategies.items(), key=lambda x: x[1]['Volatility'])
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🏆 Best Return",
                        best_return[0],
                        f"{best_return[1]['XIRR']:.1%} XIRR"
                    )
                    st.caption(f"Final Value: ${best_return[1]['Final Value']:,.0f}")
                
                with col2:
                    st.metric(
                        "⚖️ Best Risk-Adjusted",
                        best_sharpe[0],
                        f"{best_sharpe[1]['Sharpe Ratio']:.2f} Sharpe"
                    )
                    st.caption(f"Return: {best_sharpe[1]['XIRR']:.1%} | Risk: {best_sharpe[1]['Volatility']:.1%}")
                
                with col3:
                    st.metric(
                        "🛡️ Lowest Risk",
                        lowest_risk[0],
                        f"{lowest_risk[1]['Volatility']:.1%} Vol"
                    )
                    st.caption(f"Return: {lowest_risk[1]['XIRR']:.1%}")
    
    @staticmethod
    def render_interactive_charts(charts: Dict[str, go.Figure]):
        """Render interactive Plotly charts with filtering support"""
        
        # Check if we have filtered performance data
        filtered_performance = st.session_state.get('filtered_performance', {})
        
        if not filtered_performance:
            st.info("📊 Charts will show all strategies. Use filters above to focus on specific strategies.")
        else:
            st.info(f"📊 Charts filtered to show {len(filtered_performance)} selected strategies.")
        
        st.subheader("📈 Interactive Analysis Charts")
        
        # Portfolio Growth Chart
        st.markdown("### 🚀 Portfolio Growth Over Time")
        st.plotly_chart(charts['growth'], width='stretch')
        st.caption("💡 The gap between strategy lines and the dashed contribution line shows your investment gains!")
        
        # Returns Comparison
        st.markdown("### 📊 Annual Returns Comparison")
        st.plotly_chart(charts['returns'], width='stretch')
        st.caption("💡 Higher bars indicate better annual returns (XIRR)")
        
        # Risk vs Return Analysis
        st.markdown("### ⚖️ Risk vs Return Analysis")
        st.plotly_chart(charts['risk_return'], width='stretch')
        st.caption("💡 Top-left quadrant = Best (High Return, Low Risk). Bottom-right = Worst (Low Return, High Risk)")
    
    @staticmethod
    def render_strategy_recommendations(analysis_results: Dict):
        """Render detailed strategy recommendations"""
        
        st.subheader("🎯 Strategy Recommendations")
        
        performance_summary = analysis_results['performance_summary']
        
        # Get recommendations
        analyzer = SIPStrategyAnalyzer({'tickers': [], 'sip_amount': 1000})
        recommendations = analyzer.get_strategy_recommendations(performance_summary)
        
        if recommendations:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**🏆 {recommendations.get('Best Return', 'N/A')}**")
                st.caption("Highest absolute returns")
            
            with col2:
                st.success(f"**⚖️ {recommendations.get('Best Risk-Adjusted', 'N/A')}**")
                st.caption("Best return per unit of risk")
            
            with col3:
                st.warning(f"**🛡️ {recommendations.get('Lowest Risk', 'N/A')}**")
                st.caption("Most conservative option")
        
        # Implementation guidance
        st.markdown("### 📋 Implementation Guidance")
        
        guidance_text = """
        **🔄 Rebalancing Strategy:**
        - **3-Month Rebalancing**: More responsive to market changes, higher transaction costs
        - **6-Month Rebalancing**: Good balance between responsiveness and costs
        - **Static Allocation**: Lowest costs, but may miss optimization opportunities
        
        **💡 Key Considerations:**
        - **Transaction Costs**: Factor in 0.1% transaction costs for rebalancing
        - **Tax Implications**: Consider tax-loss harvesting opportunities
        - **Market Timing**: Strategies adapt to different market regimes
        - **Risk Tolerance**: Choose based on your comfort with volatility
        
        **⚠️ Important Notes:**
        - Past performance doesn't guarantee future results
        - Consider your personal financial situation
        - Diversification is key to risk management
        - Regular review and adjustment may be needed
        """
        
        st.markdown(guidance_text)
