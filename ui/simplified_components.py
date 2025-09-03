"""
Simplified UI components focused on MPT-based SIP optimization.
"""

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple, Optional
from config.settings import DEFAULT_CONFIG

class SimplifiedInputManager:
    """Manages simplified input forms focused on SIP optimization."""
    
    @staticmethod
    def render_income_and_pkr_expenses(exchange_rates: Dict[str, float]) -> Tuple[float, float]:
        """Render income inputs and configurable PKR expenses."""
        
        st.header("💰 Monthly Income Setup")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📥 Monthly Income Sources")
            usd_income = st.number_input("USD Income:", min_value=0, value=DEFAULT_CONFIG['monthly_income'], step=100)
            pkr_income = st.number_input("PKR Income:", min_value=0, value=0, step=1000)
            gbp_income = st.number_input("GBP Income:", min_value=0, value=0, step=100)
            
            # Convert all to USD
            pkr_to_usd = pkr_income / exchange_rates['usd_pkr_rate'] 
            gbp_to_usd = gbp_income * exchange_rates['gbp_usd_rate']
            
            total_monthly_income = usd_income + pkr_to_usd + gbp_to_usd
            
            st.success(f"**Total Monthly Income: ${total_monthly_income:,.2f}**")
        
        with col2:
            st.subheader("💸 PKR Exchange Rate Info")
            st.metric("Current USD/PKR Rate", f"{exchange_rates['usd_pkr_rate']:.2f}")
            st.caption("This rate is used for all PKR ↔ USD conversions")
        
        return total_monthly_income, exchange_rates['usd_pkr_rate']
    
    @staticmethod
    def render_configurable_pkr_expenses(usd_pkr_rate: float) -> float:
        """Render configurable PKR expenses section."""
        
        st.header("💸 Configure PKR Expenses")
        st.markdown("*Adjust your monthly PKR requirements - perfect for sharing with others who have different needs!*")
        
        # Initialize session state for PKR expenses if not exists
        if 'pkr_expenses' not in st.session_state:
            st.session_state.pkr_expenses = DEFAULT_CONFIG['pkr_allocations'].copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 PKR Expense Categories")
            
            # Editable PKR expenses
            updated_expenses = {}
            
            for category, default_amount in st.session_state.pkr_expenses.items():
                updated_amount = st.number_input(
                    f"{category}:",
                    min_value=0,
                    value=default_amount,
                    step=5000,
                    key=f"pkr_{category}"
                )
                updated_expenses[category] = updated_amount
            
            # Add custom expense category
            st.markdown("**Add Custom Expense:**")
            col_name, col_amount = st.columns(2)
            with col_name:
                custom_name = st.text_input("Category Name", placeholder="e.g., Local Investment", key="custom_pkr_name")
            with col_amount:
                custom_amount = st.number_input("Amount (PKR)", min_value=0, step=5000, key="custom_pkr_amount")
            
            if st.button("➕ Add Custom Expense"):
                if custom_name and custom_amount > 0:
                    st.session_state.pkr_expenses[custom_name] = custom_amount
                    st.success(f"Added {custom_name}: ₨{custom_amount:,}")
                    st.rerun()
                else:
                    st.error("Please enter both name and amount")
            
            # Update session state
            st.session_state.pkr_expenses.update(updated_expenses)
        
        with col2:
            st.subheader("📊 PKR Summary")
            
            # Calculate totals
            total_pkr_expenses = sum(st.session_state.pkr_expenses.values())
            usd_needed_for_pkr = total_pkr_expenses / usd_pkr_rate
            
            st.metric("Total PKR Expenses", f"₨{total_pkr_expenses:,.0f}")
            st.metric("USD Equivalent", f"${usd_needed_for_pkr:,.0f}")
            
            # Show breakdown
            st.markdown("**Breakdown:**")
            for category, amount in st.session_state.pkr_expenses.items():
                if amount > 0:
                    usd_equiv = amount / usd_pkr_rate
                    st.write(f"• {category}: ₨{amount:,} (${usd_equiv:.0f})")
            
            # Reset to defaults button
            if st.button("🔄 Reset to Defaults"):
                st.session_state.pkr_expenses = DEFAULT_CONFIG['pkr_allocations'].copy()
                st.rerun()
        
        return usd_needed_for_pkr
    
    @staticmethod
    def render_investment_allocation_inputs(total_income: float, pkr_expenses_usd: float) -> Tuple[float, float, float, float]:
        """Render investment allocation inputs after PKR expenses are configured."""
        
        st.header("🎯 Investment Allocation Strategy")
        
        # Calculate available for investment
        available_usd = total_income - pkr_expenses_usd
        
        if available_usd <= 0:
            st.error("Your income is less than your PKR expenses!")
            return 0, 0, 0, 0
        
        st.success(f"**Available for Investment: ${available_usd:,.2f}**")
        st.caption(f"After PKR expenses: ${total_income:,.0f} - ${pkr_expenses_usd:.0f} = ${available_usd:,.0f}")
        
        st.markdown("---")
        
        # Investment allocation
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**💵 International Markets (MPT Optimized)**")
            international_percentage = st.slider(
                "International %:", 
                min_value=0, 
                max_value=100, 
                value=60,
                help="This portion will be optimized using Modern Portfolio Theory"
            )
            international_amount = available_usd * (international_percentage / 100)
            st.caption(f"${international_amount:,.0f}/month")
            
            st.markdown("*Includes:*")
            st.caption("• US Stocks (VOO, QQQ, VTI)")
            st.caption("• International Stocks (VWO)")
            st.caption("• Bonds (BND)")
            st.caption("• Crypto (BTC, ETH)")
        
        with col2:
            st.markdown("**🏠 Local Investments**")
            local_percentage = st.slider(
                "Local %:", 
                min_value=0, 
                max_value=100, 
                value=30,
                help="Fixed allocation to local securities not available on Yahoo Finance"
            )
            local_amount = available_usd * (local_percentage / 100)
            st.caption(f"${local_amount:,.0f}/month")
            
            st.markdown("*Includes:*")
            st.caption("• PSX Stocks (Stockintel)")
            st.caption("• Mutual Funds (Behtari)")
            st.caption("• Pakistan Bond Market")
            st.caption("• Local Real Estate")
        
        with col3:
            st.markdown("**💰 Cash/Savings**")
            cash_percentage = 100 - international_percentage - local_percentage
            cash_amount = available_usd * (cash_percentage / 100)
            
            # Display cash percentage (read-only)
            st.metric("Cash %", f"{cash_percentage}%")
            st.caption(f"${cash_amount:,.0f}/month")
            
            st.markdown("*Emergency fund & liquidity*")
        
        # Validation
        if international_percentage + local_percentage > 100:
            st.error(" Total allocation cannot exceed 100%")
            return 0, 0, 0, 0
        
        # Show summary
        st.markdown("---")
        st.subheader("📊 Monthly Investment Summary")
        
        summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
        
        with summary_col1:
            st.metric("Total Available", f"${available_usd:,.0f}")
        
        with summary_col2:
            st.metric("International (MPT)", f"${international_amount:,.0f}")
        
        with summary_col3:
            st.metric("Local (Fixed)", f"${local_amount:,.0f}")
        
        with summary_col4:
            st.metric("Cash/Savings", f"${cash_amount:,.0f}")
        
        return available_usd, international_amount, local_amount, cash_amount
    
    @staticmethod
    def render_mpt_configuration() -> Tuple[List[str], Dict[str, int]]:
        """Render MPT strategy configuration with flexible asset selection."""
        
        st.header("🎯 MPT Strategy Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Asset Universe")
            st.markdown("*Choose assets for MPT optimization:*")
            
            # Predefined asset categories ( = Reliable,  = May not be available)
            asset_categories = {
                "🇺🇸 US Equity ETFs": {
                    'VOO': 'S&P 500 ETF',
                    'QQQ': 'Nasdaq 100 ETF',
                    'VTI': 'Total US Market ETF',
                    'IWM': 'Russell 2000 Small Cap ETF',
                    'ARKK': 'Innovation EFT'
                },
                "🌍 International ETFs": {
                    'VWO': 'Emerging Markets ETF',
                    'VEA': 'Developed Markets ETF',
                    'VXUS': 'Total International Stock ETF',
                    'EFA': 'EAFE ETF',
                    'VGK': 'European ETF'
                },
                "🏛️ Bonds & Fixed Income": {
                    'BND': 'Total Bond Market ETF',
                    'TLT': 'Long-Term Treasury ETF',
                    'SHY': 'Short-Term Treasury ETF',
                    'LQD': 'Investment Grade Corporate Bonds',
                    'HYG': 'High Yield Corporate Bonds'
                },
                "₿ Cryptocurrency": {
                    'BTC-USD': 'Bitcoin',
                    'ETH-USD': 'Ethereum',
                    'ADA-USD': 'Cardano',
                    'SOL-USD': 'Solana'
                },
                "🏭 Sector ETFs": {
                    'XLK': 'Technology Sector ETF',
                    'XLF': 'Financial Sector ETF',
                    'XLV': 'Healthcare Sector ETF',
                    'XLE': 'Energy Sector ETF',
                    'XLI': 'Industrial Sector ETF'
                },
                "🏗️ Commodities & REITs": {
                    'GLD': 'Gold EFT',
                    'SLV': 'Silver EFT',
                    'VNQ': 'Real Estate EFT',
                    'USO': 'Oil EFT',
                    'DBA': 'Agriculture EFT'
                }
            }
            
            selected_assets = []
            
            # Create expandable sections for each category
            for category, assets in asset_categories.items():
                with st.expander(f"{category} ({len(assets)} assets)", expanded=(category == "🇺🇸 US Equity ETFs")):
                    
                    # Quick select buttons
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button(f"Select All", key=f"select_all_{category}"):
                            for ticker in assets.keys():
                                st.session_state[f"asset_{ticker}"] = True
                            st.rerun()
                    
                    with col_b:
                        if st.button(f"Select None", key=f"select_none_{category}"):
                            for ticker in assets.keys():
                                st.session_state[f"asset_{ticker}"] = False
                            st.rerun()
                    
                    with col_c:
                        selected_in_category = sum(1 for ticker in assets.keys() 
                                                 if st.session_state.get(f"asset_{ticker}", False))
                        st.caption(f"{selected_in_category}/{len(assets)} selected")
                    
                    # Asset checkboxes
                    for ticker, name in assets.items():
                        # Set default selection for common assets
                        default_selected = ticker in ['VOO', 'QQQ', 'VTI', 'BND', 'BTC-USD', 'ETH-USD']
                        
                        if st.checkbox(f"{ticker} - {name}", 
                                     value=st.session_state.get(f"asset_{ticker}", default_selected), 
                                     key=f"asset_{ticker}"):
                            selected_assets.append(ticker)
            
            # Custom asset input
            with st.expander("➕ Add Custom Asset", expanded=False):
                st.markdown("*Add any Yahoo Finance ticker:*")
                
                col_ticker, col_name = st.columns(2)
                with col_ticker:
                    custom_ticker = st.text_input("Ticker Symbol", placeholder="AAPL", key="custom_ticker")
                with col_name:
                    custom_name = st.text_input("Display Name", placeholder="Apple Inc.", key="custom_name")
                
                if st.button("➕ Add Custom Asset"):
                    if custom_ticker and custom_name:
                        # Add to session state
                        if 'custom_assets' not in st.session_state:
                            st.session_state.custom_assets = {}
                        
                        st.session_state.custom_assets[custom_ticker.upper()] = custom_name
                        st.session_state[f"asset_{custom_ticker.upper()}"] = True
                        st.success(f"Added {custom_ticker.upper()} - {custom_name}")
                        st.rerun()
                    else:
                        st.error("Please enter both ticker and name")
                
                # Show custom assets
                if 'custom_assets' in st.session_state and st.session_state.custom_assets:
                    st.markdown("**Custom Assets:**")
                    for ticker, name in st.session_state.custom_assets.items():
                        col_check, col_remove = st.columns([3, 1])
                        with col_check:
                            if st.checkbox(f"{ticker} - {name}", 
                                         value=st.session_state.get(f"asset_{ticker}", False),
                                         key=f"asset_{ticker}"):
                                selected_assets.append(ticker)
                        with col_remove:
                            if st.button("🗑️", key=f"remove_{ticker}"):
                                del st.session_state.custom_assets[ticker]
                                if f"asset_{ticker}" in st.session_state:
                                    del st.session_state[f"asset_{ticker}"]
                                st.rerun()
            
            # Validation and summary
            st.markdown("---")
            if len(selected_assets) < 2:
                st.error(" Select at least 2 assets for optimization")
            else:
                st.success(f" {len(selected_assets)} assets selected for optimization")
                
                # Known problematic tickers (based on common issues)
                potentially_unavailable = {
                    'VEA', 'VXUS', 'EFA', 'VGK',  # Some international ETFs
                    'LQD', 'HYG',  # Some bond ETFs
                    'ADA-USD', 'SOL-USD',  # Some crypto
                    'XLK', 'XLF', 'XLV', 'XLE', 'XLI',  # Some sector ETFs
                    'SLV', 'USO', 'DBA'  # Some commodities
                }
                
                problematic_selected = [asset for asset in selected_assets if asset in potentially_unavailable]
                
                if problematic_selected:
                    st.warning(f" Note: These assets may not be available: {', '.join(problematic_selected)}")
                    st.caption("The app will validate and skip unavailable assets during analysis.")
                
                # Show selected assets summary
                with st.expander("📋 Selected Assets Summary", expanded=False):
                    for asset in sorted(selected_assets):
                        # Find the name from all categories
                        asset_name = "Unknown"
                        for category, assets in asset_categories.items():
                            if asset in assets:
                                asset_name = assets[asset]
                                break
                        
                        # Check custom assets
                        if 'custom_assets' in st.session_state and asset in st.session_state.custom_assets:
                            asset_name = st.session_state.custom_assets[asset]
                        
                        # Mark potentially problematic assets
                        warning_icon = " " if asset in potentially_unavailable else "• "
                        st.write(f"{warning_icon}{asset} - {asset_name}")
        
        with col2:
            st.subheader("🔄 Rebalancing Strategies")
            st.markdown("*Choose rebalancing frequencies to test:*")
            
            rebalancing_strategies = {}
            
            strategies = [
                ("Static (No Rebalancing)", "static"),
                ("Quarterly (3 Months)", "3m"),
                ("Semi-Annual (6 Months)", "6m"),
                ("Annual (12 Months)", "12m")
            ]
            
            strategy_months = {
                "Static (No Rebalancing)": 12,
                "Quarterly (3 Months)": 3,
                "Semi-Annual (6 Months)": 6,
                "Annual (12 Months)": 12
            }
            
            for strategy_name, key_suffix in strategies:
                if st.checkbox(strategy_name, value=True, key=f"rebal_{key_suffix}"):
                    rebalancing_strategies[strategy_name] = strategy_months[strategy_name]
            
            if not rebalancing_strategies:
                st.error(" Select at least one rebalancing strategy")
        
        return selected_assets, rebalancing_strategies
    
    @staticmethod
    def render_custom_allocation_input(selected_assets: List[str]) -> Optional[Dict[str, float]]:
        """Render custom allocation input for user's own portfolio breakdown."""
        
        if not selected_assets or len(selected_assets) < 2:
            return None
        
        st.subheader("🎯 Test Your Own Allocation")
        st.markdown("*Enter your custom allocation percentages to compare against MPT strategies:*")
        
        custom_allocation = {}
        total_allocation = 0
        
        # Create input fields for each selected asset
        cols = st.columns(min(len(selected_assets), 3))  # Max 3 columns
        
        for i, asset in enumerate(selected_assets):
            col_idx = i % len(cols)
            with cols[col_idx]:
                # Get asset name for display (comprehensive mapping)
                asset_names = {
                    # US Equity ETFs
                    'VOO': 'S&P 500', 'QQQ': 'Nasdaq 100', 'VTI': 'Total US Market',
                    'IWM': 'Russell 2000', 'ARKK': 'Innovation',
                    # International ETFs
                    'VWO': 'Emerging Markets', 'VEA': 'Developed Markets', 
                    'VXUS': 'Total International', 'EFA': 'EAFE', 'VGK': 'European',
                    # Bonds & Fixed Income
                    'BND': 'Total Bond Market', 'TLT': 'Long-Term Treasury',
                    'SHY': 'Short-Term Treasury', 'LQD': 'Investment Grade Corp',
                    'HYG': 'High Yield Corp',
                    # Cryptocurrency
                    'BTC-USD': 'Bitcoin', 'ETH-USD': 'Ethereum',
                    'ADA-USD': 'Cardano', 'SOL-USD': 'Solana',
                    # Sector ETFs
                    'XLK': 'Technology', 'XLF': 'Financial', 'XLV': 'Healthcare',
                    'XLE': 'Energy', 'XLI': 'Industrial',
                    # Commodities & REITs
                    'GLD': 'Gold', 'SLV': 'Silver', 'VNQ': 'Real Estate',
                    'USO': 'Oil', 'DBA': 'Agriculture'
                }
                
                # Add custom assets if they exist
                if 'custom_assets' in st.session_state:
                    for ticker, name in st.session_state.custom_assets.items():
                        asset_names[ticker] = name
                
                display_name = asset_names.get(asset, asset)
                
                allocation = st.number_input(
                    f"{asset} ({display_name}) %:",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0 / len(selected_assets),  # Equal weight default
                    step=0.1,
                    key=f"custom_alloc_{asset}"
                )
                
                if allocation > 0:
                    custom_allocation[asset] = allocation
                    total_allocation += allocation
        
        # Show total and validation
        col1, col2 = st.columns(2)
        
        with col1:
            if abs(total_allocation - 100) > 0.1:
                st.error(f" Total: {total_allocation:.1f}% (should be 100%)")
                return None
            else:
                st.success(f" Total: {total_allocation:.1f}%")
        
        with col2:
            if st.button("🔄 Reset to Equal Weight", key="reset_custom"):
                st.rerun()
        
        # Convert to format expected by backtesting
        if abs(total_allocation - 100) <= 0.1 and custom_allocation:
            # Normalize to exactly 100%
            normalized_allocation = {k: (v / total_allocation) * 100 for k, v in custom_allocation.items()}
            return normalized_allocation
        
        return None

class SimplifiedDisplayManager:
    """Manages simplified data display."""
    
    @staticmethod
    def render_local_allocation_guide(local_amount: float):
        """Render guide for local allocation."""
        if local_amount > 0:
            st.header("🏠 Local Investment Allocation Guide")
            
            st.markdown(f"""
            **Monthly Local Investment: ${local_amount:,.0f}**
            
            *Suggested breakdown based on your risk profile:*
            """)
            
            # Suggested local breakdown (fixed distribution)
            local_breakdown = {
                "PSX Stocks (Stockintel)": 0.35,
                "Mutual Funds (Behtari)": 0.35,
                "Pakistan Bond Market": 0.20,
                "Local Real Estate Fund": 0.10
            }
            breakdown_data = []
            for investment, percentage in local_breakdown.items():
                amount = local_amount * percentage
                breakdown_data.append({
                    "Investment": investment,
                    "Allocation %": f"{percentage*100:.0f}%",
                    "Monthly Amount": f"${amount:.0f}",
                    "Expected Return": "8-12% annually"
                })
            
            breakdown_df = pd.DataFrame(breakdown_data)
            st.dataframe(breakdown_df, width='stretch', hide_index=True)
            
            st.info("💡 **Note**: Local investments are not included in MPT optimization due to limited data availability. Allocate manually based on your research and risk tolerance.")
