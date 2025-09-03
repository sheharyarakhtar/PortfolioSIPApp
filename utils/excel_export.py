"""
Excel export utilities for comprehensive financial analysis reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import io

# Try to import openpyxl, fall back to pandas-only solution if not available
try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.utils.dataframe import dataframe_to_rows
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


class ExcelExportManager:
    """Manages comprehensive Excel export functionality."""
    
    def __init__(self):
        self.use_openpyxl = OPENPYXL_AVAILABLE
        if self.use_openpyxl:
            self.workbook = Workbook()
            # Remove default sheet
            if 'Sheet' in self.workbook.sheetnames:
                self.workbook.remove(self.workbook['Sheet'])
        else:
            self.workbook = None
    
    def create_comprehensive_report(
        self, 
        analysis_results: Dict, 
        financial_summary: Dict,
        optimal_allocation: Optional[Dict] = None,
        sip_amount: float = 0
    ) -> Tuple[bytes, str, str]:
        """Create comprehensive Excel report with multiple sheets.
        
        Returns:
            Tuple of (file_data, filename, mime_type)
        """
        
        if self.use_openpyxl:
            return self._create_openpyxl_report(analysis_results, financial_summary, optimal_allocation, sip_amount)
        else:
            return self._create_pandas_report(analysis_results, financial_summary, optimal_allocation, sip_amount)
    
    def _create_openpyxl_report(self, analysis_results, financial_summary, optimal_allocation, sip_amount):
        """Create Excel report using openpyxl for advanced formatting."""
        # Sheet 1: Analysis Results
        self._create_analysis_results_sheet(analysis_results)
        
        # Sheet 2: Allocation Recommendations
        if optimal_allocation:
            self._create_allocation_recommendations_sheet(optimal_allocation, sip_amount)
        
        # Sheet 3: Performance Summary
        self._create_performance_summary_sheet(analysis_results)
        
        # Sheet 4: Monthly Financial Summary
        self._create_financial_summary_sheet(financial_summary)
        
        # Save to bytes
        output = io.BytesIO()
        self.workbook.save(output)
        output.seek(0)
        filename = f"mpt_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        return output.getvalue(), filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    
    def _create_pandas_report(self, analysis_results, financial_summary, optimal_allocation, sip_amount):
        """Create Excel report using pandas (fallback when openpyxl is not available)."""
        
        # Create DataFrames for each sheet
        sheets = {}
        
        # Sheet 1: Analysis Results
        performance_data = []
        for strategy, metrics in analysis_results['performance_summary'].items():
            performance_data.append({
                'Strategy': strategy,
                'Final Value ($)': f"{metrics['Final Value']:,.2f}",
                'Total Return (%)': f"{metrics['Total Return']*100:.2f}%",
                'XIRR (%)': f"{metrics['XIRR']*100:.2f}%",
                'Volatility (%)': f"{metrics['Volatility']*100:.2f}%",
                'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.3f}",
                'Max Drawdown (%)': f"{metrics['Max Drawdown']*100:.2f}%"
            })
        sheets['Analysis Results'] = pd.DataFrame(performance_data)
        
        # Sheet 2: Allocation Recommendations
        if optimal_allocation:
            allocation_data = []
            total_monthly = 0
            for asset, weight in optimal_allocation.items():
                if weight > 0.001:
                    monthly_amount = sip_amount * weight
                    annual_amount = monthly_amount * 12
                    total_monthly += monthly_amount
                    allocation_data.append({
                        'Asset': asset,
                        'Allocation %': f"{weight*100:.1f}%",
                        'Monthly Amount ($)': f"${monthly_amount:.2f}",
                        'Annual Amount ($)': f"${annual_amount:,.2f}"
                    })
            
            # Add total row
            allocation_data.append({
                'Asset': 'TOTAL',
                'Allocation %': '100.0%',
                'Monthly Amount ($)': f"${total_monthly:.2f}",
                'Annual Amount ($)': f"${total_monthly*12:,.2f}"
            })
            sheets['Allocation Recommendations'] = pd.DataFrame(allocation_data)
        
        # Sheet 3: Performance Summary
        performance_summary = analysis_results['performance_summary']
        valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                           if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                           and 'Benchmark' not in name and 'Custom' not in name}
        
        if valid_strategies:
            best_return = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['Sharpe Ratio'])
            lowest_risk = min(valid_strategies.items(), key=lambda x: x[1]['Volatility'])
            
            summary_data = []
            summary_data.append({'Metric': 'Best Return Strategy', 'Strategy': best_return[0], 'Value': f"{best_return[1]['XIRR']*100:.1f}% XIRR"})
            summary_data.append({'Metric': 'Best Risk-Adjusted', 'Strategy': best_sharpe[0], 'Value': f"{best_sharpe[1]['Sharpe Ratio']:.2f} Sharpe"})
            summary_data.append({'Metric': 'Lowest Risk Strategy', 'Strategy': lowest_risk[0], 'Value': f"{lowest_risk[1]['Volatility']*100:.1f}% Volatility"})
            
            sheets['Performance Summary'] = pd.DataFrame(summary_data)
        
        # Sheet 4: Financial Summary
        financial_data = []
        
        # Income section
        income_data = financial_summary.get('income', {})
        for source, amounts in income_data.items():
            financial_data.append({
                'Category': f'INCOME - {source}',
                'USD ($)': f"${amounts.get('usd', 0):,.2f}",
                'GBP (£)': f"£{amounts.get('gbp', 0):,.2f}",
                'PKR (₨)': f"₨{amounts.get('pkr', 0):,.0f}",
                'Notes': 'Monthly income'
            })
        
        # Expenses section
        expenses_data = financial_summary.get('expenses', {})
        for expense, amounts in expenses_data.items():
            financial_data.append({
                'Category': f'EXPENSE - {expense}',
                'USD ($)': f"${amounts.get('usd', 0):,.2f}",
                'GBP (£)': f"£{amounts.get('gbp', 0):,.2f}",
                'PKR (₨)': f"₨{amounts.get('pkr', 0):,.0f}",
                'Notes': 'Monthly expense'
            })
        
        # Investment section
        investments = financial_summary.get('investments', {})
        for category, data in investments.items():
            financial_data.append({
                'Category': f'INVESTMENT - {category}',
                'USD ($)': f"${data.get('amount', 0):,.2f}",
                'GBP (£)': '',
                'PKR (₨)': '',
                'Notes': data.get('notes', '')
            })
        
        sheets['Financial Summary'] = pd.DataFrame(financial_data)
        
        # Create Excel file using pandas
        output = io.BytesIO()
        try:
            # Try to use xlsxwriter engine (usually available)
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                for sheet_name, df in sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            output.seek(0)
            filename = f"mpt_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
            return output.getvalue(), filename, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            
        except Exception:
            # Final fallback: return as CSV with all data combined
            all_data = []
            for sheet_name, df in sheets.items():
                # Add sheet header
                all_data.append([f"=== {sheet_name} ==="])
                all_data.append([])  # Empty row
                
                # Add data
                all_data.append(df.columns.tolist())
                for _, row in df.iterrows():
                    all_data.append(row.tolist())
                
                all_data.append([])  # Empty row between sheets
                all_data.append([])  # Empty row between sheets
            
            # Convert to DataFrame and CSV
            max_cols = max(len(row) for row in all_data)
            for row in all_data:
                while len(row) < max_cols:
                    row.append('')
            
            combined_df = pd.DataFrame(all_data)
            csv_data = combined_df.to_csv(index=False, header=False)
            filename = f"mpt_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            return csv_data.encode('utf-8'), filename, "text/csv"
    
    def _create_analysis_results_sheet(self, analysis_results: Dict):
        """Create detailed analysis results sheet."""
        ws = self.workbook.create_sheet("Analysis Results")
        
        # Title
        ws.merge_cells('A1:G1')
        ws['A1'] = "MPT Portfolio Analysis Results"
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
        
        # Analysis date
        ws['A2'] = f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A2'].font = Font(italic=True)
        
        # Performance data
        performance_data = []
        for strategy, metrics in analysis_results['performance_summary'].items():
            performance_data.append({
                'Strategy': strategy,
                'Final Value ($)': f"{metrics['Final Value']:,.2f}",
                'Total Return (%)': f"{metrics['Total Return']*100:.2f}%",
                'XIRR (%)': f"{metrics['XIRR']*100:.2f}%",
                'Volatility (%)': f"{metrics['Volatility']*100:.2f}%",
                'Sharpe Ratio': f"{metrics['Sharpe Ratio']:.3f}",
                'Max Drawdown (%)': f"{metrics['Max Drawdown']*100:.2f}%"
            })
        
        # Convert to DataFrame and add to sheet
        df = pd.DataFrame(performance_data)
        
        # Add headers starting from row 4
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=True), 4):
            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                if r_idx == 4:  # Header row
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        
        # Auto-adjust column widths
        for column_cells in ws.columns:
            max_length = 0
            column_letter = None
            for cell in column_cells:
                try:
                    if hasattr(cell, 'column_letter'):
                        column_letter = cell.column_letter
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                except:
                    pass
            if column_letter:
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_allocation_recommendations_sheet(self, optimal_allocation: Dict, sip_amount: float):
        """Create allocation recommendations sheet."""
        ws = self.workbook.create_sheet("Allocation Recommendations")
        
        # Title
        ws.merge_cells('A1:D1')
        ws['A1'] = "Optimal Portfolio Allocation"
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
        
        # SIP amount info
        ws['A3'] = f"Monthly SIP Amount: ${sip_amount:,.2f}"
        ws['A3'].font = Font(bold=True, size=12)
        
        # Headers
        headers = ['Asset', 'Allocation %', 'Monthly Amount ($)', 'Annual Amount ($)']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=5, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
        
        # Allocation data
        row = 6
        total_monthly = 0
        for asset, weight in optimal_allocation.items():
            if weight > 0.001:  # Only show meaningful allocations
                monthly_amount = sip_amount * weight
                annual_amount = monthly_amount * 12
                total_monthly += monthly_amount
                
                ws.cell(row=row, column=1, value=asset)
                ws.cell(row=row, column=2, value=f"{weight*100:.1f}%")
                ws.cell(row=row, column=3, value=f"${monthly_amount:.2f}")
                ws.cell(row=row, column=4, value=f"${annual_amount:,.2f}")
                row += 1
        
        # Total row
        row += 1
        ws.cell(row=row, column=1, value="TOTAL").font = Font(bold=True)
        ws.cell(row=row, column=2, value="100.0%").font = Font(bold=True)
        ws.cell(row=row, column=3, value=f"${total_monthly:.2f}").font = Font(bold=True)
        ws.cell(row=row, column=4, value=f"${total_monthly*12:,.2f}").font = Font(bold=True)
        
        # Auto-adjust column widths
        for column_cells in ws.columns:
            max_length = 0
            column_letter = None
            for cell in column_cells:
                try:
                    if hasattr(cell, 'column_letter'):
                        column_letter = cell.column_letter
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                except:
                    pass
            if column_letter:
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_performance_summary_sheet(self, analysis_results: Dict):
        """Create performance summary with key metrics."""
        ws = self.workbook.create_sheet("Performance Summary")
        
        # Title
        ws.merge_cells('A1:E1')
        ws['A1'] = "Performance Summary & Key Metrics"
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
        
        performance_summary = analysis_results['performance_summary']
        
        # Find best strategies
        valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                           if not pd.isna(metrics['XIRR']) and not pd.isna(metrics['Volatility'])
                           and 'Benchmark' not in name and 'Custom' not in name}
        
        if valid_strategies:
            # Best performers
            best_return = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['Sharpe Ratio'])
            lowest_risk = min(valid_strategies.items(), key=lambda x: x[1]['Volatility'])
            
            # Key insights section
            ws['A3'] = "KEY INSIGHTS"
            ws['A3'].font = Font(size=14, bold=True)
            
            row = 4
            insights = [
                ("Best Return Strategy:", best_return[0], f"{best_return[1]['XIRR']*100:.1f}% XIRR"),
                ("Best Risk-Adjusted:", best_sharpe[0], f"{best_sharpe[1]['Sharpe Ratio']:.2f} Sharpe Ratio"),
                ("Lowest Risk Strategy:", lowest_risk[0], f"{lowest_risk[1]['Volatility']*100:.1f}% Volatility"),
            ]
            
            for insight, strategy, metric in insights:
                ws.cell(row=row, column=1, value=insight).font = Font(bold=True)
                ws.cell(row=row, column=2, value=strategy)
                ws.cell(row=row, column=3, value=metric)
                row += 1
            
            # Detailed comparison
            row += 2
            ws.cell(row=row, column=1, value="DETAILED COMPARISON").font = Font(size=14, bold=True)
            row += 1
            
            # Headers
            headers = ['Strategy', 'Final Value', 'XIRR', 'Volatility', 'Sharpe Ratio', 'Max Drawdown']
            for col, header in enumerate(headers, 1):
                cell = ws.cell(row=row, column=col, value=header)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            row += 1
            
            # Data rows
            for strategy, metrics in valid_strategies.items():
                ws.cell(row=row, column=1, value=strategy)
                ws.cell(row=row, column=2, value=f"${metrics['Final Value']:,.0f}")
                ws.cell(row=row, column=3, value=f"{metrics['XIRR']*100:.1f}%")
                ws.cell(row=row, column=4, value=f"{metrics['Volatility']*100:.1f}%")
                ws.cell(row=row, column=5, value=f"{metrics['Sharpe Ratio']:.2f}")
                ws.cell(row=row, column=6, value=f"{metrics['Max Drawdown']*100:.1f}%")
                row += 1
        
        # Auto-adjust column widths
        for column_cells in ws.columns:
            max_length = 0
            column_letter = None
            for cell in column_cells:
                try:
                    if hasattr(cell, 'column_letter'):
                        column_letter = cell.column_letter
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                except:
                    pass
            if column_letter:
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width
    
    def _create_financial_summary_sheet(self, financial_summary: Dict):
        """Create monthly financial summary sheet."""
        ws = self.workbook.create_sheet("Monthly Financial Summary")
        
        # Title
        ws.merge_cells('A1:D1')
        ws['A1'] = "Monthly Financial Breakdown"
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
        
        ws['A2'] = f"Report Date: {datetime.now().strftime('%Y-%m-%d')}"
        ws['A2'].font = Font(italic=True)
        
        # Income Section
        row = 4
        ws.cell(row=row, column=1, value="INCOME").font = Font(size=14, bold=True, color="008000")
        row += 1
        
        # Income headers
        headers = ['Source', 'USD ($)', 'GBP (£)', 'PKR (₨)']
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="E6FFE6", end_color="E6FFE6", fill_type="solid")
        row += 1
        
        # Income data
        income_data = financial_summary.get('income', {})
        for source, amounts in income_data.items():
            ws.cell(row=row, column=1, value=source)
            ws.cell(row=row, column=2, value=f"${amounts.get('usd', 0):,.2f}")
            ws.cell(row=row, column=3, value=f"£{amounts.get('gbp', 0):,.2f}")
            ws.cell(row=row, column=4, value=f"₨{amounts.get('pkr', 0):,.0f}")
            row += 1
        
        # Total income
        total_income = financial_summary.get('total_income_usd', 0)
        ws.cell(row=row, column=1, value="TOTAL INCOME").font = Font(bold=True)
        ws.cell(row=row, column=2, value=f"${total_income:,.2f}").font = Font(bold=True)
        row += 2
        
        # Expenses Section
        ws.cell(row=row, column=1, value="EXPENSES").font = Font(size=14, bold=True, color="CC0000")
        row += 1
        
        # Expense headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="FFE6E6", end_color="FFE6E6", fill_type="solid")
        row += 1
        
        # Expense data
        expenses_data = financial_summary.get('expenses', {})
        total_expenses_usd = 0
        for expense, amounts in expenses_data.items():
            usd_amount = amounts.get('usd', 0)
            total_expenses_usd += usd_amount
            ws.cell(row=row, column=1, value=expense)
            ws.cell(row=row, column=2, value=f"${usd_amount:,.2f}")
            ws.cell(row=row, column=3, value=f"£{amounts.get('gbp', 0):,.2f}")
            ws.cell(row=row, column=4, value=f"₨{amounts.get('pkr', 0):,.0f}")
            row += 1
        
        # Total expenses
        ws.cell(row=row, column=1, value="TOTAL EXPENSES").font = Font(bold=True)
        ws.cell(row=row, column=2, value=f"${total_expenses_usd:,.2f}").font = Font(bold=True)
        row += 2
        
        # Investment Allocation Section
        ws.cell(row=row, column=1, value="INVESTMENT ALLOCATION").font = Font(size=14, bold=True, color="0066CC")
        row += 1
        
        # Investment headers
        investment_headers = ['Category', 'Amount ($)', 'Percentage (%)', 'Notes']
        for col, header in enumerate(investment_headers, 1):
            cell = ws.cell(row=row, column=col, value=header)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="E6F2FF", end_color="E6F2FF", fill_type="solid")
        row += 1
        
        # Investment data
        investments = financial_summary.get('investments', {})
        for category, data in investments.items():
            amount = data.get('amount', 0)
            percentage = data.get('percentage', 0)
            notes = data.get('notes', '')
            
            ws.cell(row=row, column=1, value=category)
            ws.cell(row=row, column=2, value=f"${amount:,.2f}")
            ws.cell(row=row, column=3, value=f"{percentage:.1f}%")
            ws.cell(row=row, column=4, value=notes)
            row += 1
        
        # Available for investment
        available_amount = financial_summary.get('available_for_investment', 0)
        row += 1
        ws.cell(row=row, column=1, value="AVAILABLE FOR INVESTMENT").font = Font(bold=True, color="0066CC")
        ws.cell(row=row, column=2, value=f"${available_amount:,.2f}").font = Font(bold=True, color="0066CC")
        
        # Auto-adjust column widths
        for column_cells in ws.columns:
            max_length = 0
            column_letter = None
            for cell in column_cells:
                try:
                    if hasattr(cell, 'column_letter'):
                        column_letter = cell.column_letter
                        if cell.value and len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                except:
                    pass
            if column_letter:
                adjusted_width = min(max_length + 2, 50)
                ws.column_dimensions[column_letter].width = adjusted_width


def create_financial_summary_from_session(
    total_income: float,
    pkr_expenses_usd: float,
    international_amount: float,
    local_amount: float,
    cash_amount: float,
    exchange_rates: Dict[str, float]
) -> Dict:
    """Create financial summary from session data."""
    
    # This would be populated from the actual user inputs in the session
    # For now, creating a template structure
    return {
        'total_income_usd': total_income,
        'income': {
            'USD Income': {'usd': total_income, 'gbp': 0, 'pkr': 0},  # Simplified
        },
        'expenses': {
            'PKR Expenses': {
                'usd': pkr_expenses_usd, 
                'gbp': 0, 
                'pkr': pkr_expenses_usd * exchange_rates.get('usd_pkr_rate', 280)
            },
        },
        'investments': {
            'International Markets (MPT)': {
                'amount': international_amount,
                'percentage': (international_amount / total_income * 100) if total_income > 0 else 0,
                'notes': 'Optimized using Modern Portfolio Theory'
            },
            'Local Markets': {
                'amount': local_amount,
                'percentage': (local_amount / total_income * 100) if total_income > 0 else 0,
                'notes': 'PSX stocks, mutual funds, bonds, real estate'
            },
            'Cash/Savings': {
                'amount': cash_amount,
                'percentage': (cash_amount / total_income * 100) if total_income > 0 else 0,
                'notes': 'Emergency fund and liquidity'
            }
        },
        'available_for_investment': international_amount + local_amount + cash_amount
    }
