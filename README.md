# MPT Portfolio Optimizer

A professional portfolio optimization tool built with Modern Portfolio Theory (MPT) for systematic investment planning.

## Features

🎯 **Professional Portfolio Optimization**
- Modern Portfolio Theory implementation
- Max Sharpe Ratio optimization
- Minimum Variance optimization
- Equal Weight baseline strategies

📊 **Comprehensive Analysis**
- Historical backtesting
- Risk-adjusted performance metrics
- Interactive visualizations
- Multiple rebalancing frequencies

💼 **Commercial-Ready Interface**
- Clean, professional UI
- Sidebar-based configuration
- Real-time portfolio optimization
- Export capabilities (CSV/Excel)

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   streamlit run main.py
   ```

3. **Configure Your Portfolio**
   - Set monthly investment amount
   - Select assets from various categories
   - Choose optimization methods
   - Set analysis period

4. **Optimize & Analyze**
   - Click "Optimize Portfolio"
   - Review performance metrics
   - Analyze risk vs return
   - Export results

## Asset Universe

The application supports a comprehensive range of assets:

- **US Equity ETFs**: VOO, QQQ, VTI, IWM, ARKK
- **International ETFs**: VWO, VEA, VXUS, EFA, VGK  
- **Bonds & Fixed Income**: BND, TLT, SHY, LQD, HYG
- **Cryptocurrency**: BTC-USD, ETH-USD, ADA-USD, SOL-USD
- **Sector ETFs**: XLK, XLF, XLV, XLE, XLI
- **Commodities & REITs**: GLD, SLV, VNQ, USO, DBA

## Optimization Methods

1. **Max Sharpe Ratio**: Maximizes risk-adjusted returns
2. **Min Variance**: Minimizes portfolio volatility  
3. **Equal Weight**: Simple baseline strategy

## Rebalancing Options

- **3 Months**: Higher maintenance, potentially better performance
- **6 Months**: Balanced approach (recommended)
- **12 Months**: Lower maintenance
- **Static**: No rebalancing

## Technical Architecture

- **Frontend**: Streamlit web interface
- **Optimization**: CVXPY for convex optimization
- **Data**: SQLite database with historical market data
- **Visualization**: Plotly for interactive charts
- **Export**: Excel/CSV export capabilities

## Performance Metrics

- **XIRR**: Annualized internal rate of return
- **Volatility**: Portfolio risk measure
- **Sharpe Ratio**: Risk-adjusted performance
- **Max Drawdown**: Largest peak-to-trough decline

## Commercial Use

This application serves as a proof-of-concept for a commercial portfolio optimization platform. Key features for commercial deployment:

- Scalable architecture
- Professional UI/UX
- Comprehensive analytics
- Export functionality
- Multiple optimization strategies

## License

This project is for demonstration purposes. Please ensure compliance with financial regulations for commercial use.