"""
Enhanced SIP Strategy module for portfolio optimization app.
Integrates the Enhanced SIP Strategy with the MPT portfolio app architecture.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SIPStrategyAnalyzer:
    """Enhanced SIP Strategy analyzer integrated with portfolio app"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.available_tickers = config['tickers']
        self.sip_amount = config['sip_amount']
        self.transaction_cost_rate = config.get('transaction_cost_rate', 0.001)
        
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """Validate ticker availability and return valid/invalid lists."""
        import yfinance as yf
        
        valid_tickers = []
        invalid_tickers = []
        
        print(f"🔍 Validating {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                # Try to get basic info for the ticker
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                # Check if we got meaningful data
                if info and ('regularMarketPrice' in info or 'previousClose' in info or 'currentPrice' in info):
                    valid_tickers.append(ticker)
                    print(f"✅ {ticker} - Valid")
                else:
                    # Try downloading a small sample to double-check
                    test_data = ticker_obj.history(period="5d")
                    if not test_data.empty and len(test_data) > 0:
                        valid_tickers.append(ticker)
                        print(f"✅ {ticker} - Valid (via history)")
                    else:
                        invalid_tickers.append(ticker)
                        print(f"❌ {ticker} - No data available")
                        
            except Exception as e:
                invalid_tickers.append(ticker)
                print(f"❌ {ticker} - Error: {str(e)[:50]}...")
        
        print(f"📊 Validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")
        return valid_tickers, invalid_tickers
    
    def download_data(self, start_date: str, end_date: str, use_cache: bool = True) -> pd.DataFrame:
        """Download and clean market data with caching support"""
        
        # Create cache key
        cache_key = f"sip_data_{start_date}_{end_date}_{'_'.join(sorted(self.available_tickers))}"
        
        # Try to use cached data if available and requested
        if use_cache and hasattr(self, '_data_cache') and cache_key in self._data_cache:
            cached_data, cached_start, cached_end = self._data_cache[cache_key]
            
            # Check if cached data covers the requested period
            if (pd.to_datetime(cached_start) <= pd.to_datetime(start_date) and 
                pd.to_datetime(cached_end) >= pd.to_datetime(end_date)):
                
                print(f"📋 Using cached data from {cached_start} to {cached_end}")
                
                # Return subset of cached data for requested period
                mask = (cached_data.index >= start_date) & (cached_data.index <= end_date)
                return cached_data.loc[mask]
        
        try:
            print(f"📊 Downloading fresh data from {start_date} to {end_date}...")
            data = yf.download(self.available_tickers, start=start_date, end=end_date, progress=False)["Close"]
            
            if data.empty:
                raise ValueError("No data downloaded")
            
            # Handle missing data with forward fill
            data = data.ffill().bfill()
            data = data.dropna(how='all')
            
            # Cache the data
            if not hasattr(self, '_data_cache'):
                self._data_cache = {}
            
            self._data_cache[cache_key] = (data, start_date, end_date)
            print(f"💾 Data cached for future use")
            
            return data
            
        except Exception as e:
            raise Exception(f"Error downloading data: {e}")
    
    def detect_market_regime(self, returns: pd.DataFrame, lookback_days: int = 252) -> str:
        """Simple market regime detection based on volatility and returns"""
        recent_returns = returns.tail(lookback_days)
        
        # Calculate market metrics
        avg_return = recent_returns.mean().mean() * 252  # Annualized
        volatility = recent_returns.std().mean() * np.sqrt(252)  # Annualized
        
        # Simple regime classification
        if avg_return > 0.15 and volatility < 0.25:
            return "bull_low_vol"
        elif avg_return > 0.10 and volatility > 0.25:
            return "bull_high_vol"
        elif avg_return < 0.05 and volatility > 0.30:
            return "bear_high_vol"
        else:
            return "neutral"
    
    def optimize_portfolio(self, returns_data: pd.DataFrame, regime: str = "neutral", 
                          method: str = "max_sharpe") -> Optional[np.ndarray]:
        """Optimize portfolio with regime-aware constraints"""
        
        # Regime-specific parameters
        regime_params = {
            "bull_low_vol": {"max_weight": 0.50, "min_weight": 0.02, "rf": 0.04},
            "bull_high_vol": {"max_weight": 0.40, "min_weight": 0.03, "rf": 0.04},
            "bear_high_vol": {"max_weight": 0.30, "min_weight": 0.05, "rf": 0.03},
            "neutral": {"max_weight": 0.40, "min_weight": 0.02, "rf": 0.04}
        }
        
        params = regime_params.get(regime, regime_params["neutral"])
        
        try:
            mu = np.array(returns_data.mean()) * 252
            Sigma = np.array(returns_data.cov()) * 252
            n = len(mu)
            
            # Add regularization to covariance matrix
            Sigma += np.eye(n) * 1e-6
            
            w = cp.Variable(n)
            
            if method == "max_sharpe":
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                constraints = [
                    (mu - params["rf"]) @ w == 1,
                    w >= params["min_weight"],
                    w <= params["max_weight"]
                ]
            else:  # min_variance
                objective = cp.Minimize(cp.quad_form(w, Sigma))
                constraints = [
                    cp.sum(w) == 1,
                    w >= params["min_weight"],
                    w <= params["max_weight"]
                ]
            
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.OSQP, verbose=False)
            
            if w.value is not None and prob.status == 'optimal':
                if method == "max_sharpe":
                    weights = w.value / np.sum(w.value)
                else:
                    weights = w.value
                return weights
            
        except Exception as e:
            print(f"Optimization failed: {e}")
        
        return None
    
    def calculate_transaction_costs(self, old_weights: np.ndarray, new_weights: np.ndarray, 
                                  portfolio_value: float) -> float:
        """Calculate realistic transaction costs"""
        if old_weights is None or len(old_weights) == 0:
            return 0
        
        turnover = np.sum(np.abs(new_weights - old_weights)) / 2
        return turnover * portfolio_value * self.transaction_cost_rate
    
    def backtest_strategy(self, data: pd.DataFrame, strategy_config: Dict) -> Dict:
        """Backtest a specific strategy with proper SIP methodology"""
        
        strategy_name = strategy_config['name']
        rebalance_freq = strategy_config['rebalance_months']
        optimization_method = strategy_config.get('method', 'max_sharpe')
        full_rebalancing = strategy_config.get('full_rebalancing', False)
        
        # Initialize tracking
        portfolio_values = pd.Series(index=data.index, dtype=float)
        monthly_contributions = pd.Series(dtype=float)
        shares_held = np.zeros(len(self.available_tickers))
        
        # Create rebalancing and SIP dates
        monthly_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq='MS')
        monthly_dates = monthly_dates[monthly_dates <= data.index[-1]]
        
        rebalance_dates = pd.date_range(start=data.index[0], end=data.index[-1], 
                                      freq=f'{rebalance_freq}MS')
        rebalance_dates = rebalance_dates[rebalance_dates <= data.index[-1]]
        
        for date in data.index:
            # Monthly SIP contribution
            if date in monthly_dates:
                monthly_contributions[date] = self.sip_amount
                
                # Rebalancing logic - need at least 1 year of history
                min_history_days = min(252, len(data.index) - 1)  # Use available data or 252 days
                if date in rebalance_dates and date > data.index[min_history_days]:
                    
                    # Get training data (exclude current date)
                    train_start = date - timedelta(days=365*2)  # Use 2 years of training data
                    train_end = date - timedelta(days=1)
                    train_data = data.loc[train_start:train_end]
                    
                    if len(train_data) > 252:  # Need at least 1 year
                        train_returns = train_data.pct_change().dropna()
                        
                        # Detect market regime
                        regime = self.detect_market_regime(train_returns)
                        
                        # Optimize portfolio
                        optimal_weights = self.optimize_portfolio(train_returns, regime, optimization_method)
                        
                        if optimal_weights is not None:
                            # Calculate current portfolio value
                            current_value = sum([shares_held[j] * data.loc[date, self.available_tickers[j]] 
                                             for j in range(len(self.available_tickers))])
                            
                            if full_rebalancing:
                                # Full Portfolio Rebalancing: Sell everything and reallocate
                                total_value = current_value + self.sip_amount
                                
                                if current_value > 0:
                                    current_weights = np.array([
                                        shares_held[j] * data.loc[date, self.available_tickers[j]] / current_value 
                                        for j in range(len(self.available_tickers))
                                    ])
                                else:
                                    current_weights = np.zeros(len(self.available_tickers))
                                
                                # Calculate transaction costs for full rebalancing
                                transaction_cost = self.calculate_transaction_costs(
                                    current_weights, optimal_weights, total_value
                                )
                                
                                # Rebalance entire portfolio with transaction costs
                                net_value = total_value - transaction_cost
                                shares_held = np.array([
                                    (net_value * optimal_weights[j]) / data.loc[date, self.available_tickers[j]]
                                    for j in range(len(self.available_tickers))
                                ])
                            else:
                                # Conservative Approach: Only allocate new contributions optimally
                                # Keep existing holdings unchanged, allocate new SIP amount according to optimal weights
                                for j in range(len(self.available_tickers)):
                                    shares_to_buy = (self.sip_amount * optimal_weights[j]) / data.loc[date, self.available_tickers[j]]
                                    shares_held[j] += shares_to_buy
                        else:
                            # Fallback to equal weight if optimization fails
                            equal_weights = np.ones(len(self.available_tickers)) / len(self.available_tickers)
                            for j in range(len(self.available_tickers)):
                                shares_to_buy = (self.sip_amount * equal_weights[j]) / data.loc[date, self.available_tickers[j]]
                                shares_held[j] += shares_to_buy
                    else:
                        # Not enough data, use equal weight
                        equal_weights = np.ones(len(self.available_tickers)) / len(self.available_tickers)
                        for j in range(len(self.available_tickers)):
                            shares_to_buy = (self.sip_amount * equal_weights[j]) / data.loc[date, self.available_tickers[j]]
                            shares_held[j] += shares_to_buy
                else:
                    # Regular SIP contribution (non-rebalancing month)
                    current_value = sum([shares_held[j] * data.loc[date, self.available_tickers[j]] 
                                      for j in range(len(self.available_tickers))])
                    
                    if current_value > 0:
                        if full_rebalancing:
                            # For full rebalancing, maintain current weights until next rebalance
                            current_weights = np.array([
                                shares_held[j] * data.loc[date, self.available_tickers[j]] / current_value 
                                for j in range(len(self.available_tickers))
                            ])
                            
                            for j in range(len(self.available_tickers)):
                                shares_to_buy = (self.sip_amount * current_weights[j]) / data.loc[date, self.available_tickers[j]]
                                shares_held[j] += shares_to_buy
                        else:
                            # For conservative approach, check if we have a previous optimal allocation
                            # If not available, use current weights as fallback
                            current_weights = np.array([
                                shares_held[j] * data.loc[date, self.available_tickers[j]] / current_value 
                                for j in range(len(self.available_tickers))
                            ])
                            
                            for j in range(len(self.available_tickers)):
                                shares_to_buy = (self.sip_amount * current_weights[j]) / data.loc[date, self.available_tickers[j]]
                                shares_held[j] += shares_to_buy
                    else:
                        # First contribution, use equal weight
                        equal_weights = np.ones(len(self.available_tickers)) / len(self.available_tickers)
                        for j in range(len(self.available_tickers)):
                            shares_to_buy = (self.sip_amount * equal_weights[j]) / data.loc[date, self.available_tickers[j]]
                            shares_held[j] += shares_to_buy
            
            # Calculate portfolio value
            portfolio_value = sum([shares_held[j] * data.loc[date, self.available_tickers[j]] 
                                for j in range(len(self.available_tickers))])
            portfolio_values[date] = portfolio_value
        
        # Forward fill portfolio values to handle non-SIP dates
        portfolio_values = portfolio_values.ffill().fillna(0)
        
        return {
            'portfolio_values': portfolio_values,
            'monthly_contributions': monthly_contributions,
            'strategy_name': strategy_name
        }
    
    def calculate_performance_metrics(self, portfolio_values: pd.Series, 
                                    monthly_contributions: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        # Ensure we have valid data
        if len(portfolio_values) == 0 or len(monthly_contributions) == 0:
            return self._get_default_metrics()
        
        # Remove any NaN values and ensure we have valid final value
        portfolio_values = portfolio_values.dropna()
        if len(portfolio_values) == 0:
            return self._get_default_metrics()
        
        # Basic metrics
        total_contributions = monthly_contributions.sum()
        final_value = portfolio_values.iloc[-1]
        
        # Handle edge cases
        if total_contributions <= 0 or np.isnan(total_contributions) or np.isnan(final_value):
            return self._get_default_metrics()
            
        total_return = (final_value / total_contributions) - 1
        
        # XIRR calculation (simplified)
        try:
            from scipy.optimize import fsolve
            
            def xnpv(rate, cash_flows, dates):
                if rate <= -1:
                    return float('inf')
                return sum([cf / (1 + rate) ** ((date - dates[0]).days / 365.0) 
                           for cf, date in zip(cash_flows, dates)])
            
            cf_dates = list(monthly_contributions.index) + [portfolio_values.index[-1]]
            cf_values = [-x for x in monthly_contributions.values] + [final_value]
            
            if len(cf_values) >= 2:
                xirr = fsolve(lambda r: xnpv(r[0], cf_values, cf_dates), [0.1])[0]
                if np.isnan(xirr) or abs(xirr) > 10:
                    xirr = total_return ** (12 / len(monthly_contributions))
            else:
                xirr = 0
        except:
            months = len(monthly_contributions)
            xirr = (final_value / total_contributions) ** (12 / months) - 1 if months > 0 else 0
        
        # Risk metrics - FIXED volatility calculation
        if len(portfolio_values) > 12:
            # Use daily returns but exclude contribution effects
            daily_returns = []
            
            for i in range(1, len(portfolio_values)):
                current_date = portfolio_values.index[i]
                prev_value = portfolio_values.iloc[i-1]
                current_value = portfolio_values.iloc[i]
                
                # Check if there was a contribution on this date
                contribution = 0
                if current_date in monthly_contributions.index:
                    contribution = monthly_contributions[current_date]
                
                # Calculate return excluding contribution effect
                if prev_value > 0:
                    daily_return = (current_value - prev_value - contribution) / prev_value
                    # Filter out extreme values
                    if -0.5 < daily_return < 1.0:
                        daily_returns.append(daily_return)
            
            if len(daily_returns) > 50:
                daily_returns_series = pd.Series(daily_returns)
                volatility = daily_returns_series.std() * np.sqrt(252)  # Annualize
            else:
                volatility = 0
            
            # Sharpe ratio
            sharpe = (xirr - 0.04) / volatility if volatility > 0 and not np.isnan(volatility) else 0
            
            # Maximum drawdown
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            volatility = 0
            sharpe = 0
            max_drawdown = 0
        
        return {
            'Total Contributions': total_contributions,
            'Final Value': final_value,
            'Total Return': total_return,
            'XIRR': xirr,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown
        }
    
    def _get_default_metrics(self) -> Dict:
        """Return default metrics when calculation fails"""
        return {
            'Total Contributions': 0.0,
            'Final Value': 0.0,
            'Total Return': 0.0,
            'XIRR': 0.0,
            'Volatility': 0.0,
            'Sharpe Ratio': 0.0,
            'Max Drawdown': 0.0
        }
    
    def backtest_user_allocation(self, data: pd.DataFrame, user_allocation: Dict[str, float]) -> Dict:
        """Backtest user's current allocation strategy"""
        
        print(f"🔍 Debug: User allocation input: {user_allocation}")
        
        # Map user allocation to available tickers with better matching
        ticker_weights = {}
        total_weight = 0
        
        # Enhanced mapping with exact matches first, then partial matches
        ticker_mapping = {
            # Exact matches for all possible assets
            'VOO (S&P 500)': 'VOO', 'QQQ (Nasdaq 100)': 'QQQ', 'VTI (Total US Market)': 'VTI',
            'IWM (Russell 2000)': 'IWM', 'ARKK (Innovation)': 'ARKK',
            'VWO (Emerging Markets)': 'VWO', 'VEA (Developed Markets)': 'VEA',
            'VXUS (Total International)': 'VXUS', 'EFA (EAFE)': 'EFA', 'VGK (European)': 'VGK',
            'BND (Total Bond Market)': 'BND', 'TLT (Long-Term Treasury)': 'TLT',
            'SHY (Short-Term Treasury)': 'SHY', 'LQD (Investment Grade Corp)': 'LQD',
            'HYG (High Yield Corp)': 'HYG',
            'Bitcoin (BTC)': 'BTC-USD', 'Ethereum (ETH)': 'ETH-USD',
            'Cardano (ADA)': 'ADA-USD', 'Solana (SOL)': 'SOL-USD',
            'XLK (Technology)': 'XLK', 'XLF (Financial)': 'XLF', 'XLV (Healthcare)': 'XLV',
            'XLE (Energy)': 'XLE', 'XLI (Industrial)': 'XLI',
            'GLD (Gold)': 'GLD', 'SLV (Silver)': 'SLV', 'VNQ (Real Estate)': 'VNQ',
            'USO (Oil)': 'USO', 'DBA (Agriculture)': 'DBA',
            # Direct ticker matches
            'VOO': 'VOO', 'QQQ': 'QQQ', 'VWO': 'VWO', 'VTI': 'VTI', 'BND': 'BND',
            'BTC-USD': 'BTC-USD', 'ETH-USD': 'ETH-USD', 'TLT': 'TLT', 'GLD': 'GLD'
        }
        
        # First try exact matches
        for investment, weight in user_allocation.items():
            if weight > 0:  # Only process non-zero allocations
                if investment in ticker_mapping:
                    ticker = ticker_mapping[investment]
                    if ticker in self.available_tickers:
                        ticker_weights[ticker] = weight / 100  # Convert percentage to decimal
                        total_weight += weight / 100
                        print(f"✅ Mapped '{investment}' -> '{ticker}' ({weight}%)")
                    else:
                        print(f"⚠️ Ticker '{ticker}' not in available tickers for '{investment}'")
                else:
                    # Try partial matching for unmapped investments
                    mapped = False
                    for key, ticker in ticker_mapping.items():
                        if (key.lower() in investment.lower() or investment.lower() in key.lower()) and ticker in self.available_tickers:
                            ticker_weights[ticker] = weight / 100
                            total_weight += weight / 100
                            print(f"✅ Partial match '{investment}' -> '{ticker}' ({weight}%)")
                            mapped = True
                            break
                    if not mapped:
                        print(f"❌ Could not map '{investment}' to any available ticker")
        
        print(f"🎯 Final ticker weights: {ticker_weights}")
        print(f"📊 Total weight: {total_weight:.2%}")
        
        # Normalize weights if needed
        if total_weight > 0:
            if abs(total_weight - 1.0) > 0.01:  # If not close to 100%
                print(f"⚖️ Normalizing weights from {total_weight:.2%} to 100%")
                ticker_weights = {k: v / total_weight for k, v in ticker_weights.items()}
        else:
            print("⚠️ No valid allocations found, using equal weight fallback")
            # Equal weight fallback
            ticker_weights = {ticker: 1/len(self.available_tickers) for ticker in self.available_tickers}
        
        # Simple SIP backtest with fixed allocation
        portfolio_values = pd.Series(index=data.index, dtype=float)
        monthly_contributions = pd.Series(dtype=float)
        shares_held = {ticker: 0 for ticker in self.available_tickers}
        
        monthly_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq='MS')
        monthly_dates = monthly_dates[monthly_dates <= data.index[-1]]
        
        for date in data.index:
            if date in monthly_dates:
                monthly_contributions[date] = self.sip_amount
                
                # Invest according to user allocation
                for ticker in self.available_tickers:
                    if ticker in ticker_weights and ticker in data.columns:
                        weight = ticker_weights[ticker]
                        shares_to_buy = (self.sip_amount * weight) / data.loc[date, ticker]
                        shares_held[ticker] += shares_to_buy
            
            # Calculate portfolio value
            portfolio_value = sum([shares_held[ticker] * data.loc[date, ticker] 
                                for ticker in self.available_tickers if ticker in data.columns])
            portfolio_values[date] = portfolio_value
        
        # Forward fill portfolio values to handle non-SIP dates
        portfolio_values = portfolio_values.ffill().fillna(0)
        
        return {
            'portfolio_values': portfolio_values,
            'monthly_contributions': monthly_contributions,
            'strategy_name': 'User Allocation'
        }
    
    def backtest_benchmark(self, data: pd.DataFrame) -> Dict:
        """S&P 500 benchmark strategy"""
        # Use VOO as S&P 500 proxy
        if 'VOO' in data.columns:
            sp500_data = data['VOO'].dropna()
        else:
            # Fallback to downloading S&P 500 index
            try:
                sp500_data = yf.download("^GSPC", start=data.index[0], end=data.index[-1], progress=False)["Close"]
                sp500_data = sp500_data.ffill().bfill().dropna()
            except:
                # Last resort - use first column of data
                sp500_data = data.iloc[:, 0]
        
        portfolio_values = pd.Series(index=data.index, dtype=float)
        monthly_contributions = pd.Series(dtype=float)
        
        monthly_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq='MS')
        monthly_dates = monthly_dates[monthly_dates <= data.index[-1]]
        
        shares_held = 0
        
        for date in data.index:
            if date in monthly_dates:
                monthly_contributions[date] = self.sip_amount
                
                # Get S&P 500 price for this date
                if date in sp500_data.index:
                    price = sp500_data.loc[date]
                else:
                    # Find closest available price
                    available_dates = sp500_data.index[sp500_data.index <= date]
                    if len(available_dates) > 0:
                        price = sp500_data.loc[available_dates[-1]]
                    else:
                        available_dates = sp500_data.index[sp500_data.index > date]
                        if len(available_dates) > 0:
                            price = sp500_data.loc[available_dates[0]]
                        else:
                            continue
                
                if price > 0:
                    shares_held += self.sip_amount / price
            
            # Calculate portfolio value
            if date in sp500_data.index:
                current_price = sp500_data.loc[date]
            else:
                # Find closest available price
                available_dates = sp500_data.index[sp500_data.index <= date]
                if len(available_dates) > 0:
                    current_price = sp500_data.loc[available_dates[-1]]
                else:
                    available_dates = sp500_data.index[sp500_data.index > date]
                    if len(available_dates) > 0:
                        current_price = sp500_data.loc[available_dates[0]]
                    else:
                        current_price = sp500_data.iloc[0] if len(sp500_data) > 0 else 1000
            
            portfolio_values[date] = shares_held * current_price
        
        # Forward fill portfolio values to handle non-SIP dates
        portfolio_values = portfolio_values.ffill().fillna(0)
        
        return {
            'portfolio_values': portfolio_values,
            'monthly_contributions': monthly_contributions,
            'strategy_name': 'S&P 500 Benchmark'
        }
    
    def run_comprehensive_analysis(self, start_date: str, end_date: str, 
                                 user_allocation: Optional[Dict[str, float]] = None) -> Dict:
        """Run comprehensive SIP analysis including user allocation comparison"""
        
        # Download data
        data = self.download_data(start_date, end_date)
        
        # Define strategies to test
        strategies = [
            {'name': 'Static Equal Weight', 'rebalance_months': 12, 'method': 'equal_weight'},
            {'name': 'Max Sharpe (3M)', 'rebalance_months': 3, 'method': 'max_sharpe'},
            {'name': 'Max Sharpe (6M)', 'rebalance_months': 6, 'method': 'max_sharpe'},
            {'name': 'Min Variance (3M)', 'rebalance_months': 3, 'method': 'min_variance'},
            {'name': 'Min Variance (6M)', 'rebalance_months': 6, 'method': 'min_variance'},
        ]
        
        results = {}
        
        # Backtest each strategy
        for strategy in strategies:
            if strategy['method'] == 'equal_weight':
                results[strategy['name']] = self.backtest_equal_weight(data)
            else:
                results[strategy['name']] = self.backtest_strategy(data, strategy)
        
        # Add user allocation if provided
        if user_allocation:
            results['Your Current Allocation'] = self.backtest_user_allocation(data, user_allocation)
        
        # Add benchmark (S&P 500)
        results['S&P 500 Benchmark'] = self.backtest_benchmark(data)
        
        # Calculate performance metrics for all strategies
        performance_summary = {}
        for name, result in results.items():
            metrics = self.calculate_performance_metrics(
                result['portfolio_values'], 
                result['monthly_contributions']
            )
            performance_summary[name] = metrics
        
        return {
            'results': results,
            'performance_summary': performance_summary,
            'data': data
        }
    
    def backtest_equal_weight(self, data: pd.DataFrame) -> Dict:
        """Simple equal weight strategy for comparison"""
        portfolio_values = pd.Series(index=data.index, dtype=float)
        monthly_contributions = pd.Series(dtype=float)
        shares_held = np.zeros(len(self.available_tickers))
        
        monthly_dates = pd.date_range(start=data.index[0], end=data.index[-1], freq='MS')
        monthly_dates = monthly_dates[monthly_dates <= data.index[-1]]
        
        equal_weights = np.ones(len(self.available_tickers)) / len(self.available_tickers)
        
        for date in data.index:
            if date in monthly_dates:
                monthly_contributions[date] = self.sip_amount
                
                for j in range(len(self.available_tickers)):
                    shares_to_buy = (self.sip_amount * equal_weights[j]) / data.loc[date, self.available_tickers[j]]
                    shares_held[j] += shares_to_buy
            
            portfolio_value = sum([shares_held[j] * data.loc[date, self.available_tickers[j]] 
                                for j in range(len(self.available_tickers))])
            portfolio_values[date] = portfolio_value
        
        # Forward fill portfolio values to handle non-SIP dates
        portfolio_values = portfolio_values.ffill().fillna(0)
        
        return {
            'portfolio_values': portfolio_values,
            'monthly_contributions': monthly_contributions,
            'strategy_name': 'Static Equal Weight'
        }
    
    def create_interactive_charts(self, analysis_results: Dict) -> Dict[str, go.Figure]:
        """Create interactive Plotly charts for web display"""
        
        results = analysis_results['results']
        performance_summary = analysis_results['performance_summary']
        
        charts = {}
        
        # 1. Portfolio Growth Chart
        fig_growth = go.Figure()
        
        # Create a more comprehensive color mapping with distinct colors for each strategy
        color_palette = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange  
            '#2ca02c',  # Green
            '#d62728',  # Red
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#aec7e8',  # Light Blue
            '#ffbb78',  # Light Orange
            '#98df8a',  # Light Green
            '#ff9896',  # Light Red
            '#c5b0d5',  # Light Purple
            '#c49c94',  # Light Brown
            '#f7b6d3',  # Light Pink
            '#c7c7c7',  # Light Gray
            '#dbdb8d',  # Light Olive
            '#9edae5'   # Light Cyan
        ]
        
        # Assign colors to strategies dynamically
        colors = {}
        strategy_names = list(results.keys())
        for i, strategy_name in enumerate(strategy_names):
            colors[strategy_name] = color_palette[i % len(color_palette)]
        
        # Ensure benchmark and custom allocation have specific colors
        if 'S&P 500 Benchmark' in colors:
            colors['S&P 500 Benchmark'] = '#e377c2'  # Pink for benchmark
        if 'Your Custom Allocation' in colors:
            colors['Your Custom Allocation'] = '#8c564b'  # Brown for custom
        
        for strategy_name, result in results.items():
            portfolio_values = result['portfolio_values'].dropna()
            if len(portfolio_values) > 0:
                final_value = performance_summary[strategy_name]['Final Value']
                xirr = performance_summary[strategy_name]['XIRR']
                
                fig_growth.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values.values,
                    mode='lines',
                    name=f'{strategy_name} (${final_value:,.0f}, {xirr:.1%} XIRR)',
                    line=dict(color=colors.get(strategy_name, '#17becf'), width=3),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                'Date: %{x}<br>' +
                                'Value: $%{y:,.0f}<extra></extra>'
                ))
        
        # Add cumulative contributions line (use any strategy's contributions as they're all the same)
        contributions_strategy = None
        for strategy_name in results.keys():
            if 'monthly_contributions' in results[strategy_name]:
                contributions_strategy = strategy_name
                break
        
        if contributions_strategy:
            contributions = results[contributions_strategy]['monthly_contributions'].cumsum()
            fig_growth.add_trace(go.Scatter(
                x=contributions.index,
                y=contributions.values,
                mode='lines',
                name='💰 Total Contributions (Your Money In)',
                line=dict(color='white', width=3, dash='dash', shape='hv'),  # 'hv' creates step function
                hovertemplate='<b>💰 Total Contributions</b><br>' +
                            'Date: %{x}<br>' +
                            'Amount: $%{y:,.0f}<br>' +
                            '<i>This is how much you\'ve invested so far</i><extra></extra>'
            ))
        
        fig_growth.update_layout(
            title='🚀 SIP Portfolio Growth: See How Your Monthly Investment Grows!',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            hovermode='x unified',
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
            height=600
        )
        
        charts['growth'] = fig_growth
        
        # 2. Returns Comparison Chart
        fig_returns = go.Figure()
        
        strategy_names = []
        xirr_returns = []
        colors_list = []
        
        # Sort by performance for better visualization
        sorted_strategies = sorted(performance_summary.items(), 
                                 key=lambda x: x[1]['XIRR'] if not np.isnan(x[1]['XIRR']) else -999, 
                                 reverse=True)
        
        for name, metrics in sorted_strategies:
            if not np.isnan(metrics['XIRR']):
                strategy_names.append(name)
                xirr_returns.append(metrics['XIRR'] * 100)  # Convert to percentage
                colors_list.append(colors.get(name, '#17becf'))
        
        fig_returns.add_trace(go.Bar(
            x=strategy_names,
            y=xirr_returns,
            marker_color=colors_list,
            text=[f'{val:.1f}%' for val in xirr_returns],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>' +
                        'Annual Return: %{y:.1f}%<extra></extra>'
        ))
        
        fig_returns.update_layout(
            title='📊 Annual Returns: Which Strategy Wins?',
            xaxis_title='Strategy',
            yaxis_title='Annual Return (XIRR) %',
            height=500,
            showlegend=False
        )
        
        charts['returns'] = fig_returns
        
        # 3. Risk vs Return Scatter Plot
        fig_risk_return = go.Figure()
        
        # Collect data for scatter plot
        for name, metrics in performance_summary.items():
            if (not np.isnan(metrics['Volatility']) and not np.isnan(metrics['XIRR']) 
                and metrics['Volatility'] > 0):
                volatility = metrics['Volatility'] * 100  # Convert to percentage
                xirr = metrics['XIRR'] * 100  # Convert to percentage
                
                fig_risk_return.add_trace(go.Scatter(
                    x=[volatility],
                    y=[xirr],
                    mode='markers+text',
                    name=name,
                    marker=dict(
                        size=15,
                        color=colors.get(name, '#17becf'),
                        line=dict(width=2, color='black')
                    ),
                    text=[name],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>' +
                                'Risk: %{x:.1f}%<br>' +
                                'Return: %{y:.1f}%<extra></extra>'
                ))
        
        fig_risk_return.update_layout(
            title='⚖️ Risk vs Return: Find Your Sweet Spot!',
            xaxis_title='Risk (Volatility) % - Lower is Better',
            yaxis_title='Return (XIRR) % - Higher is Better',
            height=500,
            showlegend=False
        )
        
        charts['risk_return'] = fig_risk_return
        
        return charts
    
    def get_strategy_recommendations(self, performance_summary: Dict) -> Dict[str, str]:
        """Get strategy recommendations based on performance metrics"""
        
        valid_strategies = {name: metrics for name, metrics in performance_summary.items() 
                           if not np.isnan(metrics['XIRR']) and not np.isnan(metrics['Volatility'])}
        
        recommendations = {}
        
        if valid_strategies:
            # Best return
            best_return = max(valid_strategies.items(), key=lambda x: x[1]['XIRR'])
            recommendations['Best Return'] = f"{best_return[0]} ({best_return[1]['XIRR']:.1%} XIRR)"
            
            # Best risk-adjusted return (Sharpe ratio)
            best_sharpe = max(valid_strategies.items(), key=lambda x: x[1]['Sharpe Ratio'])
            recommendations['Best Risk-Adjusted'] = f"{best_sharpe[0]} (Sharpe: {best_sharpe[1]['Sharpe Ratio']:.2f})"
            
            # Lowest risk
            lowest_risk = min(valid_strategies.items(), key=lambda x: x[1]['Volatility'])
            recommendations['Lowest Risk'] = f"{lowest_risk[0]} (Vol: {lowest_risk[1]['Volatility']:.1%})"
        
        return recommendations
    
    def get_optimal_allocation_weights(self, data: pd.DataFrame, method: str = "min_variance") -> Optional[Dict[str, float]]:
        """Get the actual optimal allocation weights from the best strategy."""
        
        try:
            # Use the most recent 2 years of data for optimization
            recent_data = data.tail(504)  # Approximately 2 years of daily data
            returns_data = recent_data.pct_change().dropna()
            
            if len(returns_data) < 252:  # Need at least 1 year
                return None
            
            # Detect current market regime
            regime = self.detect_market_regime(returns_data)
            
            # Get optimal weights
            optimal_weights = self.optimize_portfolio(returns_data, regime, method)
            
            if optimal_weights is not None:
                # Convert to dictionary with asset names
                allocation_dict = {}
                for i, ticker in enumerate(self.available_tickers):
                    if i < len(optimal_weights):
                        allocation_dict[ticker] = float(optimal_weights[i])
                
                return allocation_dict
            
        except Exception as e:
            print(f"Error getting optimal allocation: {e}")
        
        return None
