"""
SQLite Database Client Module
Simple client for accessing local market data database
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class LocalDataClient:
    """Configurable data client for financial data retrieval (SQLite DB or CSV)"""
    
    def __init__(self, data_source: str = "csv", db_path: str = "data/market_data.db", 
                 csv_path: str = "data/ticker_data_v2.csv"):
        """
        Initialize data client
        
        Args:
            data_source: 'csv' or 'db' - determines data source
            db_path: Path to SQLite database file
            csv_path: Path to CSV file
        """
        self.data_source = data_source.lower()
        self.db_path = db_path
        self.csv_path = csv_path
        self._csv_data = None  # Cache for CSV data
        
        # Validate data source
        if self.data_source not in ['csv', 'db']:
            raise ValueError("data_source must be 'csv' or 'db'")
        
        # Check if files exist
        if self.data_source == 'db' and not os.path.exists(self.db_path):
            raise FileNotFoundError(f"Database file not found: {self.db_path}")
        elif self.data_source == 'csv' and not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        print(f"📊 LocalDataClient initialized with data_source='{self.data_source}'")
        
        # Default exchange rates (can be updated as needed)
        # Removed exchange rates - not needed for pure MPT optimization
    
    def _load_csv_data(self) -> pd.DataFrame:
        """Load and cache CSV data"""
        if self._csv_data is None:
            print(f"📊 Loading CSV data from {self.csv_path}...")
            self._csv_data = pd.read_csv(self.csv_path)
            # Handle different date formats that might exist in CSV
            try:
                self._csv_data['Date'] = pd.to_datetime(self._csv_data['Date'], format='mixed')
            except:
                # Fallback to standard parsing
                self._csv_data['Date'] = pd.to_datetime(self._csv_data['Date'])
            print(f"✅ Loaded {len(self._csv_data)} rows, {len(self._csv_data['Ticker'].unique())} unique tickers")
        return self._csv_data
    
    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers"""
        if self.data_source == 'csv':
            csv_data = self._load_csv_data()
            return sorted(csv_data['Ticker'].unique().tolist())
        else:  # db
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT DISTINCT Ticker FROM market_data ORDER BY Ticker"
                result = pd.read_sql_query(query, conn)
                return result['Ticker'].tolist()
    
    def get_prices(self, ticker: str, start_date: str = None, end_date: str = None, 
                   lookback_days: int = 365) -> pd.DataFrame:
        """Get historical price data for a single ticker"""
        try:
            # Calculate date range if not provided
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=lookback_days)
                start_date = start_dt.strftime('%Y-%m-%d')
            
            if self.data_source == 'csv':
                csv_data = self._load_csv_data()
                # Filter for specific ticker and date range
                ticker_data = csv_data[
                    (csv_data['Ticker'] == ticker) & 
                    (csv_data['Date'] >= start_date) & 
                    (csv_data['Date'] <= end_date)
                ].copy()
                
                if ticker_data.empty:
                    print(f"No data found for {ticker} between {start_date} and {end_date}")
                    return pd.DataFrame()
                
                # Set date as index and select relevant columns
                ticker_data = ticker_data.set_index('Date')
                # Select columns that exist in CSV (handle Adj Close vs Close)
                available_cols = ticker_data.columns.tolist()
                cols_to_select = []
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in available_cols:
                        cols_to_select.append(col)
                
                df = ticker_data[cols_to_select].copy()
                # Rename columns to match yfinance format (lowercase)
                df.columns = [col.lower() for col in df.columns]
                
                return df
            
            else:  # db
                with sqlite3.connect(self.db_path) as conn:
                    query = """
                    SELECT Date, Open, High, Low, Close, Volume
                    FROM market_data 
                    WHERE Ticker = ? AND Date >= ? AND Date <= ?
                    ORDER BY Date
                    """
                    
                    df = pd.read_sql_query(query, conn, params=[ticker, start_date, end_date])
                    
                    if df.empty:
                        print(f"No data found for {ticker} between {start_date} and {end_date}")
                        return pd.DataFrame()
                    
                    # Convert date to datetime and set as index
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                    
                    # Rename columns to match yfinance format (lowercase)
                    df.columns = [col.lower() for col in df.columns]
                    
                    return df
                
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def download_multiple_tickers(self, tickers: List[str], start_date: str, end_date: str, 
                                 progress: bool = False) -> pd.DataFrame:
        """Download data for multiple tickers"""
        all_data = {}
        failed_tickers = []
        
        print(f"📊 Loading data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        if self.data_source == 'csv':
            csv_data = self._load_csv_data()
            # Filter data for date range
            date_filtered = csv_data[
                (csv_data['Date'] >= start_date) & 
                (csv_data['Date'] <= end_date)
            ]
            
            for ticker in tickers:
                try:
                    ticker_data = date_filtered[date_filtered['Ticker'] == ticker].copy()
                    
                    if not ticker_data.empty:
                        ticker_data = ticker_data.set_index('Date')
                        # Remove duplicate dates by keeping the last occurrence
                        ticker_data = ticker_data[~ticker_data.index.duplicated(keep='last')]
                        all_data[ticker] = ticker_data['Close']
                        print(f"✅ {ticker}: {len(ticker_data)} data points")
                    else:
                        print(f"❌ {ticker}: No data in date range")
                        failed_tickers.append(ticker)
                        
                except Exception as e:
                    print(f"❌ {ticker}: Error - {str(e)[:50]}...")
                    failed_tickers.append(ticker)
        
        else:  # db
            with sqlite3.connect(self.db_path) as conn:
                for ticker in tickers:
                    try:
                        query = """
                        SELECT Date, Close
                        FROM market_data 
                        WHERE Ticker = ? AND Date >= ? AND Date <= ?
                        ORDER BY Date
                        """
                        
                        df = pd.read_sql_query(query, conn, params=[ticker, start_date, end_date])
                        
                        if not df.empty:
                            df['Date'] = pd.to_datetime(df['Date'])
                            df = df.set_index('Date')
                            # Remove duplicate dates by keeping the last occurrence
                            df = df[~df.index.duplicated(keep='last')]
                            all_data[ticker] = df['Close']
                            print(f"✅ {ticker}: {len(df)} data points")
                        else:
                            print(f"❌ {ticker}: No data in date range")
                            failed_tickers.append(ticker)
                            
                    except Exception as e:
                        print(f"❌ {ticker}: Error - {str(e)[:50]}...")
                        failed_tickers.append(ticker)
        
        if not all_data:
            print("No data loaded for any ticker")
            return pd.DataFrame()
        
        # Combine all data into single DataFrame
        combined_df = pd.DataFrame(all_data)
        
        # Forward fill missing data and drop rows with all NaN
        combined_df = combined_df.ffill().bfill()
        combined_df = combined_df.dropna(how='all')
        
        if failed_tickers:
            print(f"⚠️ No data available for: {failed_tickers}")
        
        print(f"✅ Successfully loaded data for {len(combined_df.columns)} tickers")
        return combined_df
    
    # Removed exchange rate functionality - not needed for MPT optimization
    
    def validate_tickers(self, tickers: List[str]) -> Tuple[List[str], List[str]]:
        """Validate ticker availability"""
        available_tickers = set(self.get_available_tickers())
        
        valid_tickers = []
        invalid_tickers = []
        
        data_source_name = "CSV file" if self.data_source == 'csv' else "database"
        print(f"🔍 Validating {len(tickers)} tickers against local {data_source_name}...")
        
        for ticker in tickers:
            if ticker in available_tickers:
                valid_tickers.append(ticker)
                print(f"✅ {ticker} - Valid")
            else:
                invalid_tickers.append(ticker)
                print(f"❌ {ticker} - Not found in {data_source_name}")
        
        print(f"📊 Validation complete: {len(valid_tickers)} valid, {len(invalid_tickers)} invalid")
        return valid_tickers, invalid_tickers
    
    def get_sp500_data(self, start_date: str, end_date: str) -> pd.Series:
        """Get S&P 500 data for benchmarking"""
        # Try S&P 500 index first, then ETF proxies
        sp500_options = ['^GSPC', 'SPY', 'VOO', 'IVV']
        
        for ticker in sp500_options:
            try:
                data = self.get_prices(ticker, start_date, end_date)
                if not data.empty:
                    print(f"📈 Using {ticker} as S&P 500 benchmark")
                    return data['close']
            except:
                continue
        
        data_source_name = "CSV file" if self.data_source == 'csv' else "database"
        print(f"⚠️ No S&P 500 data found in {data_source_name}")
        return pd.Series()
    
    def get_historical_data_with_cache(self, tickers: List[str], start_date: str, end_date: str, 
                                     use_cache: bool = True) -> pd.DataFrame:
        """Get historical data (SQLite is already fast, so cache is less critical)"""
        return self.download_multiple_tickers(tickers, start_date, end_date)

# Global instance for easy import - configured for CSV by default
local_data_client = LocalDataClient(data_source="csv")
