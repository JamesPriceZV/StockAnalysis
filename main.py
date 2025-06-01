import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Expanded Nasdaq-100 stock list
NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX',
    'CMCSA', 'CSCO', 'INTC', 'PEP', 'COST', 'TMUS', 'AMGN', 'CHTR', 'SBUX', 'AMD',
    'META', 'AVGO', 'TXN', 'QCOM', 'HON', 'INTU', 'BKNG', 'ISRG', 'GILD', 'MU'
]

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_stock_data(stocks, period='5y'):
    """Download stock data with caching."""
    try:
        data = yf.download(stocks, period=period, group_by='ticker', auto_adjust=True, threads=True)
        if data.empty:
            raise ValueError("No data downloaded")
        return data
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        raise

def calculate_likelihood_metrics(df, past_30_days, past_365_days):
    """Calculate likelihood metrics for a single stock."""
    # Calculate daily returns
    daily_returns = df['Close'].pct_change().dropna()
    
    # Short-term likelihood (last 30 days)
    recent_returns = daily_returns[daily_returns.index >= past_30_days]
    short_term_likelihood = (recent_returns > 0).mean() if len(recent_returns) > 0 else np.nan
    
    # Long-term likelihood (weekly returns for last 52 weeks)
    weekly_closes = df['Close'].resample('W').last().dropna()
    weekly_returns = weekly_closes.pct_change().dropna()
    weekly_recent = weekly_returns[weekly_returns.index >= past_365_days]
    long_term_likelihood = (weekly_recent > 0).mean() if len(weekly_recent) > 0 else np.nan
    
    return short_term_likelihood, long_term_likelihood

def analyze_stocks(stocks):
    """Analyze stocks for gain likelihood with optimized processing
    """
    try:
        data = download_stock_data(stocks)
    except Exception as e:
        st.error(f"Failed to download stock data. Please check symbols and connection. Error: {e}")
        return pd.DataFrame()

    results = []
    today = datetime.today()
    past_30_days = today - timedelta(days=30)
    past_365_days = today - timedelta(days=365)

    for stock in stocks:
        try:
            # Extract stock-specific data
            if isinstance(data.columns, pd.MultiIndex):
                df = data[stock]  # Select data for the current stock
            else:
                df = data  # Use the entire data if it's not a MultiIndex

            short_term_likelihood, long_term_likelihood = calculate_likelihood_metrics(df, past_30_days, past_365_days)

            if not (np.isnan(short_term_likelihood) or np.isnan(long_term_likelihood)):
                results.append({
                    'symbol': stock,
                    'short_term_likelihood': short_term_likelihood,
                    'long_term_likelihood': long_term_likelihood
                })
        except Exception as e:
            st.warning(f"Error processing {stock}: {e}")
            logger.error(f"Error processing {stock}: {e}")
            continue

    results_df = pd.DataFrame(results)
    return results_df

# Streamlit app layout
st.title("Nasdaq-100 Stock Gain Likelihood Analyzer")

st.markdown("""
This app analyzes Nasdaq-100 stocks to identify those statistically likely to experience short-term and long-term gains based on historical data:
- **Short-term Likelihood**: Proportion of days with positive returns in the past 30 days.
- **Long-term Likelihood**: Proportion of weeks with positive returns in the past 52 weeks.
""")

# Fetch and analyze data
results_df = analyze_stocks(NASDAQ_100)

# Check if we have data to display
if not results_df.empty:
    # Interactive Scatter Plot
    fig = px.scatter(
        results_df,
        x='short_term_likelihood',
        y='long_term_likelihood',
        text='symbol',
        title='Short-term vs Long-term Gain Likelihood',
        labels={
            'short_term_likelihood': 'Short-term Gain Likelihood (%)',
            'long_term_likelihood': 'Long-term Gain Likelihood (%)'
        },
        hover_data={'symbol': True, 'short_term_likelihood': ':.2%', 'long_term_likelihood': ':.2%'}
    )
    fig.update_traces(textposition='top center', marker=dict(size=8))

    # Highlight selected stock
    selected_stock = st.text_input("Enter a stock symbol to highlight (e.g., AAPL):", "").upper()
    if selected_stock in results_df['symbol'].values:
        selected_data = results_df[results_df['symbol'] == selected_stock]
        fig.add_scatter(
            x=selected_data['short_term_likelihood'],
            y=selected_data['long_term_likelihood'],
            mode='markers',
            marker=dict(color='red', size=12),
            name=selected_stock
        )
    elif selected_stock:
        st.warning(f"{selected_stock} is not in the Nasdaq-100 list.")

    st.plotly_chart(fig, use_container_width=True)

    # Display tables and other information only if we have data
    if len(results_df) > 0:
        # Data Table
        st.markdown("### Stock Data Table")
        results_df_display = results_df.copy()
        results_df_display['short_term_likelihood'] = results_df_display['short_term_likelihood'].apply(lambda x: f"{x:.2%}")
        results_df_display['long_term_likelihood'] = results_df_display['long_term_likelihood'].apply(lambda x: f"{x:.2%}")
        st.dataframe(results_df_display, use_container_width=True)

        # Top Performers
        st.markdown("### Recommendations")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Top 5 for Short-term Gains")
            top_short = results_df.nlargest(5, 'short_term_likelihood')[['symbol', 'short_term_likelihood']]
            top_short['short_term_likelihood'] = top_short['short_term_likelihood'].apply(lambda x: f"{x:.2%}")
            st.table(top_short)

        with col2:
            st.markdown("#### Top 5 for Long-term Gains")
            top_long = results_df.nlargest(5, 'long_term_likelihood')[['symbol', 'long_term_likelihood']]
            top_long['long_term_likelihood'] = top_long['long_term_likelihood'].apply(lambda x: f"{x:.2%}")
            st.table(top_long)

# Disclaimer
st.markdown("""
**Disclaimer**: This analysis uses historical data and does not guarantee future performance. It is not financial advice. Always conduct thorough research before investing.
""")