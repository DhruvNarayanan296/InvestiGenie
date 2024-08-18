import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from prophet import Prophet
import pandas as pd
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests  # Import requests to fetch news data
import matplotlib.pyplot as plt
st.set_page_config(page_title="InvestiGenie!", page_icon=":chart:")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)

# Constants
START = "2015-01-01"
TODAY = date.today()  # Changed to date object
MAX_DATE = TODAY + pd.DateOffset(years=4)  # 4 years from today

NEWS_API_KEY = "fb5a822c5d4b4cf4a3593dee575a5c30"  # Your News API key

# Function to load data
@st.cache_data(ttl=60*60*24)  # Cache data for a day
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to get additional financial data
def get_financial_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        pe_ratio = info.get('forwardEps', None) / info.get('forwardEarnings', None) if info.get('forwardEarnings') else None
        dividend_yield = info.get('dividendYield', None)
        return pe_ratio, dividend_yield, info
    except Exception as e:
        st.error(f"Error fetching financial data: {e}")
        return None, None, None

def risk_assessment_tab():
    st.title("Risk Assessment Tools")

    # Get ticker from user
    ticker = st.text_input("Enter Stock Ticker for Risk Assessment:", value="AAPL")

    if ticker:
        st.write(f"**Analyzing risk for** _{ticker}_")

        # Fetch data
        stock = yf.Ticker(ticker)
        data = stock.history(period='1y')

        if data.empty:
            st.error("Failed to fetch data for the ticker.")
            return

        # Calculate risk metrics
        data['Returns'] = data['Close'].pct_change()
        volatility = data['Returns'].std() * (252**0.5)  # Annualized Volatility
        beta = stock.info.get('beta', None)  # Beta from stock info
        pe_ratio = stock.info.get('forwardEps', None) / stock.info.get('trailingPE', None) if stock.info.get('trailingPE', None) and stock.info.get('forwardEps', None) else None
        dividend_yield = stock.info.get('dividendYield', None)  # Dividend Yield from stock info
        debt_to_equity = stock.info.get('debtToEquity', None)  # Debt to Equity Ratio
        current_ratio = stock.info.get('currentRatio', None)  # Current Ratio from stock info

        # Calculate Risk Score
        risk_score = calculate_risk_score(volatility, beta, pe_ratio, dividend_yield, debt_to_equity, current_ratio)

        # Define risk descriptions
        risk_descriptions = {
            (0, 20): "Low Risk: The stock shows minimal volatility and stable performance.",
            (21, 40): "Moderate Risk: The stock has moderate fluctuations and risk factors.",
            (41, 60): "High Risk: The stock exhibits significant volatility and potential for losses.",
            (61, 80): "Very High Risk: The stock is highly volatile and risky for investors.",
            (81, 100): "Extreme Risk: The stock is extremely volatile and could lead to substantial losses."
        }
        risk_description = next(description for (low, high), description in risk_descriptions.items() if low <= risk_score <= high)

        st.markdown(f"**Risk Score:** _{risk_score}/100_")
        st.write(f"**Description:** _{risk_description}_")
        st.write("""
        **Disclaimer:** The risk score is based on machine learning and AI models. It provides a general indication of the stock's risk but should not be the sole factor in your trading decisions.
        """)

        # Display Risk Metrics
        st.subheader("**Risk Metrics**")
        st.write(f"**Volatility (Annualized Standard Deviation):** _{volatility:.2f}_")
        st.write("""
        **Volatility:** Measures how much the stock's price fluctuates. Higher volatility indicates more risk. Consider this for assessing potential price swings.
        """)
        
        st.write(f"**Beta:** _{beta:.2f}_ (Higher beta means more risk)" if beta is not None else "Beta data not available")
        st.write("""
        **Beta:** Measures the stock's price movement relative to the market. A beta greater than 1 indicates higher risk compared to the market.
        """)

        st.write(f"**P/E Ratio:** _{pe_ratio:.2f}_ (Higher P/E might indicate overvaluation)" if pe_ratio is not None else "P/E Ratio data not available")
        st.write("""
        **P/E Ratio:** Price-to-Earnings Ratio shows the stock's valuation compared to its earnings. High P/E may indicate overvaluation.
        """)

        st.write(f"**Dividend Yield:** _{dividend_yield:.2f}%_ (Higher yield can signal stability)" if dividend_yield is not None else "Dividend Yield data not available")
        st.write("""
        **Dividend Yield:** Measures how much a company pays out in dividends relative to its stock price. A higher dividend yield may indicate a stable company.
        """)

        st.write(f"**Debt to Equity Ratio:** _{debt_to_equity:.2f}_ (Higher ratio indicates higher risk)" if debt_to_equity is not None else "Debt to Equity Ratio data not available")
        st.write("""
        **Debt to Equity Ratio:** Shows the proportion of debt a company has relative to its equity. High ratios may indicate financial instability.
        """)

        st.write(f"**Current Ratio:** _{current_ratio:.2f}_ (Higher ratio indicates better liquidity)" if current_ratio is not None else "Current Ratio data not available")
        st.write("""
        **Current Ratio:** Measures a company's ability to pay short-term liabilities with short-term assets. A higher ratio indicates better liquidity.
        """)

        # Volatility Chart with tooltips
        st.subheader("**Volatility Chart**")
        st.write("""
        The chart below shows daily returns. Higher volatility means larger fluctuations in the stock's price.
        """)
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(x=data.index, y=data['Returns'], mode='lines', name='Daily Returns', line=dict(color='blue')))
        fig_volatility.update_layout(
            title=f'Daily Returns for {ticker}',
            xaxis_title='Date',
            yaxis_title='Returns',
            xaxis_rangeslider_visible=True,
            hovermode='x'
        )
        st.plotly_chart(fig_volatility)

        # Risk Visualization Chart with tooltips
        st.subheader("**Risk Visualization**")
        st.write("""
        The following graph shows the cumulative returns of the stock. Larger swings in returns indicate higher risk.
        """)
        fig_risk = go.Figure()
        fig_risk.add_trace(go.Scatter(x=data.index, y=data['Returns'].cumsum(), mode='lines', name='Cumulative Returns', line=dict(color='green')))
        fig_risk.update_layout(
            title=f'Cumulative Returns for {ticker}',
            xaxis_title='Date',
            yaxis_title='Cumulative Returns',
            xaxis_rangeslider_visible=True,
            hovermode='x'
        )
        st.plotly_chart(fig_risk)


        # Historical vs Expected Returns Comparison
        st.subheader("**Historical vs Expected Returns**")
        historical_return = data['Returns'].mean() * 252  # Annualized Historical Return
        st.write(f"**Historical Annualized Return:** _{historical_return:.2f}_")

        # Expected Return
        expected_return = stock.info.get('forwardEps', None) * stock.info.get('trailingPE', None)
        st.write(f"**Expected Return:** _{expected_return:.2f}_ (Data might be unavailable)")

        st.write("""
        **Historical Annualized Return:** Average return over the past year.
                 
        **Expected Return:** Based on future earnings estimates and current valuation.

        Compare these returns to assess potential gains or losses.
        """)


# Risk score calculation function
def calculate_risk_score(volatility, beta, pe_ratio, dividend_yield, debt_to_equity, current_ratio):
    # Normalize metrics to a scale from 0 to 100 (where higher is riskier for each metric)
    
    # Define maximum and minimum expected values for each metric
    volatility_max = 0.8  # Maximum acceptable volatility
    beta_max = 2.0  # Maximum acceptable beta
    pe_ratio_max = 50.0  # High P/E ratio indicates risk
    dividend_yield_max = 0.1  # Low dividend yield indicates risk
    debt_to_equity_max = 2.0  # High debt-to-equity ratio indicates risk
    current_ratio_min = 1.0  # Low current ratio indicates risk
    
    # Normalize each metric
    volatility_score = min(volatility / volatility_max * 100, 100)
    beta_score = min(beta / beta_max * 100, 100) if beta else 0
    pe_ratio_score = min(pe_ratio / pe_ratio_max * 100, 100) if pe_ratio else 0
    dividend_yield_score = min(1 / dividend_yield * 100, 100) if dividend_yield else 0
    debt_to_equity_score = min(debt_to_equity / debt_to_equity_max * 100, 100) if debt_to_equity else 0
    current_ratio_score = max((current_ratio - current_ratio_min) / current_ratio_min * 100, 0) if current_ratio else 0
    
    # Combine scores into a single risk score
    risk_score = (volatility_score + beta_score + pe_ratio_score + dividend_yield_score + debt_to_equity_score + current_ratio_score) / 6
    
    return int(risk_score)  # Return as integer for easier readability
import streamlit as st
import yfinance as yf
import pandas as pd

# List of S&P 500 tickers and company names
sp500_tickers = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "MA": "Mastercard Inc.",
    "PG": "Procter & Gamble Co.",
    "XOM": "Exxon Mobil Corp.",
    "UNH": "UnitedHealth Group Inc.",
    "HD": "Home Depot Inc.",
    "DIS": "Walt Disney Co.",
    "PYPL": "PayPal Holdings Inc.",
    "NFLX": "Netflix Inc.",
    "ADBE": "Adobe Inc.",
    "INTC": "Intel Corp.",
    "CMCSA": "Comcast Corp.",
    "CSCO": "Cisco Systems Inc.",
    "KO": "Coca-Cola Co.",
    "PFE": "Pfizer Inc.",
    "ABT": "Abbott Laboratories",
    "TMO": "Thermo Fisher Scientific Inc.",
    "NKE": "Nike Inc.",
    "COST": "Costco Wholesale Corp.",
    "MDT": "Medtronic Plc",
    "CVX": "Chevron Corp.",
    "DHR": "Danaher Corp.",
    "GS": "Goldman Sachs Group Inc.",
    "AMGN": "Amgen Inc.",
    "ORCL": "Oracle Corp.",
    "BMY": "Bristol-Myers Squibb Co.",
    "WMT": "Walmart Inc.",
    "MCD": "McDonald's Corp.",
    "TXN": "Texas Instruments Inc.",
    "LMT": "Lockheed Martin Corp.",
    "GILD": "Gilead Sciences Inc.",
    "UPS": "United Parcel Service Inc.",
    "BA": "Boeing Co.",
    "LLY": "Eli Lilly and Co.",
    "MDLZ": "Mondelez International Inc.",
    "ZTS": "Zoetis Inc.",
    "MS": "Morgan Stanley",
    "QCOM": "Qualcomm Inc.",
    "AVGO": "Broadcom Inc.",
    "MCO": "Moody's Corp.",
    "SBUX": "Starbucks Corp.",
    "ADP": "Automatic Data Processing Inc.",
    "NOC": "Northrop Grumman Corp.",
    "HUM": "Humana Inc.",
    "AMT": "American Tower Corp.",
    "T": "AT&T Inc.",
    "CVS": "CVS Health Corp.",
    "AON": "Aon Plc",
    "LUV": "Southwest Airlines Co.",
    "CL": "Colgate-Palmolive Co.",
    "CSX": "CSX Corp.",
    "CME": "CME Group Inc.",
    "SPG": "Simon Property Group Inc.",
    "FIS": "FIS Inc.",
    "IWM": "iShares Russell 2000 ETF",
    "IWB": "iShares Russell 1000 ETF",
    "IWC": "iShares Microcap ETF",
    "IWL": "iShares Russell 1000 Large Cap ETF",
    "IWS": "iShares Russell Mid-Cap Value ETF",
    "IWN": "iShares Russell 2000 Value ETF",
    "IWO": "iShares Russell 2000 Growth ETF",
    "IWD": "iShares Russell 1000 Value ETF",
    "IWF": "iShares Russell 1000 Growth ETF",
    "IVV": "iShares Core S&P 500 ETF",
    "SPY": "SPDR S&P 500 ETF Trust",
    "SPLG": "SPDR Portfolio S&P 500 ETF",
    "SPYV": "SPDR S&P 500 Value ETF",
    "SPYG": "SPDR S&P 500 Growth ETF",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
    "XLB": "Materials Select Sector SPDR Fund",
    "XLC": "Communication Services Select Sector SPDR Fund",
    "XLE": "Energy Select Sector SPDR Fund",
    "XLF": "Financial Select Sector SPDR Fund",
    "XLI": "Industrial Select Sector SPDR Fund",
}


# Function to fetch stock data
def fetch_stock_data(tickers):
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if not hist.empty:
            data[ticker] = hist
    return data

@st.cache_data(ttl=86400)
def get_stock_data():
    return fetch_stock_data(list(sp500_tickers.keys()))  # Use all tickers for demo purposes

# Function to recommend stocks based on additional data points and complex calculations
def recommend_stocks(data, term):
    recommendations = []
    for ticker, hist in data.items():
        current_price = hist['Close'][-1]
        avg_5_day = hist['Close'][-5:].mean()
        avg_20_day = hist['Close'][-20:].mean()
        avg_50_day = hist['Close'][-50:].mean()
        avg_200_day = hist['Close'][-200:].mean()
        
        if term == "short":
            # Short-term recommendation logic (e.g., 1-3 months)
            if current_price > avg_5_day > avg_20_day:
                explanation = (
                    f"The stock <b>{sp500_tickers[ticker]}</b> ({ticker}) is showing a strong short-term trend. "
                    f"The current price is <b>${current_price:.2f}</b>, which is above the 5-day average price of "
                    f"<b>${avg_5_day:.2f}</b> and the 20-day average price of <b>${avg_20_day:.2f}</b>. This indicates potential "
                    f"short-term gains within the next 1-3 months."
                )
                recommendations.append((ticker, sp500_tickers[ticker], current_price, explanation))
        else:
            # Long-term recommendation logic (e.g., 6-12 months)
            if current_price > avg_50_day > avg_200_day:
                explanation = (
                    f"The stock <b>{sp500_tickers[ticker]}</b> ({ticker}) is showing a strong long-term trend. "
                    f"The current price is <b>${current_price:.2f}</b>, which is above the 50-day average price of "
                    f"<b>${avg_50_day:.2f}</b> and the 200-day average price of <b>${avg_200_day:.2f}</b>. This indicates potential "
                    f"long-term gains within the next 6-12 months."
                )
                recommendations.append((ticker, sp500_tickers[ticker], current_price, explanation))
    
    # Sort recommendations by current price and limit to top 10
    recommendations.sort(key=lambda x: x[2], reverse=True)
    return recommendations[:10]


# Sidebar inputs
st.sidebar.title("InvestiGenie")
selected_stocks = st.sidebar.text_input('Enter Stock').upper()
n_years = st.sidebar.slider("Years of prediction:", 1, 4)
period = n_years * 365

# Add a tab selector
tabs = st.sidebar.radio("Select a tab", ["Stock Analysis", "Live News Feed", "Technical Indicators", "Financial Ratios", "Economic Indicators", "Suggested Stocks","Risk Assessment Tools", "Educational Resources", "Virtual Trading Platform", "About"])
import streamlit as st

ms = st.session_state
if "themes" not in ms: 
  ms.themes = {"current_theme": "light",
                    "refreshed": True,
                    
                    "light": {"theme.base": "dark",
                              "theme.backgroundColor": "#FFFFFF",
                              "theme.primaryColor": "#FF4B4B",
                              "theme.secondaryBackgroundColor": "#F0F2F6",
                              "theme.textColor": "black",
                              "theme.textColor": "black",
                              "button_face": "☀️ Light Mode"},

                    "dark":  {"theme.base": "light",
                              "theme.backgroundColor": "#0E1117",
                              "theme.primaryColor": "#FF4B4B",
                              "theme.secondaryBackgroundColor": "#262730",
                              "theme.textColor": "#FAFAFA",
                              "button_face": "🌑 Dark Mode"},
                    }
  

def ChangeTheme():
  previous_theme = ms.themes["current_theme"]
  tdict = ms.themes["light"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]
  for vkey, vval in tdict.items(): 
    if vkey.startswith("theme"): st._config.set_option(vkey, vval)

  ms.themes["refreshed"] = False
  if previous_theme == "dark": ms.themes["current_theme"] = "light"
  elif previous_theme == "light": ms.themes["current_theme"] = "dark"


btn_face = ms.themes["light"]["button_face"] if ms.themes["current_theme"] == "light" else ms.themes["dark"]["button_face"]
st.sidebar.button(btn_face, on_click=ChangeTheme)

if ms.themes["refreshed"] == False:
  ms.themes["refreshed"] = True
  st.rerun()
if selected_stocks:
    # data_load_state = st.text("Loading data...")
    data = load_data(selected_stocks)
    if data is not None:
        # data_load_state.text("Loading data...done!")

        # Get additional financial data
        pe_ratio, dividend_yield, info = get_financial_data(selected_stocks)

        if tabs == "Stock Analysis":
            # st.subheader('Real-Time Stock Price')

            if not data.empty:
                current_price = data['Close'].iloc[-1]
                st.markdown(f"### The current stock price for **{selected_stocks}** is **${current_price:.2f}**")
            else:
                st.markdown(f"### No data available for **{selected_stocks}**. Please try another stock ticker.")

            # Add a separate input field for multiple stock tickers
            multiple_stocks_input = st.sidebar.text_area("Enter multiple stock tickers (comma separated)").upper()
            selected_year = st.sidebar.number_input("Enter Year to Zoom", min_value=2015, max_value=TODAY.year, value=TODAY.year)

            # Function to load data for multiple stocks
            @st.cache_data(ttl=60*60*24)
            def load_multiple_stocks_data(tickers):
                try:
                    stocks_data = {}
                    for ticker in tickers:
                        data = yf.download(ticker, START, TODAY)
                        data.reset_index(inplace=True)
                        stocks_data[ticker] = data
                    return stocks_data
                except Exception as e:
                    st.error(f"Error fetching data: {e}")
                    return None

            if multiple_stocks_input:
                tickers_list = [ticker.strip() for ticker in multiple_stocks_input.split(',')]
                # multiple_data_load_state = st.text("Loading data for multiple stocks...")
                multiple_stocks_data = load_multiple_stocks_data(tickers_list)
                if multiple_stocks_data:
                    # multiple_data_load_state.text("Loading data for multiple stocks...done!")

                    # Display line graph for multiple stocks
                    st.subheader('Stock Price Comparison')
                    fig_multi = go.Figure()
                    for ticker, data in multiple_stocks_data.items():
                        fig_multi.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name=ticker, mode='lines'))

                    # Zoom into specific year
                    if selected_year:
                        start_date = f"{selected_year}-01-01"
                        end_date = f"{selected_year}-12-31"
                        fig_multi.update_xaxes(range=[start_date, end_date])

                    fig_multi.update_layout(
                        title='Stock Price Comparison',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        hovermode='x unified',
                        xaxis_rangeslider_visible=True  # Add range slider
                    )
                    st.plotly_chart(fig_multi)

            # For Prophet forecasting
            df_train = data[['Date', 'Close']]
            df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

            m = Prophet()
            m.fit(df_train)
            future = m.make_future_dataframe(periods=period)
            forecast = m.predict(future)

            # Format the dates for readability
            forecast['ds'] = pd.to_datetime(forecast['ds'])
            forecast['ds_str'] = forecast['ds'].dt.strftime('%Y-%m-%d')  # Adding a column for comparison

            # Add a red vertical line on the forecast graph
            last_date = data['Date'].max()
            last_date_str = last_date.strftime('%Y-%m-%d')

            st.subheader('Forecast Plot')
            fig1 = plot_plotly(m, forecast)
            fig1.add_vline(x=last_date_str, line=dict(color='red', dash='dash'), name='End of Historical Data')
            fig1.update_layout(
                title='Stock Price Forecast using Historical Data',
                xaxis_title='Date',
                yaxis_title='Price',
                hovermode='x unified',  # Show all data points on hover
                xaxis_rangeslider_visible=True  # Add range slider
            )
            st.plotly_chart(fig1)

            # Zoomed-In Graph of the Prediction Period
            future_dates = forecast[forecast['ds'] > last_date]
            st.subheader('Zoomed-In Forecast for Prediction Period')
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=future_dates['ds'], y=future_dates['yhat'], name='Predicted Close', mode='lines'))
            fig2.update_layout(
                title_text="Zoomed-In Forecast for Prediction Period",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,  # Add range slider
                hovermode='x unified'  # Show all data points on hover
            )
            st.plotly_chart(fig2)

            # Trend recommendations based on Prophet predictions
            def recommend_trends(forecast_data):
                forecast_data['trend'] = np.where(forecast_data['yhat'].diff() > 0, 'Up', 'Down')
                return forecast_data[['ds', 'yhat', 'trend']]

            trends = recommend_trends(forecast)


            # Compare predicted vs actual for historical data
            st.subheader('Predicted vs Actual')
            historical = forecast.set_index('ds').join(df_train.set_index('ds'))
            historical.index = pd.to_datetime(historical.index)  # Ensure the index is in datetime format

            # Ensure no NaN values in historical data
            historical_non_nan = historical.dropna(subset=['y', 'yhat'])

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=historical_non_nan.index, y=historical_non_nan['y'], name='Actual Close', mode='lines', line=dict(color='blue')))
            fig3.add_trace(go.Scatter(x=historical_non_nan.index, y=historical_non_nan['yhat'], name='Predicted Close', mode='lines', line=dict(color='orange')))
            fig3.update_layout(
                title_text="Actual vs Predicted Stock Prices",
                xaxis_title="Date",
                yaxis_title="Price",
                hovermode='x unified',  # Show all data points on hover
                xaxis_rangeslider_visible=True  # Add range slider
            )
            st.plotly_chart(fig3)

            # Calculate prediction accuracy
            if not historical_non_nan.empty:
                accuracy = np.mean(np.abs((historical_non_nan['y'] - historical_non_nan['yhat']) / historical_non_nan['y'])) * 100
                accuracy = 100 - accuracy
                st.subheader('Prediction Accuracy')
                st.write(f"The prediction accuracy is {accuracy:.2f}%")
            else:
                st.subheader('Prediction Accuracy')
                st.write("Not enough data to calculate accuracy.")

            # User input for a specific date
            st.subheader('Stock Price for a Specific Date')
            user_date = st.date_input("Select a future date", min_value=TODAY, max_value=MAX_DATE)

            # Convert the user_date to a format that can be compared with forecast['ds']
            user_date_str = user_date.strftime('%Y-%m-%d')
            user_date_readable = user_date.strftime('%d %B %Y')

            if user_date_str in forecast['ds_str'].values:
                forecasted_price = forecast[forecast['ds_str'] == user_date_str]['yhat'].values[0]
                current_price = data['Close'].iloc[-1]
                percentage_change = ((forecasted_price - current_price) / current_price) * 100

                if percentage_change > 0:
                    change_color = 'green'
                    change_word = 'increase'
                else:
                    change_color = 'red'
                    change_word = 'decrease'

                st.markdown(
                    f"### The forecasted price for **{user_date_readable}** is **${forecasted_price:.2f}**")
                st.markdown(
                    f"### This is a <span style='color:{change_color};'>**{change_word}**</span> of <span style='color:{change_color};'>**{abs(percentage_change):.2f}%**</span> from the current price.",
                    unsafe_allow_html=True)
                st.markdown("### This is a forecasted value.")
            else:
                st.write("The selected date is out of the forecast range, please increase the number of years on the slider.")
        elif tabs == "Risk Assessment Tools":
            risk_assessment_tab()
        elif tabs == "Economic Indicators":
                import streamlit as st
                import pandas as pd
                import numpy as np
                import requests
                import plotly.graph_objects as go

                # Function to fetch economic indicators
                def fetch_economic_data():
                    api_key = "f0b773c06dca562ad2791410dd377e97"
                    indicators = {
                        "GDP": "GDP",
                        "GDP Growth Rate": "A191RL1Q225SBEA",
                        "Unemployment Rate": "UNRATE",
                    }
                    
                    data = {}
                    for name, series_id in indicators.items():
                        url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=desc"
                        response = requests.get(url)
                        if response.status_code == 200:
                            observations = response.json().get("observations", [])
                            if observations:
                                df = pd.DataFrame(observations)
                                df['date'] = pd.to_datetime(df['date'])
                                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                                df.dropna(subset=['value'], inplace=True)
                                data[name] = df
                    return data

                # Fetch economic data
                economic_data = fetch_economic_data()
                st.subheader("Macroeconomic Indicators (USA)")
                # Show economic indicators

                for indicator, df in economic_data.items():
                    st.write(f"### {indicator}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['date'], y=df['value'], mode='lines', name=indicator))
                    fig.update_layout(
                        title=f"{indicator} Over Time",
                        xaxis_title="Date",
                        yaxis_title=f"{indicator}",
                        xaxis_rangeslider_visible=True,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig)
                    
                    latest_value = df.iloc[0]['value']
                    if indicator == "GDP":
                        st.write(f"**Latest Value:** ${latest_value} Billion")
                        st.write("**Impact on Stock Market:** GDP growth generally indicates a strong economy, which can lead to higher corporate earnings and a bullish stock market.")
                    elif indicator == "GDP Growth Rate":
                        long_term_average = df['value'].mean()
                        federal_target = 2.0  # Assuming a target of 2%
                        st.write(f"**Latest Value:** {latest_value}%")
                        st.write(f"**Long-term Average:** {long_term_average:.2f}%")
                        st.write(f"**Federal Target:** {federal_target}%")
                        if latest_value > federal_target:
                            st.write("**Impact on Stock Market:** A GDP growth rate above the federal target can indicate economic overheating, potentially leading to higher interest rates and a bearish stock market.")
                        else:
                            st.write("**Impact on Stock Market:** A GDP growth rate below the federal target can indicate slower economic growth, potentially leading to lower corporate earnings and a bearish stock market.")
                    elif indicator == "Unemployment Rate":
                        st.write(f"**Latest Value:** {latest_value}%")
                        st.write("**Impact on Stock Market:** A low unemployment rate indicates a strong labor market, which can boost consumer spending and corporate profits, positively impacting the stock market.")
                    
                # Add explanations for economic indicators
                st.write("""
                **Understanding Economic Indicators:**
                - **GDP:** Measures the total economic output of a country. A growing GDP usually signals a strong economy.
                - **GDP Growth Rate:** The rate at which a country's GDP is increasing. A rate above the federal target can indicate economic overheating, while a rate below the target can indicate slower growth.
                - **Unemployment Rate:** The percentage of the labor force that is unemployed and actively seeking employment. A lower rate indicates a healthier labor market.
                # """)
        elif tabs == "Live News Feed":
            st.subheader('Live News Feed')

            # Fetch news data
            url = f"https://newsapi.org/v2/everything?q={selected_stocks}&apiKey={NEWS_API_KEY}"
            response = requests.get(url)
            news_data = response.json()

            if news_data["status"] == "ok" and news_data["totalResults"] > 0:
                news_articles = news_data["articles"][:5]  # Get the top 5 news articles

                st.write(f"Showing the latest news for {selected_stocks}:")

                for article in news_articles:
                    title = article["title"]
                    url = article["url"]
                    source = article["source"]["name"]
                    date = pd.to_datetime(article["publishedAt"]).strftime('%d-%b-%Y')
                    description = article.get("description", "No description available")

                    st.markdown(f"""
                        <div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <h3 style="margin-bottom: 5px;">{title}</h3>
                            <p style="color: #888; margin-bottom: 5px;"><i>{source} - {date}</i></p>
                            <p>{description}</p>
                            <a href="{url}" target="_blank" style="color: #3498db;">Read more</a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No news articles found for this stock.")
        elif tabs == "Technical Indicators":
            st.subheader('Technical Indicators')
            import numpy as np

            # Calculate existing indicators
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['SMA_200'] = data['Close'].rolling(window=200).mean()
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            data['RSI'] = rsi

            data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

            # Helper functions to fetch scores (replace with your actual logic)
            def calculate_technical_indicators_score(data):
                score = 0
                
                # SMA cross
                if data['SMA_50'].iloc[-1] > data['SMA_200'].iloc[-1]:
                    score += 40
                else:
                    score += 10
                
                # RSI
                if data['RSI'].iloc[-1] < 30:
                    score += 30
                elif data['RSI'].iloc[-1] > 70:
                    score -= 30
                else:
                    score += 10
                
                # MACD
                if data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1]:
                    score += 30
                else:
                    score -= 10

                return min(max(score, 0), 100)

            def fetch_news_sentiment_score(ticker):
                # Placeholder: Implement actual news sentiment analysis
                np.random.seed(0)
                return np.random.randint(40, 60)  # Placeholder value

            def fetch_financial_ratios_score(ticker):
                # Placeholder: Implement actual financial ratios analysis
                np.random.seed(1)
                return np.random.randint(40, 60)  # Placeholder value

            def fetch_forecasted_data_score(ticker):
                # Placeholder: Implement actual forecasted data analysis
                np.random.seed(2)
                return np.random.randint(40, 60)  # Placeholder value

            def calculate_buy_sell_score(technical_indicators, news_sentiment, financial_ratios, forecasted_data):
                weights = {
                    'technical_indicators': 0.4,
                    'news_sentiment': 0.3,
                    'financial_ratios': 0.2,
                    'forecasted_data': 0.1
                }
                
                total_score = (technical_indicators * weights['technical_indicators'] +
                            news_sentiment * weights['news_sentiment'] +
                            financial_ratios * weights['financial_ratios'] +
                            forecasted_data * weights['forecasted_data'])
                
                total_score = min(max(total_score, 0), 100)
                return total_score

            # Calculate scores
            technical_indicators_score = calculate_technical_indicators_score(data)
            news_sentiment_score = fetch_news_sentiment_score(selected_stocks)
            financial_ratios_score = fetch_financial_ratios_score(selected_stocks)
            forecasted_data_score = fetch_forecasted_data_score(selected_stocks)

            buy_sell_score = calculate_buy_sell_score(technical_indicators_score, news_sentiment_score, financial_ratios_score, forecasted_data_score)

            # Display Buy/Sell Scale
            st.write(f"## Buy/Sell Recommendation Score")
            st.write(f"**Buy/Sell Score:** {buy_sell_score:.2f}/100")
            st.write(f"**Recommendation:** {'Buy' if buy_sell_score > 50 else 'Sell'}")

            # Color-coded scale
            color = 'green' if buy_sell_score > 50 else 'red'
            st.markdown(f"<div style='background-color: {color}; color: white; padding: 10px;'>Buy/Sell Score: {buy_sell_score:.2f}/100</div>", unsafe_allow_html=True)

            # Disclaimer
            st.write("""
            **Disclaimer:**
            The Buy/Sell recommendation score is determined by machine learning models and AI based on various data sources. This should not be considered as financial advice. Always perform your own research or consult with a financial advisor before making investment decisions.
            """)

            # Calculate additional indicators
            # MACD
            data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
            data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20).mean()
            data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
            data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
            
            # Average True Range (ATR)
            high_low = data['High'] - data['Low']
            high_close = abs(data['High'] - data['Close'].shift())
            low_close = abs(data['Low'] - data['Close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['ATR'] = tr.rolling(window=14).mean()
            
            # Stochastic Oscillator
            data['Lowest_Low'] = data['Low'].rolling(window=14).min()
            data['Highest_High'] = data['High'].rolling(window=14).max()
            data['Stochastic_Oscillator'] = 100 * (data['Close'] - data['Lowest_Low']) / (data['Highest_High'] - data['Lowest_Low'])
            
            # Fibonacci Retracement Levels
            max_price = data['Close'].max()
            min_price = data['Close'].min()
            diff = max_price - min_price
            data['Fib_23.6'] = max_price - 0.236 * diff
            data['Fib_38.2'] = max_price - 0.382 * diff
            data['Fib_61.8'] = max_price - 0.618 * diff
            
            # Plot Moving Averages
            st.write("## Moving Averages")
            fig4 = go.Figure()
            fig4.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', mode='lines'))
            fig4.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='50-Day SMA', mode='lines'))
            fig4.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], name='200-Day SMA', mode='lines'))
            fig4.update_layout(
                title_text="Stock Price and Moving Averages",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig4)
            
            st.write("""
            **Moving Averages (SMA):**
            - **50-Day SMA**: Short-term trend.
            - **200-Day SMA**: Long-term trend.
            - **Trading Decision:** When the short-term SMA crosses above the long-term SMA, it may indicate a buying opportunity. Conversely, when the short-term SMA crosses below the long-term SMA, it may indicate a selling opportunity.
            """)

            # Plot RSI
            st.write("## Relative Strength Index (RSI)")
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(x=data['Date'], y=data['RSI'], name='RSI', mode='lines'))
            fig5.add_hline(y=70, line=dict(color='red', dash='dash'), name='Overbought Threshold')
            fig5.add_hline(y=30, line=dict(color='green', dash='dash'), name='Oversold Threshold')
            fig5.update_layout(
                title_text="Relative Strength Index (RSI)",
                xaxis_title="Date",
                yaxis_title="RSI",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig5)
            
            st.write("""
            **Relative Strength Index (RSI):**
            - Measures the speed and change of price movements.
            - **Overbought (>70):** May indicate that the stock is overvalued.
            - **Oversold (<30):** May indicate that the stock is undervalued.
            - **Trading Decision:** Traders might buy when RSI is below 30 and sell when RSI is above 70.
            """)

            # Plot MACD
            st.write("## Moving Average Convergence Divergence (MACD)")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=data['Date'], y=data['MACD'], name='MACD Line', mode='lines'))
            fig6.add_trace(go.Scatter(x=data['Date'], y=data['MACD_Signal'], name='MACD Signal Line', mode='lines'))
            fig6.update_layout(
                title_text="MACD Indicator",
                xaxis_title="Date",
                yaxis_title="MACD",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig6)
            
            st.write("""
            **MACD (Moving Average Convergence Divergence):**
            - Measures the difference between the 12-day and 26-day EMA.
            - **Signal Line:** 9-day EMA of the MACD line.
            - **Trading Decision:** Buy when the MACD crosses above the Signal Line and sell when it crosses below.
            """)

            # Plot Bollinger Bands
            st.write("## Bollinger Bands")
            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', mode='lines'))
            fig7.add_trace(go.Scatter(x=data['Date'], y=data['BB_Upper'], name='Upper Band', mode='lines', line=dict(color='red', dash='dash')))
            fig7.add_trace(go.Scatter(x=data['Date'], y=data['BB_Middle'], name='Middle Band (SMA)', mode='lines', line=dict(color='blue')))
            fig7.add_trace(go.Scatter(x=data['Date'], y=data['BB_Lower'], name='Lower Band', mode='lines', line=dict(color='red', dash='dash')))
            fig7.update_layout(
                title_text="Bollinger Bands",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig7)
            
            st.write("""
            **Bollinger Bands:**
            - Consists of a middle band (SMA) and two outer bands (standard deviation lines).
            - **Upper Band:** Price is high.
            - **Lower Band:** Price is low.
            - **Trading Decision:** A price touching the upper band may indicate overbought conditions, while a price touching the lower band may indicate oversold conditions.
            """)

            # Plot ATR
            st.write("## Average True Range (ATR)")
            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(x=data['Date'], y=data['ATR'], name='ATR', mode='lines'))
            fig8.update_layout(
                title_text="Average True Range (ATR)",
                xaxis_title="Date",
                yaxis_title="ATR",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig8)
            
            st.write("""
            **Average True Range (ATR):**
            - Measures market volatility.
            - **Trading Decision:** High ATR values indicate higher volatility, and low ATR values indicate lower volatility. Traders might use ATR to set stop-loss levels or adjust position sizes.
            """)

            # Plot Stochastic Oscillator
            st.write("## Stochastic Oscillator")
            fig9 = go.Figure()
            fig9.add_trace(go.Scatter(x=data['Date'], y=data['Stochastic_Oscillator'], name='Stochastic Oscillator', mode='lines'))
            fig9.add_hline(y=80, line=dict(color='red', dash='dash'), name='Overbought Threshold')
            fig9.add_hline(y=20, line=dict(color='green', dash='dash'), name='Oversold Threshold')
            fig9.update_layout(
                title_text="Stochastic Oscillator",
                xaxis_title="Date",
                yaxis_title="Stochastic Oscillator",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig9)
            
            st.write("""
            **Stochastic Oscillator:**
            - Measures the level of the close relative to the high-low range over a period.
            - **Overbought (>80):** May indicate a selling opportunity.
            - **Oversold (<20):** May indicate a buying opportunity.
            - **Trading Decision:** Buy when the oscillator is below 20 and sell when it is above 80.
            """)

            # Calculate Fibonacci Retracement Levels
            max_price = data['Close'].max()
            min_price = data['Close'].min()
            diff = max_price - min_price
            data['Fib_23.6'] = max_price - 0.236 * diff
            data['Fib_38.2'] = max_price - 0.382 * diff
            data['Fib_61.8'] = max_price - 0.618 * diff

            # Plot Fibonacci Retracement Levels
            st.write("## Fibonacci Retracement Levels")
            fig10 = go.Figure()
            fig10.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close Price', mode='lines'))
            fig10.add_trace(go.Scatter(x=[data['Date'].iloc[0], data['Date'].iloc[-1]], y=[data['Fib_23.6'].iloc[0], data['Fib_23.6'].iloc[-1]], name='23.6% Retracement', mode='lines', line=dict(color='orange', dash='dash')))
            fig10.add_trace(go.Scatter(x=[data['Date'].iloc[0], data['Date'].iloc[-1]], y=[data['Fib_38.2'].iloc[0], data['Fib_38.2'].iloc[-1]], name='38.2% Retracement', mode='lines', line=dict(color='purple', dash='dash')))
            fig10.add_trace(go.Scatter(x=[data['Date'].iloc[0], data['Date'].iloc[-1]], y=[data['Fib_61.8'].iloc[0], data['Fib_61.8'].iloc[-1]], name='61.8% Retracement', mode='lines', line=dict(color='green', dash='dash')))
            fig10.update_layout(
                title_text="Fibonacci Retracement Levels",
                xaxis_title="Date",
                yaxis_title="Price",
                xaxis_rangeslider_visible=True,
                hovermode='x unified'
            )
            st.plotly_chart(fig10)

            st.write("""
            **Fibonacci Retracement Levels:**
            - Horizontal lines that indicate possible support and resistance levels based on Fibonacci ratios.
            
            **Trading Significance:**
            - **Above 23.6% Level:** If the closing price is above this level, it suggests that the stock is maintaining its bullish trend.
            - **Between 23.6% and 38.2% Levels:** This range often acts as a strong support zone. A price drop below the 23.6% level but holding above the 38.2% level indicates the stock is in a correction phase within an overall uptrend.
            - **Between 38.2% and 61.8% Levels:** This range indicates a deeper correction within the trend. Holding above the 38.2% level but staying below the 61.8% level might still be seen as a bullish continuation.
            - **Under 61.8% Level:** A drop below this level might indicate a reversal of the bullish trend and could be a signal for bearishness.
            """)

        elif tabs == "Suggested Stocks":

            st.title("Suggested Stocks")

            tab1, tab2 = st.tabs(["Short Term Recommendations", "Long Term Recommendations"])

            def format_recommendation(ticker, name, price, explanation):
                formatted_explanation = explanation
                bullet_points = formatted_explanation.split('. ')
                return f"""
                <div style='color: lightgray;'>
                    <h3>{name} ({ticker})</h3>
                    <ul>
                        <li><b>Current Price</b>: ${price:.2f}</li>
                        {"".join([f"<li>{point.strip()}</li>" for point in bullet_points])}
                    </ul>
                </div>
                """

            with tab1:
                st.header("Short Term Recommendations")
                stock_data = get_stock_data()
                short_term_recommendations = recommend_stocks(stock_data, "short")
                if short_term_recommendations:
                    for ticker, name, price, explanation in short_term_recommendations:
                        st.markdown(format_recommendation(ticker, name, price, explanation), unsafe_allow_html=True)
                else:
                    st.write("No data available for short-term trending stocks. Please try again later.")
                st.write("\n\n**Disclaimer:** This is not financial advice.")

            with tab2:
                st.header("Long Term Recommendations")
                stock_data = get_stock_data()
                long_term_recommendations = recommend_stocks(stock_data, "long")
                if long_term_recommendations:
                    for ticker, name, price, explanation in long_term_recommendations:
                        st.markdown(format_recommendation(ticker, name, price, explanation), unsafe_allow_html=True)
                else:
                    st.write("No data available for long-term trending stocks. Please try again later.")
                st.write("\n\n**Disclaimer:** This is not financial advice.")

        elif tabs == "About":
            st.title("About This App")

            st.write("""
                Welcome to InvestiGenie! This application leverages machine learning and artificial intelligence to provide insights into stock market trends and predictions.

                **Features:**
                - **Real-Time Stock Price:** Get the current price of any stock.
                - **Stock Price Comparison:** Compare the prices of multiple stocks over a specified time period.
                - **Forecasting:** Predict future stock prices using advanced machine learning models like Prophet.
                - **Trending Stocks:** Analyze news and market data to suggest the best stocks to buy, with detailed recommendations for long-term and short-term investments.

                **Technologies Used:**
                - **Machine Learning & AI:** We use the Prophet library for time series forecasting, which helps in predicting future stock prices based on historical data.
                - **APIs:** The app fetches real-time and historical stock data from Yahoo Finance and Financial Modeling Prep APIs.
                - **Streamlit:** This app is built using Streamlit, a powerful library that allows for the creation of custom web applications with Python.

                **How It Works:**
                - **Forecasting:** The Prophet model is trained on historical stock price data to predict future prices. The model takes into account various factors such as trends and seasonal effects.
                - **Trending Stocks Analysis:** The app analyzes recent news sentiment and market data to suggest the best stocks for investment. It evaluates stocks based on sentiment scores, P/E ratios, market caps, and other financial metrics.

                **Disclaimer:**
                The predictions and suggestions provided by this app are based on historical data and current market trends. Please conduct your own research or consult with a financial advisor before making any investment decisions.

                **Credits:**
                - This application was built by Dhruv Narayanan.
            """)

            st.write("Thank you for using InvestiGenie!")

        elif tabs == "Financial Ratios":
            st.subheader('Financial Ratios for Selected Stock')

            def display_ratio(name, value, explanation):
                st.markdown(f"### {name}")
                if value is not None:
                    st.write(f"**Value:** {value:.2f}")
                else:
                    st.write("Data not available.")
                st.write(explanation)

            # Fetch financial data
            stock = yf.Ticker(selected_stocks)
            pe_ratio = stock.info.get('trailingPE', None)
            dividend_yield = stock.info.get('dividendYield', None)

            # P/E Ratio
            pe_ratio_explanation = """
            The P/E (Price-to-Earnings) Ratio is a measure of the current share price relative to its per-share earnings. 
            A high P/E ratio could mean that a stock's price is high relative to earnings and possibly overvalued. Conversely, 
            a low P/E ratio might indicate that the current stock price is low relative to earnings.
            """
            display_ratio("P/E Ratio", pe_ratio, pe_ratio_explanation)

            # Dividend Yield
            dividend_yield_explanation = """
            The Dividend Yield is a financial ratio that shows how much a company pays out in dividends each year relative 
            to its stock price. It is expressed as a percentage. A high dividend yield can indicate a good income-generating 
            investment, but it could also suggest that the company's stock price is low or its future growth prospects are dim.
            """
            display_ratio("Dividend Yield", dividend_yield * 100 if dividend_yield else None, dividend_yield_explanation)

            # Other financial data
            st.write("### Additional Financial Data")
            if stock.info:
                market_cap = stock.info.get('marketCap', 'N/A')
                trailing_pe = stock.info.get('trailingPE', 'N/A')
                fifty_two_week_high = stock.info.get('fiftyTwoWeekHigh', 'N/A')
                fifty_two_week_low = stock.info.get('fiftyTwoWeekLow', 'N/A')
                average_volume = stock.info.get('averageVolume', 'N/A')
                regular_market_previous_close = stock.info.get('regularMarketPreviousClose', 'N/A')

                st.write(f"**Market Cap:** ${market_cap / 1e9:.2f} Billion")
                st.write(f"**PE Ratio (TTM):** {trailing_pe}")
                st.write(f"**52 Week High:** ${fifty_two_week_high:.2f}")
                st.write(f"**52 Week Low:** ${fifty_two_week_low:.2f}")
                st.write(f"**Average Volume:** {average_volume}")
                st.write(f"**Previous Close:** ${regular_market_previous_close:.2f}")

                additional_data_explanation = """
                - **Market Cap:** The total market value of a company's outstanding shares. It's a measure of a company's size.
                - **PE Ratio (TTM):** Trailing twelve months P/E ratio. Helps assess if a stock is over or undervalued.
                - **52 Week High/Low:** The highest and lowest prices at which a stock has traded over the past year. Useful for identifying trends.
                - **Average Volume:** The average number of shares traded per day. High volume can indicate high investor interest.
                - **Previous Close:** The last closing price of the stock. Useful for comparing the current price.
                """
                st.write(additional_data_explanation)

    
            else:
                st.write("Additional financial data is not available.")

        # Add a new tab for Educational Resources
        elif tabs == "Educational Resources":
            st.subheader('Educational Resources')

            st.markdown("""
            ## Stock Market Basics
            - **Stock**: A type of security that signifies ownership in a corporation and represents a claim on part of the corporation's assets and earnings.
            - **Ticker Symbol**: An abbreviation used to uniquely identify publicly traded shares of a particular stock on a particular stock market.
            - **Stock Price**: The current price at which a particular stock can be bought or sold.
            
            ## Key Financial Metrics
            - **P/E Ratio**: The Price-to-Earnings ratio is a valuation metric that compares the current price of a stock to its earnings per share (EPS). 
            - **Formula**: P/E Ratio = Stock Price / Earnings Per Share (EPS)
            - **Usage**: A higher P/E ratio might indicate that a stock is overvalued, or investors are expecting high growth rates in the future.
            - **Dividend Yield**: The dividend yield shows how much a company pays out in dividends each year relative to its stock price.
            - **Formula**: Dividend Yield = Annual Dividends Per Share / Stock Price
            - **Usage**: A higher dividend yield can be attractive for income-seeking investors.

            ## Technical Indicators
            - **Moving Averages**: A calculation to analyze data points by creating a series of averages of different subsets of the full data set.
            - **Simple Moving Average (SMA)**: The unweighted mean of the previous n data points.
            - **Usage**: Moving averages help smooth out price data to create a trend-following indicator.
            - **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements.
            - **Formula**: RSI = 100 - (100 / (1 + RS)), where RS = Average Gain / Average Loss
            - **Usage**: RSI values above 70 indicate overbought conditions, while values below 30 indicate oversold conditions.

            ## Additional Resources
            - [Investopedia](https://www.investopedia.com): A comprehensive resource for financial education.
            - [Yahoo Finance](https://finance.yahoo.com): A platform for stock quotes, market data, and financial news.
            - [MarketWatch](https://www.marketwatch.com): Provides the latest stock market, financial and business news.

            ---
            **Disclaimer**: The educational content provided here is for informational purposes only and should not be considered as financial advice. Always consult with a professional financial advisor before making any investment decisions.
            """)

        elif tabs == "Virtual Trading Platform":
            import streamlit as st
            import pandas as pd
            import yfinance as yf
            import sqlite3
            from fpdf import FPDF
            import tempfile
            import base64
            import uuid

            # Initialize session state and database connection
            conn = sqlite3.connect('portfolio.db')
            c = conn.cursor()

            # Create tables with UserID column
            c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                        (UserID TEXT, ticker TEXT, shares REAL, average_price REAL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS cash (UserID TEXT, amount REAL)''')
            c.execute('''CREATE TABLE IF NOT EXISTS transactions
                        (UserID TEXT, ticker TEXT, shares REAL, price REAL, type TEXT)''')

            # Default values
            DEFAULT_CASH = 1000000
            DEFAULT_PORTFOLIO = {'cash': DEFAULT_CASH, 'stocks': {}}
            DEFAULT_TRANSACTIONS = []

            # Function to fetch or create a user profile based on the username
            def get_user_data(username):
                # Generate UserID based on username
                user_id = str(uuid.uuid3(uuid.NAMESPACE_DNS, username))

                # Check if user data already exists in the database
                c.execute('SELECT amount FROM cash WHERE UserID = ?', (user_id,))
                cash_data = c.fetchone()

                if cash_data:
                    st.session_state['portfolio'] = {'cash': cash_data[0], 'stocks': {}}
                    for row in c.execute('SELECT ticker, shares, average_price FROM portfolio WHERE UserID = ?', (user_id,)):
                        st.session_state['portfolio']['stocks'][row[0]] = {'shares': row[1], 'average_price': row[2]}

                    st.session_state['transactions'] = []
                    for row in c.execute('SELECT ticker, shares, price, type FROM transactions WHERE UserID = ?', (user_id,)):
                        st.session_state['transactions'].append({'ticker': row[0], 'shares': row[1], 'price': row[2], 'type': row[3]})
                else:
                    # Initialize new user profile with default values
                    st.session_state['portfolio'] = DEFAULT_PORTFOLIO.copy()
                    st.session_state['transactions'] = DEFAULT_TRANSACTIONS.copy()
                    c.execute('INSERT INTO cash (UserID, amount) VALUES (?, ?)', (user_id, st.session_state['portfolio']['cash']))
                    conn.commit()

                st.session_state['UserID'] = user_id
                st.session_state['username'] = username

            # Function to fetch the current price of a stock
            def fetch_current_price(ticker):
                ticker_obj = yf.Ticker(ticker)
                data = ticker_obj.history(period='1d', interval='1m')
                if not data.empty:
                    return data['Close'].iloc[-1]
                return None

            # Function to update the portfolio in the database
            def update_portfolio_db():
                user_id = st.session_state['UserID']
                c.execute('DELETE FROM portfolio WHERE UserID = ?', (user_id,))
                c.execute('DELETE FROM cash WHERE UserID = ?', (user_id,))
                c.execute('DELETE FROM transactions WHERE UserID = ?', (user_id,))
                c.execute('INSERT INTO cash (UserID, amount) VALUES (?, ?)', (user_id, st.session_state['portfolio']['cash']))
                for ticker, stock_data in st.session_state['portfolio']['stocks'].items():
                    c.execute('INSERT INTO portfolio (UserID, ticker, shares, average_price) VALUES (?, ?, ?, ?)', 
                            (user_id, ticker, stock_data['shares'], stock_data['average_price']))
                for transaction in st.session_state['transactions']:
                    c.execute('INSERT INTO transactions (UserID, ticker, shares, price, type) VALUES (?, ?, ?, ?, ?)', 
                            (user_id, transaction['ticker'], transaction['shares'], transaction['price'], transaction['type']))
                conn.commit()

            # Function to reset the portfolio
            def reset_portfolio():
                st.session_state['portfolio'] = DEFAULT_PORTFOLIO.copy()
                st.session_state['transactions'] = DEFAULT_TRANSACTIONS.copy()
                update_portfolio_db()

            # Function to calculate the total portfolio value
            def calculate_portfolio_value():
                total_value = st.session_state['portfolio']['cash']
                for ticker, stock_data in st.session_state['portfolio']['stocks'].items():
                    current_price = fetch_current_price(ticker)
                    if current_price:
                        total_value += stock_data['shares'] * current_price
                return total_value

            # Function to calculate percentage change in stock price
            def calculate_percentage_change(current_price, average_price):
                return ((current_price - average_price) / average_price) * 100

            # Function to export the portfolio to a PDF file
            def export_to_pdf(portfolio_value):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Portfolio Report", ln=True, align='C')

                pdf.cell(200, 10, txt=f"Cash: ${st.session_state['portfolio']['cash']:,.2f}", ln=True)
                pdf.cell(200, 10, txt=f"Portfolio Value: ${portfolio_value:,.2f}", ln=True)
                pdf.cell(200, 10, txt="Stocks:", ln=True)

                for stock, stock_data in st.session_state['portfolio']['stocks'].items():
                    pdf.cell(200, 10, txt=f"{stock}: {stock_data['shares']} shares at an average price of ${stock_data['average_price']:,.2f}", ln=True)

                pdf.cell(200, 10, txt="Transaction History:", ln=True)
                for transaction in st.session_state['transactions']:
                    pdf.cell(200, 10, txt=f"{transaction['type'].capitalize()} {transaction['shares']} shares of {transaction['ticker']} at ${transaction['price']:,.2f} per share", ln=True)

                return pdf.output(dest='S').encode('latin1')

            # Main function to run the Streamlit app
            def main():
                st.title("🪙 Virtual Trading Platform")

                # Initialize session state with default values if not already set
                if 'portfolio' not in st.session_state:
                    st.session_state['portfolio'] = DEFAULT_PORTFOLIO.copy()
                    st.session_state['transactions'] = DEFAULT_TRANSACTIONS.copy()

                st.subheader("🔑 Enter Password")
                username = st.text_input("Please enter your password or create one to get started with your Virtual Trading Platform:")

                if username and 'UserID' not in st.session_state:
                    get_user_data(username)
                    st.experimental_rerun()

                # Dashboard with current or default values
                st.subheader("📈 Stock Trading")
                ticker = st.text_input("Enter Stock Ticker:", value="AAPL")
                if ticker:
                    current_price = fetch_current_price(ticker)
                    if current_price:
                        st.markdown(f"**Current price of {ticker} is** 🏷️ ${current_price:,.2f}")
                    else:
                        st.error("Failed to fetch current price for the ticker.")

                st.subheader("💵 Buy/Sell Shares")
                buy_shares = st.number_input("Buy Shares:", min_value=0.0, step=1.0, format="%.2f")
                if st.button("Buy"):
                    if ticker and buy_shares > 0:
                        total_cost = buy_shares * current_price
                        if st.session_state['portfolio']['cash'] >= total_cost:
                            st.session_state['portfolio']['cash'] -= total_cost
                            if ticker in st.session_state['portfolio']['stocks']:
                                stock_data = st.session_state['portfolio']['stocks'][ticker]
                                total_shares = stock_data['shares'] + buy_shares
                                avg_price = (stock_data['shares'] * stock_data['average_price'] + total_cost) / total_shares
                                st.session_state['portfolio']['stocks'][ticker] = {'shares': total_shares, 'average_price': avg_price}
                            else:
                                st.session_state['portfolio']['stocks'][ticker] = {'shares': buy_shares, 'average_price': current_price}
                            st.session_state['transactions'].append({'ticker': ticker, 'shares': buy_shares, 'price': current_price, 'type': 'buy'})
                            update_portfolio_db()
                            st.success(f"Bought {buy_shares} shares of {ticker} at ${current_price:,.2f} per share.")
                        else:
                            st.error("Insufficient cash to complete the transaction.")

                sell_shares = st.number_input("Sell Shares:", min_value=0.0, step=1.0, format="%.2f")
                if st.button("Sell"):
                    if ticker and sell_shares > 0:
                        if ticker in st.session_state['portfolio']['stocks'] and st.session_state['portfolio']['stocks'][ticker]['shares'] >= sell_shares:
                            total_revenue = sell_shares * current_price
                            st.session_state['portfolio']['cash'] += total_revenue
                            stock_data = st.session_state['portfolio']['stocks'][ticker]
                            stock_data['shares'] -= sell_shares
                            if stock_data['shares'] == 0:
                                del st.session_state['portfolio']['stocks'][ticker]
                            st.session_state['transactions'].append({'ticker': ticker, 'shares': sell_shares, 'price': current_price, 'type': 'sell'})
                            update_portfolio_db()
                            st.success(f"Sold {sell_shares} shares of {ticker} at ${current_price:,.2f} per share.")
                        else:
                            st.error(f"Not enough shares of {ticker} to sell.")

                # Portfolio value and summary
                portfolio_value = calculate_portfolio_value()
                initial_investment = DEFAULT_CASH
                portfolio_value_percentage_change = ((portfolio_value - initial_investment) / initial_investment) * 100
                st.markdown(f"**Current Portfolio Value:** 💰 ${portfolio_value:,.2f} **({portfolio_value_percentage_change:.2f}%**)", unsafe_allow_html=True)

                st.markdown(f"**Available Cash:** 💵 ${st.session_state['portfolio']['cash']:,.2f}", unsafe_allow_html=True)

                if st.session_state['portfolio']['stocks']:
                    st.write("**Stocks in Portfolio:**")
                    stocks_data = []
                    for ticker, stock_data in st.session_state['portfolio']['stocks'].items():
                        current_price = fetch_current_price(ticker)
                        if current_price:
                            percentage_change = calculate_percentage_change(current_price, stock_data['average_price'])
                            color = "green" if percentage_change >= 0 else "red"
                            stocks_data.append({
                                "Ticker": ticker,
                                "Shares": stock_data['shares'],
                                "Average Price": f"${stock_data['average_price']:,.2f}",
                                "Current Price": f"${current_price:,.2f}",
                                "Change (%)": f"<span style='color:{color}'>{percentage_change:.2f}%</span>"
                            })
                    st.write(pd.DataFrame(stocks_data).to_html(escape=False, index=False), unsafe_allow_html=True)

                if st.session_state['transactions']:
                    st.write("**Transaction History:**")
                    transaction_data = []
                    for transaction in st.session_state['transactions']:
                        transaction_data.append({
                            "Type": transaction['type'].capitalize(),
                            "Ticker": transaction['ticker'],
                            "Shares": transaction['shares'],
                            "Price": f"${transaction['price']:,.2f}",
                        })
                    st.write(pd.DataFrame(transaction_data).to_html(index=False), unsafe_allow_html=True)

                st.subheader("📄 Export Portfolio Report")
                if st.button("Generate PDF"):
                    pdf_report = export_to_pdf(portfolio_value)
                    pdf_temp_file = tempfile.NamedTemporaryFile(delete=False)
                    pdf_temp_file.write(pdf_report)
                    pdf_temp_file.close()
                    with open(pdf_temp_file.name, 'rb') as f:
                        pdf_data = f.read()
                    b64_pdf = base64.b64encode(pdf_data).decode('latin1')
                    href = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="portfolio_report.pdf">Download Portfolio Report PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)

                st.subheader("🔄 Reset Portfolio")
                if st.button("Reset Portfolio"):
                    reset_portfolio()
                    st.success("Portfolio has been reset.")

            if __name__ == "__main__":
                main()



    else:
        st.error("Failed to load data. Please check the stock ticker and try again.")
else:
    st.info("Please enter a stock ticker to get started.")

# Disclaimer
st.markdown("""
---
**Disclaimer:** This application is for informational and educational purposes only and should not be considered as investment advice. The predictions and trends provided are based on historical data and machine learning models, and they do not guarantee future performance. Please consult with a financial advisor before making any investment decisions.
""")
