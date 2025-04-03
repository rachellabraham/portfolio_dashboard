import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go
import datetime

# Function to refresh the dashboard
def refresh():
    st.rerun()

# Function to compute CAGR contributions
def cagr_contribution(data):
    name = data.name
    yearly_returns = data.dropna().to_frame()

    if yearly_returns.empty or len(yearly_returns) < 2:
        return None  

    n_T = (yearly_returns.index[-1] - yearly_returns.index[0]).days / 365
    overall_cagr = (yearly_returns[name].iloc[-1] / yearly_returns[name].iloc[0]) ** (1 / n_T) - 1

    yearly_returns['Year'] = yearly_returns.index.year
    annual_ret = (yearly_returns.groupby('Year')[name].last() / yearly_returns.groupby('Year')[name].first() - 1).to_dict()

    geometric_contrib = []
    log_returns = [math.log(1 + r) if (1 + r) > 0 else None for r in annual_ret.values()]
    total_log = sum([lr for lr in log_returns if lr is not None])

    for lr in log_returns:
        if lr is None:
            geom_cont = -overall_cagr
        elif total_log == 0:
            geom_cont = 0.0
        else:
            weight = lr / total_log
            geom_cont = weight * overall_cagr
        geometric_contrib.append(geom_cont)

    contributions = pd.DataFrame({
        "Year": list(annual_ret.keys()),
        "Geometric Contribution": geometric_contrib
    }).set_index("Year").round(4)

    return contributions

# Function to compute CVaR
def compute_cvar(returns, portfolio_value, confidence_level=0.05):
    sorted_returns = returns.sort_values()
    cutoff_index = int(confidence_level * len(sorted_returns))
    
    cvar_loss_pct = sorted_returns[:cutoff_index].mean()
    cvar_gain_pct = sorted_returns[-cutoff_index:].mean()
    
    last_value = portfolio_value.iloc[-1]  
    cvar_loss_amount = cvar_loss_pct * last_value
    cvar_gain_amount = cvar_gain_pct * last_value
    
    return cvar_loss_pct, cvar_loss_amount, cvar_gain_pct, cvar_gain_amount

# Streamlit UI
st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("📊 Portfolio Value, Risk & CAGR Analysis")

# Sidebar Configuration
with st.sidebar:
    st.header("Inputs")

    if st.button("🔄 Refresh Dashboard"):
        refresh()

    tickers = st.text_input("📌 Enter Stock Symbols (comma-separated)", "AAPL, MSFT, TSLA")
    tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

    start_date = st.date_input("📅 Start Date", datetime.date(2015, 4, 1))
    end_date = st.date_input("📅 End Date", datetime.date.today(), min_value=start_date)

    investment_dict = {ticker: st.number_input(f"Investment in {ticker} ($)", min_value=0.0, value=1000.0) for ticker in tickers}

    calculate = st.button("📉 Calculate Portfolio")

# Main Dashboard
if calculate:
    stock_data = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            stock_data[ticker] = data["Close"]
        except Exception as e:
            st.error(f"⚠️ Error fetching data for {ticker}: {e}")

    if not stock_data.empty:
        portfolio_value = pd.DataFrame()
        for ticker in tickers:
            shares_bought = investment_dict[ticker] / stock_data[ticker].iloc[0]
            portfolio_value[ticker] = shares_bought * stock_data[ticker]

        portfolio_value["Total Portfolio Value"] = portfolio_value.sum(axis=1)

        # Portfolio Value Chart
        start_value = round(portfolio_value["Total Portfolio Value"].iloc[0])
        end_value = round(portfolio_value["Total Portfolio Value"].iloc[-1])

        st.markdown(
            f"<h3>💹 Portfolio Value Over Time: <span style='font-weight:bold;'>${start_value:,.0f} → ${end_value:,.0f}</span></h3>", 
            unsafe_allow_html=True
        )
        st.line_chart(portfolio_value["Total Portfolio Value"])

        # Determine return frequency
        if (portfolio_value.index[-1] - portfolio_value.index[0]).days > 252:
            returns = portfolio_value["Total Portfolio Value"].resample('ME').last().pct_change().dropna()
            period = "monthly"
        else:
            returns = portfolio_value["Total Portfolio Value"].pct_change().dropna()
            period = "daily"

        # Compute CVaR
        cvar_loss_pct, cvar_loss_amount, cvar_gain_pct, cvar_gain_amount = compute_cvar(returns, portfolio_value["Total Portfolio Value"])

        # Display CVaR results
        st.subheader("📉 Conditional Value at Risk (CVaR)")
        st.markdown(
            f"Using **{period}** returns, the expected **loss** in the worst 5% of periods is "
            f"**{cvar_loss_pct:.2%}** (**\\${cvar_loss_amount:,.0f}**) and the expected **gain** "
            f"in the best 5% of periods is **{cvar_gain_pct:.2%}** (**\\${cvar_gain_amount:,.0f}**)."
        )

        # Calculate CAGR contribution
        cagr_results = cagr_contribution(portfolio_value["Total Portfolio Value"])
        if cagr_results is not None and not cagr_results.empty:
            st.subheader("📊 CAGR Contribution Breakdown")

            # Waterfall Chart with Plotly
            geo_contrib = cagr_results['Geometric Contribution'].tolist()
            total_contribution = sum(geo_contrib)
            x_labels = list(map(str, cagr_results.index)) + ["Total"]
            y_values = geo_contrib + [total_contribution]
            measure_types = ['relative' for _ in geo_contrib] + ['total']
            text_labels = [f"{i:.2%}" for i in geo_contrib] + [f"{total_contribution:.2%}"]

            fig = go.Figure(go.Waterfall(
                measure=measure_types,
                x=x_labels,
                y=y_values,
                text=text_labels,
                connector={"line": {"color": "black", "width": 1, "dash": "dot"}}
            ))

            fig.update_layout(
                title="📊 Waterfall Chart: CAGR Contribution",
                xaxis_title="Year",
                yaxis_title="Contribution to CAGR",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=40, r=40, t=40, b=40)
            )
            fig.update_xaxes(type='category')
            st.plotly_chart(fig)




