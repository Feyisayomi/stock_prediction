import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, render_template, send_file, request
import json
from fbprophet.serialize import model_to_json, model_from_json
import streamlit as st
import yfinance as yf
from datetime import date, datetime
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go


st.title("Stock Price Prediction App")
st.markdown("This application enables you to predit the future value of any stock in yahoo Finance in any number of days.")

 #Change sidebar color
st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#D6EAF8,#D6EAF8);
    color: black;
}
</style>
""",
    unsafe_allow_html=True,
)

### Set bigger font style
st.markdown(
	"""
<style>
.big-font {
	fontWeight: bold;
    font-size:22px !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:40px !important;
        color: #f9a01b !important;
        padding-top: 30px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="container">
        <img class="logo-img" src="pictur.png">
        <p class="logo-text">Stock Prediction app</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown("<p class='big-font'><font color='black'>Prediction days</font></p>", unsafe_allow_html=True)
no_of_days = int(st.sidebar.number_input('Number of days to predict:', min_value=0, max_value=1000000, value=365, step=1))
stock_name = st.sidebar.text_input('Symbol of stock to predict:')
start  = datetime(2015,1,1)
end = date.today()
data_df = yf.Ticker(stock_name)
data = data_df.history(start = start, end=end, interval="1d")

st.subheader("Data")
st.write(data.tail())
data = data.reset_index()
df = data[["Date", "Close"]].copy()
df= df.rename(columns={"Date": "ds", "Close": "y"})


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="Close"))
	fig.layout.update(title_text='raw data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

if st.button("predict"):
    model = Prophet(
    growth="linear",
    seasonality_mode="additive",
    changepoint_prior_scale =30,
    seasonality_prior_scale=35,
    holidays_prior_scale = 20,
    daily_seasonality = False,
    weekly_seasonality = False,
    yearly_seasonality = False,).add_seasonality(
    name="monthly",
    period =30.5,
    fourier_order = 55).add_seasonality(
    name="daily",
    period =1,
    fourier_order = 15).add_seasonality(
    name="weekly",
    period = 7,
    fourier_order = 20).add_seasonality(
    name="yearly",
    period =365.25,
    fourier_order = 20).add_seasonality(
    name="quarterly",
    period =365.25/4,
    fourier_order = 15,
    prior_scale = 15)
    model.fit(df)

    future = model.make_future_dataframe(periods=no_of_days)
    forecast = model.predict(future)

    st.subheader("Prediction Data")
    st.write(forecast.tail(30))

    st.subheader(f'Forecast plot for {no_of_days} days')
    fig1 = plot_plotly(model, forecast)
    st.write(fig1)

    st.subheader("Forecast components")
    fig2 = model.plot_components(forecast)
    st.write(fig2)