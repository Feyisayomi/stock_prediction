import pandas as pd
from fbprophet import Prophet
from flask import Flask, render_template, send_file, request
from fbprophet.serialize import model_to_json, model_from_json
from datetime import date, datetime
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import dill as pickle
import matplotlib.pyplot as plt

app = Flask(__name__)

apple_model = pickle.load(open("notebooks/apple.pkl","rb"))
bitcoin_model = pickle.load(open("notebooks/bitcoin.pkl","rb"))
nio_model = pickle.load(open("notebooks/nio.pkl","rb"))
bmw_model = pickle.load(open("notebooks/bmw.pkl","rb"))
siemens_model = pickle.load(open("notebooks/siemens.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

def prediction(model, period):
    future = model.make_future_dataframe(periods =period, freq="M")
    forecast = model.predict(future)
    future_ = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    ax = model.plot(forecast);
    df = model.plot_components(forecast);
    return(future_, ax, df)

@app.route('/predict', methods=['POST'])
def predict():
    stocks = request.form["stock"]
    period = request.form["horizon"]

    for stock in stocks:
        if stock == "APPL":
            model = apple_model

        elif stock == "NIO":
            model = nio_model
            
        elif stock == "BTC":
            model = bitcoin_model

        elif stock == "BMW":
            model = bmw_model

        elif stock == "SIEMENS.DE":
            model = siemens_model
        model

    predictions = prediction(model, period)
    return render_template('predict.html', predictions)

if __name__ == "__main__":
    app.run(debug=True)
