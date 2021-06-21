import pandas as pd
from fbprophet import Prophet
from flask import Flask, render_template, send_file, request
import json
from fbprophet.serialize import model_to_json, model_from_json
import streamlit as st
import yfinance as yf
from datetime import date, datetime
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import dill as pickle

app = Flask(__name__)
apple_model = pickle.load(open("apple.pkl","rb"))
bitcoin_model = pickle.load(open("bitcoin.pkl","rb"))
nio_model = pickle.load(open("nio.pkl","rb"))
bmw_model = pickle.load(open("bmw.pkl","rb"))
siemens_model = pickle.load(open("siemens.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET','POST'])
def predict():
    pass

if __name__ == "main":
    app.run(debug=True)
