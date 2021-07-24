import pandas as pd
from fbprophet import Prophet
from flask import Flask, render_template, send_file, request
import dill as pickle
import mpld3

app = Flask(__name__,
            static_url_path='/static', 
            static_folder='html/static',
            template_folder='templates')

apple_model = pickle.load(open("notebooks/apple.pkl","rb"))
bitcoin_model = pickle.load(open("notebooks/bitcoin.pkl","rb"))
nio_model = pickle.load(open("notebooks/nio.pkl","rb"))
bmw_model = pickle.load(open("notebooks/bmw.pkl","rb"))
siemens_model = pickle.load(open("notebooks/siemens.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    period = int(request.form["horizon"])
    stock = request.form["stock"]
    model = None

    if stock == "apple":
        model = apple_model

    elif stock == "nio":
        model = nio_model
        
    elif stock == "bitcoin":
        model = bitcoin_model

    elif stock == "bmw":
        model = bmw_model

    elif stock == "siemens":
        model = siemens_model

    future = model.make_future_dataframe(periods =period, freq="D")
    forecast = model.predict(future)
    future_ = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    ax = model.plot(forecast);
    ax.savefig('html/static/ax.png')
    df = model.plot_components(forecast);
    df.savefig('html/static/df_.png')

    return render_template("predict.html", Prediction_data =[future_.to_html(classes='data')],titles=future_.columns.values, Forecast = "html/static/ax.png", Components= "html/static/df_.png")

if __name__ == "__main__":
    app.run(debug=True)
