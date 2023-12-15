import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price
    
# Additional function for feature engineering
def feature_engineering(options_data, current_price, expiration_date):
    options_data['Moneyness'] = current_price / options_data['strike']
    options_data['TimeToExpiration'] = (pd.to_datetime(expiration_date) - pd.Timestamp.now()).days
    return options_data

# Function to fetch the current stock price
def fetch_current_stock_price(ticker):
    hist = ticker.history(period="1d")
    return hist['Close'][-1]

# Modified function to fetch both call and put options data
def fetch_options_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    exp_dates = ticker.options
    if not exp_dates:
        return None, None, None
    expiration_date = exp_dates[0]
    options_data = ticker.option_chain(expiration_date)
    return options_data.calls, options_data.puts, expiration_date
# Preparing data for models
def prepare_data(options_data):
    X = options_data[['Moneyness', 'TimeToExpiration', 'impliedVolatility','strike']]
    y = options_data['lastPrice']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def calculate_bs_prices(test_data, current_price, T, r, sigma, option_type='call'):
    bs_prices = []
    for index, row in test_data.iterrows():
        K = row['strike']
        bs_price = black_scholes(current_price, K, T, r, sigma, option_type)
        bs_prices.append(bs_price)
    return bs_prices


# Streamlit app
def main():
    st.title("Stock Options Analysis")

    # User input for stock symbol
    stock_symbol = st.text_input("Enter the stock ticker symbol:")
    
    if stock_symbol:
        # Models
        models = {
            "SVR": SVR(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor()
        }
        ticker = yf.Ticker(stock_symbol)
        calls_data, puts_data, expiration_date = fetch_options_data(stock_symbol)
        current_price = fetch_current_stock_price(ticker)

        # Check if data is available
        if calls_data is not None and puts_data is not None:
            calls_data = feature_engineering(calls_data, current_price, expiration_date)
            puts_data = feature_engineering(puts_data, current_price, expiration_date)

            # Preparing data for models
            X_train_calls, X_test_calls, y_train_calls, y_test_calls = prepare_data(calls_data)
            X_train_puts, X_test_puts, y_train_puts, y_test_puts = prepare_data(puts_data)

            # Get model predictions for calls and puts
            option_price_predictions_calls = train_predict_evaluate_plot(models, X_train_calls, X_test_calls, y_train_calls, y_test_calls, "Calls")
            option_price_predictions_puts = train_predict_evaluate_plot(models, X_train_puts, X_test_puts, y_train_puts, y_test_puts, "Puts")

            # Plotting comparison for Calls and Puts
            plot_comparison(y_test_calls, option_price_predictions_calls, bs_prices_calls, 'Option Price Predictions Comparison - Calls')
            plot_comparison(y_test_puts, option_price_predictions_puts, bs_prices_puts, 'Option Price Predictions Comparison - Puts')

            # Calculate MSE, MAE, and RMSE for Black-Scholes model (Calls and Puts)
            # [Your calculations and st.write to display results]

#bs_prices_calls = calculate_bs_prices(X_test_calls, current_stock_price, T, r, sigma, 'call')
#bs_prices_puts = calculate_bs_prices(X_test_puts, current_stock_price, T, r, sigma, 'put')
def train_predict_evaluate_plot(models, X_train, X_test, y_train, y_test, option_type):
    model_predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        model_predictions[name] = predictions
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        st.write(f"{option_type} - {name} MSE: {mse:.3f}, MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        # Plotting feature importances
        if hasattr(model, 'feature_importances_'):
            fig, ax = plt.subplots(figsize=(10, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)
            ax.barh(range(len(indices)), importances[indices], color='b', align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([X_train.columns[i] for i in indices])
            ax.set_xlabel('Relative Importance')
            st.pyplot(fig)

def plot_comparison(test_y, predictions_dict, bs_prices, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(len(test_y)), test_y, label='Real Option Prices', color='green')

    colors = ['blue', 'orange', 'purple', 'brown']
    for (model_name, predictions), color in zip(predictions_dict.items(), colors):
        ax.plot(range(len(predictions)), predictions, label=f'{model_name} Predictions', color=color)

    ax.plot(range(len(bs_prices)), bs_prices, label='Black-Scholes Prices', color='red')
    ax.set_title(title)
    ax.set_xlabel('Options')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)

# Define function for Streamlit to run
if __name__ == "__main__":
    main()