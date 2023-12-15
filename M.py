from metaflow import FlowSpec, step
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import norm
import matplotlib.pyplot as plt
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model


class OptionsPricingFlow(FlowSpec):

    @step
    def start(self):
        self.experiment = Experiment(
            api_key="RQCAcvGME97oAoYj35Wu3wu0Q",
            project_name="jf4856nyu-comet-Final",
            workspace="jf4856"
        )
        self.stock_symbol = 'AAPL'  # example stock symbol
        self.next(self.fetch_data)

    @step
    def fetch_data(self):
        # Fetch data using yfinance
        ticker = yf.Ticker(self.stock_symbol)
        exp_dates = ticker.options
        if not exp_dates:
            raise ValueError("No options data available.")
        expiration_date = exp_dates[0]
        options_data = ticker.option_chain(expiration_date)
        current_price = ticker.history(period="1d")['Close'][-1]

        self.calls_data = options_data.calls
        self.puts_data = options_data.puts
        self.expiration_date = expiration_date
        self.current_price = current_price
        self.experiment.log_dataset_info(name="options_data", version="v1")
        self.next(self.feature_engineering)

    @step
    def feature_engineering(self):
        # Feature engineering
        self.calls_data['Moneyness'] = self.current_price / self.calls_data['strike']
        self.calls_data['TimeToExpiration'] = (pd.to_datetime(self.expiration_date) - pd.Timestamp.now()).days / 365
        self.puts_data['Moneyness'] = self.current_price / self.puts_data['strike']
        self.puts_data['TimeToExpiration'] = (pd.to_datetime(self.expiration_date) - pd.Timestamp.now()).days / 365
        self.experiment.log_other("feature_engineering_done", True)
        self.next(self.split_data)

    @step
    def split_data(self):
        # Split data into training and testing sets
        X = self.calls_data[['Moneyness', 'TimeToExpiration', 'impliedVolatility', 'strike']]
        y = self.calls_data['lastPrice']
        self.X_train_calls, self.X_test_calls, self.y_train_calls, self.y_test_calls = train_test_split(X, y, test_size=0.2, random_state=42)
        self.experiment.log_table("train_data_head", self.X_train_calls.head())
        self.next(self.train_model)

    @step
    def train_model(self):
        # Train models
        self.models = {
            "SVR": SVR(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor()
        }
        self.model_predictions = {}
        for name, predictions in self.model_predictions.items():
            mse = mean_squared_error(self.y_test_calls, predictions)
            mae = mean_absolute_error(self.y_test_calls, predictions)
            self.experiment.log_metric(f"{name}_mse", mse)
            self.experiment.log_metric(f"{name}_mae", mae)
        self.next(self.calculate_black_scholes_prices)

    @step
    def calculate_black_scholes_prices(self):
        # Calculate Black-Scholes prices
        self.bs_prices = []
        for index, row in self.X_test_calls.iterrows():
            bs_price = self.black_scholes(self.current_price, row['strike'], 30 / 365, 0.01, 0.2)
            self.bs_prices.append(bs_price)
        self.next(self.plot_comparison)

    @step
    def plot_comparison(self):
        # Plot the comparison
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.y_test_calls)), self.y_test_calls, label='Real Option Prices', color='green')
        
        colors = ['blue', 'orange', 'purple', 'brown']
        for (model_name, predictions), color in zip(self.model_predictions.items(), colors):
            plt.plot(range(len(predictions)), predictions, label=f'{model_name} Predictions', color=color)
        
        plt.plot(range(len(self.bs_prices)), self.bs_prices, label='Black-Scholes Prices', color='red')
        plt.title(f'Option Price Predictions Comparison - {self.stock_symbol}')
        plt.xlabel('Options')
        plt.ylabel('Price')
        plt.legend()
        plt.show()
        self.next(self.end)

    @staticmethod
    def black_scholes(S, K, T, r, sigma):
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @step
    def end(self):
        print("Flow is complete!")
        self.experiment.end()

if __name__ == '__main__':
    OptionsPricingFlow()
