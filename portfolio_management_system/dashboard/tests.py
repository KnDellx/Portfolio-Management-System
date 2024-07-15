# dashboard/tests.py

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.test import TestCase
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import yfinance as yf

class LSTMStockPredictionTest(TestCase):

    def test_lstm_stock_prediction(self):
        ticker = "AAPL"
        start_date = "2020-01-01"
        forecast_days = 15

        # 解析日期字符串，假设格式为 YYYY-MM-DD
        start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        end_date = datetime.now().date()

        # 下载股票数据
        df = yf.download(ticker, start=start_date, end=end_date)

        # 数据处理
        data = df['Adj Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(data)

        def prepare_data(data, n_steps):
            X, y = [], []
            for i in range(len(data) - n_steps):
                X.append(data[i:i + n_steps])
                y.append(data[i + n_steps])
            return np.array(X), np.array(y)

        n_steps = 15
        X, y = prepare_data(data_normalized, n_steps)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # LSTM Model
        lstm_model = Sequential()
        lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(n_steps, 1)))
        lstm_model.add(Dropout(0.2))
        lstm_model.add(Dense(units=1))
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                       callbacks=[EarlyStopping(monitor='val_loss', patience=25)], verbose=1)
        y_pred_lstm = lstm_model.predict(X_test)
        y_pred_lstm = scaler.inverse_transform(y_pred_lstm)
        y_test_orig_lstm = scaler.inverse_transform(y_test)
        mse_lstm = mean_squared_error(y_test_orig_lstm, y_pred_lstm)

        # ARIMA Model
        ts_data = df['Adj Close']
        train_size = int(len(ts_data) * 0.8)
        train_data, test_data = ts_data[:train_size], ts_data[train_size:]

        def adf_test(timeseries):
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(timeseries)
            return result[1]

        def find_best_diff_order(data):
            for i in range(4):
                p_value = adf_test(data)
                if p_value < 0.05:
                    return i
                data = data.diff().dropna()
            return i

        I = find_best_diff_order(ts_data)
        mse_arima = float('inf')
        best_p, best_q = 0, 0
        for p in range(5):
            for q in range(5):
                model = ARIMA(ts_data, order=(p, I, q))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test_data))
                actual_values = ts_data[-len(test_data):].values
                mse = mean_squared_error(actual_values, forecast)
                if mse < mse_arima:
                    mse_arima = mse
                    best_p, best_q = p, q

        # Random Forest Model
        param_grid = {
            'n_estimators': np.arange(60, 131, 10),
            'learning_rate': np.arange(0.01, 0.11, 0.01),
            'max_depth': np.arange(3, 7)
        }
        gbr_model = GradientBoostingRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train.reshape(-1, n_steps), y_train.ravel())
        gbr_best_model = grid_search.best_estimator_
        gbr_best_model.fit(X_train.reshape(-1, n_steps), y_train.ravel())
        y_pred_rf = gbr_best_model.predict(X_test.reshape(-1, n_steps))
        y_pred_rf = scaler.inverse_transform(y_pred_rf.reshape(-1, 1))
        y_test_orig_rf = scaler.inverse_transform(y_test)
        mse_rf = mean_squared_error(y_test_orig_rf, y_pred_rf)

        min_mse = min(mse_lstm, mse_arima, mse_rf)
        if min_mse == mse_lstm:
            last_window = data_normalized[-n_steps:].reshape(1, n_steps, 1)
            predicted_prices = []
            for _ in range(forecast_days):
                predicted_price = lstm_model.predict(last_window)
                predicted_prices.append(predicted_price)
                last_window = np.append(last_window[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            model_name = "LSTM"
            mse = mse_lstm
        elif min_mse == mse_arima:
            last_window = ts_data[-n_steps:].values
            predicted_prices = []
            for _ in range(forecast_days):
                model = ARIMA(last_window, order=(best_p, I, best_q))
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=1)
                predicted_prices.append(forecast[0])
                last_window = np.append(last_window[1:], forecast[0])
            model_name = "ARIMA"
            mse = mse_arima
        else:
            last_window = data_normalized[-n_steps:].reshape(1, -1)
            predicted_prices = []
            for _ in range(forecast_days):
                predicted_price = gbr_best_model.predict(last_window)
                predicted_prices.append(predicted_price)
                last_window = np.append(last_window[:, 1:], predicted_price).reshape(1, -1)
            predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
            model_name = "Random Forest"
            mse = mse_rf

        future_dates = [str((datetime.now().date() + timedelta(days=i)).isoformat()) for i in range(1, forecast_days + 1)]
        predicted_prices = predicted_prices.flatten().tolist()

        data = {
            'success': True,
            'model': model_name,
            'mse': mse,
            'future_dates': future_dates,
            'predicted_prices': predicted_prices
        }

        # Write data to a JSON file
        with open('prediction_results.json', 'w') as json_file:
            json.dump(data, json_file, indent=4)

        # Assertions for test case (optional)
        self.assertTrue(data['success'])
        self.assertIn('model', data)
        self.assertIn('mse', data)
        self.assertIn('future_dates', data)
        self.assertIn('predicted_prices', data)
