# backtesting/views.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from itertools import product
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
import json
from datetime import datetime
import traceback

# 获取用户输入的股票数据
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Adj Close']]
    return data

# 计算技术指标
def calculate_technical_indicators(df, sma_short_window, sma_long_window):
    df['SMA_Short'] = ta.trend.sma_indicator(df['Adj Close'], window=sma_short_window)
    df['SMA_Long'] = ta.trend.sma_indicator(df['Adj Close'], window=sma_long_window)
    df['RSI'] = ta.momentum.rsi(df['Adj Close'], window=14)
    macd = ta.trend.MACD(df['Adj Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    return df

# 生成交易信号
def generate_signals(df):
    df['RSI_buy_signal'] = (df['RSI'] < 30).astype(int)
    df['RSI_sell_signal'] = (df['RSI'] > 70).astype(int)
    df['SMA_buy_signal'] = ((df['SMA_Short'] > df['SMA_Long']) & (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1))).astype(int)
    df['SMA_sell_signal'] = ((df['SMA_Short'] < df['SMA_Long']) & (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1))).astype(int)
    df['MACD_buy_signal'] = ((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'])).astype(int)
    df['MACD_sell_signal'] = ((df['MACD'] < df['MACD_signal']) & (df['MACD'].shift(1) >= df['MACD_signal'])).astype(int)
    df['buy_signal'] = df[['RSI_buy_signal', 'SMA_buy_signal', 'MACD_buy_signal']].sum(axis=1)
    df['sell_signal'] = df[['RSI_sell_signal', 'SMA_sell_signal', 'MACD_sell_signal']].sum(axis=1)
    return df

# 回测策略并计算绩效指标
def backtest_strategy(df, initial_balance, market_data, buy_ratio=0.1, sell_ratio=0.1, stop_loss=0.05, take_profit=0.1, rebalance_period='M'):
    balance = initial_balance
    position = 0
    buy_signals = []
    sell_signals = []
    portfolio_values = []
    last_buy_price = None

    for i in range(len(df)):
        # 再平衡检查
        if i > 0 and df.index[i].to_period(rebalance_period) != df.index[i-1].to_period(rebalance_period):
            current_value = balance + position * df['Adj Close'].iloc[i]
            position_value = current_value * buy_ratio
            cash_value = current_value - position_value
            position = position_value / df['Adj Close'].iloc[i]
            balance = cash_value

        # 检查止损和止盈条件
        if last_buy_price is not None:
            if (df['Adj Close'].iloc[i] / last_buy_price - 1) <= -stop_loss:
                balance += position * df['Adj Close'].iloc[i]
                position = 0
                last_buy_price = None
                sell_signals.append((df.index[i], df['Adj Close'].iloc[i]))
            elif (df['Adj Close'].iloc[i] / last_buy_price - 1) >= take_profit:
                balance += position * df['Adj Close'].iloc[i]
                position = 0
                last_buy_price = None
                sell_signals.append((df.index[i], df['Adj Close'].iloc[i]))

        if df['buy_signal'].iloc[i] > 0 and balance > 0:
            buy_amount = balance * buy_ratio
            position += buy_amount / df['Adj Close'].iloc[i]
            balance -= buy_amount
            buy_signals.append((df.index[i], df['Adj Close'].iloc[i]))
            last_buy_price = df['Adj Close'].iloc[i]
        elif df['sell_signal'].iloc[i] > 0 and position > 0:
            sell_amount = position * sell_ratio
            balance += sell_amount * df['Adj Close'].iloc[i]
            position -= sell_amount
            sell_signals.append((df.index[i], df['Adj Close'].iloc[i]))
        
        portfolio_values.append(balance + position * df['Adj Close'].iloc[i])
    
    if position > 0:
        balance += position * df['Adj Close'].iloc[-1]

    final_balance = balance
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    market_returns = market_data.pct_change().dropna()
    covariance_matrix = np.cov(returns, market_returns)
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    risk_free_rate = 0.01
    alpha = np.mean(returns) - (risk_free_rate + beta * (np.mean(market_returns) - risk_free_rate))
    total_return = (final_balance - initial_balance) / initial_balance

    return final_balance, buy_signals, sell_signals, beta, sharpe_ratio, alpha, total_return

# 优化策略参数
def optimize_strategy(stock_data, market_data, initial_balance, stop_loss, take_profit, rebalance_period, sma_short_window_range, sma_long_window_range):
    best_sharpe_ratio = -np.inf
    best_params = None
    best_results = None

    for sma_short_window, sma_long_window in product(sma_short_window_range, sma_long_window_range):
        if sma_short_window >= sma_long_window:
            continue

        data_with_indicators = calculate_technical_indicators(stock_data.copy(), sma_short_window, sma_long_window)
        if 'SMA_Short' not in data_with_indicators.columns or 'SMA_Long' not in data_with_indicators.columns:
            continue
        data_with_signals = generate_signals(data_with_indicators)
        final_balance, buy_signals, sell_signals, beta, sharpe_ratio, alpha, total_return = backtest_strategy(
            data_with_signals, initial_balance, market_data, stop_loss=stop_loss, take_profit=take_profit, rebalance_period=rebalance_period
        )

        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            best_params = (sma_short_window, sma_long_window)
            best_results = (final_balance, buy_signals, sell_signals, beta, sharpe_ratio, alpha, total_return)

    return best_params, best_results

# 绘制信号图
def plot_signals(df, buy_signals, sell_signals, ticker):
    fig = go.Figure()

    # 绘制股票价格及其移动平均线
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], mode='lines', name='SMA_Short'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], mode='lines', name='SMA_Long'))
    
    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))

    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

    fig.update_layout(title=f'{ticker} Trade Signal', xaxis_title='Date', yaxis_title='Price')

    # 绘制RSI
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[30, 30], mode='lines', line=dict(color='red', dash='dash'), name='RSI 30'))
    fig_rsi.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[70, 70], mode='lines', line=dict(color='red', dash='dash'), name='RSI 70'))

    fig_rsi.update_layout(title='RSI', xaxis_title='Date', yaxis_title='RSI')

    # 绘制MACD
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal'))
    fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_diff'], name='MACD_diff'))

    fig_macd.update_layout(title='MACD', xaxis_title='Date', yaxis_title='MACD')

    fig.show()
    fig_rsi.show()
    fig_macd.show()

@csrf_exempt
@require_POST
def backtesting_engine(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            ticker = data.get('ticker')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_balance = float(data.get('initial_balance'))
            stop_loss = float(data.get('stop_loss'))
            take_profit = float(data.get('take_profit'))

            stock_data = get_stock_data(ticker, start_date, end_date)
            stock_data_cleaned = stock_data.dropna()

            market_data = get_stock_data('^GSPC', start_date, end_date)['Adj Close']

            sma_short_window_range = range(10, 50, 5)
            sma_long_window_range = range(50, 200, 10)

            best_params, best_results = optimize_strategy(
                stock_data_cleaned, market_data, initial_balance, stop_loss, take_profit, 'M',
                sma_short_window_range, sma_long_window_range
            )

            if best_params is None:
                return JsonResponse({"message": "A suitable combination of parameters was not found."}, status=400)

            sma_short_window, sma_long_window = best_params
            final_balance, buy_signals, sell_signals, beta, sharpe_ratio, alpha, total_return = best_results

            data_with_indicators = calculate_technical_indicators(stock_data_cleaned.copy(), sma_short_window, sma_long_window)
            data_with_signals = generate_signals(data_with_indicators)
            plot_signals(data_with_signals, buy_signals, sell_signals, ticker)
            
            plot_data = {
                'dates': [date.strftime('%Y-%m-%d') for date in data_with_signals.index],
                'adj_close': data_with_signals['Adj Close'].tolist(),
                'sma_short': data_with_signals['SMA_Short'].tolist(),
                'sma_long': data_with_signals['SMA_Long'].tolist(),
                'rsi': data_with_signals['RSI'].tolist(),
                'macd': data_with_signals['MACD'].tolist(),
                'macd_signal': data_with_signals['MACD_signal'].tolist(),
                'macd_diff': data_with_signals['MACD_diff'].tolist(),
                'buy_signals': [(date.strftime('%Y-%m-%d'), price) for date, price in buy_signals],
                'sell_signals': [(date.strftime('%Y-%m-%d'), price) for date, price in sell_signals]
            }
            response_data = {
                'final_balance': final_balance,
                'buy_signals': plot_data['buy_signals'],
                'sell_signals': plot_data['sell_signals'],
                'beta': beta,
                'sharpe_ratio': sharpe_ratio,
                'alpha': alpha,
                'total_return': total_return,
                'sma_short_window': sma_short_window,
                'sma_long_window': sma_long_window,
                'plot_data': plot_data
            }
            print(response_data)
            return JsonResponse(response_data)
            

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({"message": f"An error occurred: {str(e)}"}, status=500)

    return JsonResponse({"message": "Invalid request method."}, status=405)