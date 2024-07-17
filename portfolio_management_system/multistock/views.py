
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
from itertools import product
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from datetime import datetime
import traceback
import plotly
# 获取股票数据
def get_stock_data(tickers, start, end):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start, end=end)
        if data[ticker].empty:
            print(f"Data download failure：{ticker}")
    return data

# 计算技术指标
def calculate_technical_indicators(df, short_window, long_window):
    df['SMA_Short'] = df['Adj Close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_Long'] = df['Adj Close'].rolling(window=long_window, min_periods=1).mean()
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
    df['buy_signal'] = df['buy_signal'].apply(lambda x: 1 if x > 0 else 0)
    df['sell_signal'] = df['sell_signal'].apply(lambda x: 1 if x > 0 else 0)
    return df

# 风险管理
def apply_stop_loss_take_profit(df, stop_loss, take_profit):
    df['stop_loss'] = df['Adj Close'] * (1 - stop_loss)
    df['take_profit'] = df['Adj Close'] * (1 + take_profit)
    return df

# 回测策略并计算绩效指标
def backtest(df, initial_capital, stop_loss, take_profit):
    capital = initial_capital
    positions = 0
    portfolio = []

    buy_fraction = 0.2  
    sell_fraction = 0.2  

    for index, row in df.iterrows():
        if row['buy_signal'] == 1.0:  # Buy signal
            shares_to_buy = (capital * buy_fraction) // row['Adj Close']
            positions += shares_to_buy
            capital -= shares_to_buy * row['Adj Close']
        elif row['sell_signal'] == 1.0:  # Sell signal
            shares_to_sell = positions * sell_fraction
            capital += shares_to_sell * row['Adj Close']
            positions -= shares_to_sell

        # Check stop loss
        if positions > 0 and row['Adj Close'] <= row['stop_loss']:
            capital += positions * row['Adj Close']
            positions = 0

        # Check take profit
        if positions > 0 and row['Adj Close'] >= row['take_profit']:
            capital += positions * row['Adj Close']
            positions = 0

        portfolio.append(capital + positions * row['Adj Close'])

    df['portfolio'] = portfolio
    return df



# 计算投资组合的最终价值和年化收益率
def calculate_performance(df, initial_capital):
    final_value = df['portfolio'].iloc[-1]
    annualized_return = (final_value / initial_capital) ** (1 / ((df.index[-1] - df.index[0]).days / 252)) - 1
    return final_value, annualized_return

# 优化策略参数
def optimize_strategy(stock_data, initial_amount, stop_loss, take_profit, short_window_range, long_window_range):
    best_sharpe_ratio = -np.inf
    best_params = None
    best_results = None

    for short_window, long_window in product(short_window_range, long_window_range):
        if short_window >= long_window:
            continue

        results = {}
        for stock, data in stock_data.items():
            data = calculate_technical_indicators(data.copy(), short_window, long_window)
            data = generate_signals(data)
            data = apply_stop_loss_take_profit(data, stop_loss, take_profit)
            results[stock] = backtest(data, initial_amount / len(stock_data), stop_loss, take_profit)

        final_values = []
        for stock in stock_data.keys():
            final_value, _ = calculate_performance(results[stock], initial_amount / len(stock_data))
            final_values.append(final_value)

        avg_final_value = np.mean(final_values)
        if avg_final_value > best_sharpe_ratio:
            best_sharpe_ratio = avg_final_value
            best_params = (short_window, long_window)
            best_results = results

    return best_params, best_results

# 绘制信号图
def plot_combined_signals(stock_data, results, tickers):
    plots = {}
    for ticker in tickers:
        df = results[ticker]
        
        buy_signals = df[df['buy_signal'] == 1]
        sell_signals = df[df['sell_signal'] == 1]

        fig = go.Figure()

        # 绘制股票价格及其移动平均线
        fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Short'], mode='lines', name='SMA_Short'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_Long'], mode='lines', name='SMA_Long'))
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Adj Close'], mode='markers', marker=dict(symbol='triangle-up', color='green', size=10), name='Buy Signal'))

        if not sell_signals.empty:
            fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Adj Close'], mode='markers', marker=dict(symbol='triangle-down', color='red', size=10), name='Sell Signal'))

        fig.update_layout(title=f'{ticker} Trade Signal', xaxis_title='Date', yaxis_title='Price')

        # 绘制RSI
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[30, 30], mode='lines', line=dict(color='red', dash='dash'), name='RSI 30'))
        fig_rsi.add_trace(go.Scatter(x=[df.index[0], df.index[-1]], y=[70, 70], mode='lines', line=dict(color='red', dash='dash'), name='RSI 70'))

        fig_rsi.update_layout(title=f'{ticker} RSI', xaxis_title='Date', yaxis_title='RSI')

        # 绘制MACD
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='Signal'))
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_diff'], name='MACD_diff'))

        fig_macd.update_layout(title=f'{ticker} MACD', xaxis_title='Date', yaxis_title='MACD')

        plots[ticker] = {
            'price_plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'rsi_plot': json.dumps(fig_rsi, cls=plotly.utils.PlotlyJSONEncoder),
            'macd_plot': json.dumps(fig_macd, cls=plotly.utils.PlotlyJSONEncoder),
        }
    return plots

@csrf_exempt
def multistock_engine(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            tickers = data.get('tickers')
            start_date = data.get('start_date')
            end_date = data.get('end_date')
            initial_amount = float(data.get('initial_amount', 100000))
            stop_loss = float(data.get('stop_loss', 0.05))
            take_profit = float(data.get('take_profit', 0.1))
            short_window_range = range(int(data.get('short_window_min', 10)), int(data.get('short_window_max', 50)), int(data.get('short_window_step', 5)))
            long_window_range = range(int(data.get('long_window_min', 50)), int(data.get('long_window_max', 200)), int(data.get('long_window_step', 10)))
            
            stock_data = get_stock_data(tickers, start_date, end_date)
            best_params, best_results = optimize_strategy(stock_data, initial_amount, stop_loss, take_profit, short_window_range, long_window_range)
            plots = plot_combined_signals(stock_data, best_results, tickers)
            final_values = []

            # 计算每支股票的最终价值并存储股票名称和最终价值的元组
            for stock, data in stock_data.items():
                final_value, _ = calculate_performance(best_results[stock], initial_amount / len(stock_data))
                final_values.append((stock, final_value))
            
            # 计算总资产和每支股票的占比
            total_final_value = sum(final_value for _, final_value in final_values)
            stock_percentages = []
            for stock_name, final_value in final_values:
                if total_final_value != 0:
                    percentage = final_value / total_final_value
                else:
                    percentage = 0.0
                stock_percentages.append((stock_name, percentage))

            return JsonResponse({
                'sma_short_window': best_params[0],
                'sma_long_window': best_params[1],
                'plots': plots,
                'total_final_value': total_final_value,
                'stock_percentages': stock_percentages,
            }, status=200)

        except Exception as e:
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method.'}, status=405)
