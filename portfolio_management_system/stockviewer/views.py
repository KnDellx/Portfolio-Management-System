
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import numpy as np

# 计算Alpha和Beta
def calculate_alpha_beta(stock_returns, market_returns):
    mean_stock_return = np.mean(stock_returns)
    mean_market_return = np.mean(market_returns)

    covariance = np.cov(stock_returns, market_returns)[0, 1]
    variance_market = np.var(market_returns)
    beta = covariance / variance_market

    alpha = mean_stock_return - beta * mean_market_return

    return alpha, beta

# 路由获取股票数据
@csrf_exempt
def get_stock_data(request):
    try:
        if request.method == 'POST':
            stock_symbol = request.POST.get('symbol')
            stock = yf.Ticker(stock_symbol)

            stock_prices_pre = yf.download(stock_symbol, start='2020-01-01', end=None)
            stock_prices = stock_prices_pre['Close']
            stock_returns = stock_prices.pct_change().dropna()

            market_prices_pre = yf.download('^GSPC', start='2020-01-01', end=None)
            market_prices = market_prices_pre['Close']
            market_returns = market_prices.pct_change().dropna()

            alpha, beta = calculate_alpha_beta(stock_returns, market_returns)

            info = stock.info
            current_price = stock.history(period='1d')['Close'].iloc[0]
            data = {
                'symbol': info.get('symbol'),
                'longName': info.get('longName'),
                'currency': info.get('currency'),
                'regularMarketPrice': current_price,
                'regularMarketVolume': info.get('regularMarketVolume'),
                'regularMarketDayHigh': info.get('regularMarketDayHigh'),
                'regularMarketDayLow': info.get('regularMarketDayLow'),
                'regularMarketPreviousClose': info.get('regularMarketPreviousClose'),
                'Alpha': alpha,
                'Beta': beta,
            }

            hist = stock.history(period="1y")
            if not hist.empty:
                # 生成股票价格曲线图
                price_trace = go.Scatter(x=hist.index, y=hist['Close'], mode='lines+markers', name='Stock Price',line=dict(color='red'))
                price_layout = go.Layout(
                title=f'{stock_symbol} Stock Price',  # 设置标题
                xaxis=dict(title='Date'),  # 设置x轴标题
                yaxis=dict(title='Price (USD)'),  # 设置y轴标题
             
                )
                
                price_fig = go.Figure(data=[price_trace], layout=price_layout)
                price_plot = pio.to_json(price_fig)

                # 生成股票百分比变化曲线图
                percent_change = hist['Close'].pct_change().dropna() * 100
                percent_change_trace = go.Scatter(x=percent_change.index, y=percent_change, mode='lines+markers', name='Percent Change')
                percent_change_layout = go.Layout(title=f'{stock_symbol} Price Percent Change', xaxis=dict(title='Date'), yaxis=dict(title='Percent Change (%)'))
                percent_change_fig = go.Figure(data=[percent_change_trace], layout=percent_change_layout)
                percent_change_plot = pio.to_json(percent_change_fig)


                data['price_plot'] = price_plot
                data['percent_change_plot'] = percent_change_plot

            return JsonResponse(data)

        else:
            return JsonResponse({'error': 'Only POST method is allowed for this endpoint.'}, status=405)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
