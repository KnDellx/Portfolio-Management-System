# 在 Django 应用程序中的 views.py 文件中
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import statsmodels.api as sm
import numpy as np

# 计算Alpha和Beta
def calculate_alpha_beta(stock_returns, market_returns):
    # Calculate mean returns
    mean_stock_return = np.mean(stock_returns)
    mean_market_return = np.mean(market_returns)
    
    # Calculate Beta
    covariance = np.cov(stock_returns, market_returns)[0, 1]
    variance_market = np.var(market_returns)
    beta = covariance / variance_market
    
    # Calculate Alpha
    alpha = mean_stock_return - beta * mean_market_return
    
    return alpha, beta

# 路由获取股票数据
@csrf_exempt
def get_stock_data(request):
    try:
        if request.method == 'POST':
            stock_symbol = request.POST.get('symbol')
            stock = yf.Ticker(stock_symbol)

            # 获取股票价格数据
            stock_prices_pre = yf.download(stock_symbol, start='2020-01-01', end=None)
            stock_prices = stock_prices_pre['Close']
            stock_returns = stock_prices.pct_change().dropna()
            
            # 获取market数据
            market_prices_pre = yf.download('^GSPC', start='2020-01-01', end=None)
            market_prices = market_prices_pre['Close']
            market_returns = market_prices.pct_change().dropna()

            # 计算Alpha和Beta
            alpha, beta = calculate_alpha_beta(stock_returns, market_returns)

            # 获取股票信息
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

            # 获取历史股票数据并生成图像
            hist = stock.history(period="1y")
            if not hist.empty:
                # 生成股票价格曲线图
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(hist.index, hist['Close'], marker='o', linestyle='-', color='b')
                plt.title(f"{stock_symbol} Stock Price")
                plt.xlabel("Date")
                plt.ylabel("Price (USD)")
                plt.grid(True)

                # 生成股票百分比变化曲线图
                plt.subplot(2, 1, 2)
                plt.plot(hist.index, hist['Close'].pct_change() * 100, marker='o', linestyle='-', color='r')
                plt.title(f"{stock_symbol} Price Percent Change")
                plt.xlabel("Date")
                plt.ylabel("Percent Change (%)")
                plt.grid(True)

                plt.tight_layout()  # 调整子图布局，防止重叠

                # 将图像转换为 base64 编码的字符串
                img_stream = BytesIO()
                plt.savefig(img_stream, format='png')
                img_stream.seek(0)
                plot_url = base64.b64encode(img_stream.read()).decode('utf-8')
                plt.close()  # 关闭图形，释放资源

                # 将 base64 编码的图像字符串添加到数据中
                data['plot'] = plot_url

            return JsonResponse(data)

        else:
            return JsonResponse({'error': 'Only POST method is allowed for this endpoint.'}, status=405)

    except Exception as e:
        traceback.print_exc()  # 打印异常堆栈信息
        return JsonResponse({'error': str(e)}, status=500)