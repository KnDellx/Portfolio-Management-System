import csv
import json
import random
import requests
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import Portfolio, StockHolding
from riskprofile.models import RiskProfile
from riskprofile.views import risk_profile

# 预测模型所需要的库
from django.views.decorators.csrf import csrf_exempt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime

# AlphaVantage API
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import subprocess as sp

# 平衡多个API密钥的使用
def get_alphavantage_key():
  alphavantage_keys = [
    settings.ALPHAVANTAGE_KEY1,
    settings.ALPHAVANTAGE_KEY2,
    settings.ALPHAVANTAGE_KEY3,
    settings.ALPHAVANTAGE_KEY4,
    settings.ALPHAVANTAGE_KEY5,
    settings.ALPHAVANTAGE_KEY6,
    settings.ALPHAVANTAGE_KEY7,
  ]
  # 随机选择
  return random.choice(alphavantage_keys)

@login_required
# 展示用户的投资组合仪表盘，检查用户石是否有风险档案，然后获取或创建用户的投资组合
def dashboard(request):
  # objects提供了一系列方法来执行数据库查询，filter接受任意数量的位置参数和关键字参数返回满足给定条件的所有记录
  if RiskProfile.objects.filter(user=request.user).exists():
    try:
      # 获取满足给定条件的单个对象
      portfolio = Portfolio.objects.get(user=request.user)
    except:
      portfolio = Portfolio.objects.create(user=request.user)
    portfolio.update_investment()
    holding_companies = StockHolding.objects.filter(portfolio=portfolio)
    holdings = []
    sectors = [[], []]
    sector_wise_investment = {}
    stocks = [[], []]
    for c in holding_companies:
      company_symbol = c.company_symbol
      company_name = c.company_name
      number_shares = c.number_of_shares
      investment_amount = c.investment_amount
      average_cost = investment_amount / number_shares
      holdings.append({
        'CompanySymbol': company_symbol,
        'CompanyName': company_name,
        'NumberShares': number_shares,
        'InvestmentAmount': investment_amount,
        'AverageCost': average_cost,
      })
      stocks[0].append(round((investment_amount / portfolio.total_investment) * 100, 2))
      stocks[1].append(company_symbol)
      if c.sector in sector_wise_investment:
        sector_wise_investment[c.sector] += investment_amount
      else:
        sector_wise_investment[c.sector] = investment_amount
    for sec in sector_wise_investment.keys():
      sectors[0].append(round((sector_wise_investment[sec] / portfolio.total_investment) * 100, 2))
      sectors[1].append(sec)

    # Adding
    news = fetch_news()
    ###

    context = {
      'holdings': holdings,
      'totalInvestment': portfolio.total_investment,
      'stocks': stocks,
      'sectors': sectors,
      'news': news
    }

    return render(request, 'dashboard/dashboard.html', context)
  else:
    return redirect(risk_profile)


def get_portfolio_insights(request):
  try:
    portfolio = Portfolio.objects.get(user=request.user)
    holding_companies = StockHolding.objects.filter(portfolio=portfolio)
    fd = FundamentalData(key=get_alphavantage_key(), output_format='json')
    portfolio_beta = 0
    portfolio_pe = 0
    for c in holding_companies:
      data, meta_data = fd.get_company_overview(symbol=c.company_symbol)
      portfolio_beta += float(data['Beta']) * (c.investment_amount / portfolio.total_investment)
      portfolio_pe += float(data['PERatio']) * (c.investment_amount / portfolio.total_investment)
    return JsonResponse({"PortfolioBeta": portfolio_beta, "PortfolioPE": portfolio_pe})
  except Exception as e:
    return JsonResponse({"Error": str(e)})


def update_values(request):
  try:
    portfolio = Portfolio.objects.get(user=request.user)
    current_value = 0
    unrealized_pnl = 0
    growth = 0
    holding_companies = StockHolding.objects.filter(portfolio=portfolio)
    stockdata = {}
    for c in holding_companies:
      ts = TimeSeries(key=get_alphavantage_key(), output_format='json')
      data, meta_data = ts.get_quote_endpoint(symbol=c.company_symbol)
      last_trading_price = float(data['05. price'])
      pnl = (last_trading_price * c.number_of_shares) - c.investment_amount
      net_change = pnl / c.investment_amount
      stockdata[c.company_symbol] = {
        'LastTradingPrice': last_trading_price,
        'PNL': pnl,
        'NetChange': net_change * 100
      }
      current_value += (last_trading_price * c.number_of_shares)
      unrealized_pnl += pnl
    growth = unrealized_pnl / portfolio.total_investment
    return JsonResponse({
      "StockData": stockdata, 
      "CurrentValue": current_value,
      "UnrealizedPNL": unrealized_pnl,
      "Growth": growth * 100
    })
  except Exception as e:
    return JsonResponse({"Error": str(e)})


def get_financials(request):
  try:
    fd = FundamentalData(key=get_alphavantage_key(), output_format='json')
    data, meta_data = fd.get_company_overview(symbol=request.GET.get('symbol'))
    financials = {
      "52WeekHigh": data['52WeekHigh'],
      "52WeekLow": data['52WeekLow'],
      "Beta": data['Beta'],
      "BookValue": data['BookValue'],
      "EBITDA": data['EBITDA'],
      "EVToEBITDA": data['EVToEBITDA'],
      "OperatingMarginTTM": data['OperatingMarginTTM'],
      "PERatio": data['PERatio'],
      "PriceToBookRatio": data['PriceToBookRatio'],
      "ProfitMargin": data['ProfitMargin'],
      "ReturnOnAssetsTTM": data['ReturnOnAssetsTTM'],
      "ReturnOnEquityTTM": data['ReturnOnEquityTTM'],
      "Sector": data['Sector'],
    }
    return JsonResponse({ "financials": financials })
  except Exception as e:
    return JsonResponse({"Error": str(e)})


def add_holding(request):
  if request.method == "POST":
    try:
      portfolio = Portfolio.objects.get(user=request.user)
      holding_companies = StockHolding.objects.filter(portfolio=portfolio)
      company_symbol = request.POST['company'].split('(')[1].split(')')[0]
      company_name = request.POST['company'].split('(')[0].strip()
      number_stocks = int(request.POST['number-stocks'])
      ts = TimeSeries(key=get_alphavantage_key(), output_format='json')
      data, meta_data = ts.get_daily(symbol=company_symbol, outputsize='full')
      buy_price = float(data[request.POST['date']]['4. close'])
      fd = FundamentalData(key=get_alphavantage_key(), output_format='json')
      data, meta_data = fd.get_company_overview(symbol=company_symbol)
      sector = data['Sector']

      found = False
      for c in holding_companies:
        if c.company_symbol == company_symbol:
          c.buying_value.append([buy_price, number_stocks])
          c.save()
          found = True

      if not found:
        c = StockHolding.objects.create(
          portfolio=portfolio, 
          company_name=company_name, 
          company_symbol=company_symbol,
          number_of_shares=number_stocks,
          sector=sector
        )
        c.buying_value.append([buy_price, number_stocks])
        c.save()

      return HttpResponse("Success")
    except Exception as e:
      print(e)
      return HttpResponse(e)

def send_company_list(request):
  with open('nasdaq-listed.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    rows = []
    for row in csv_reader:
      if line_count == 0:
        line_count += 1
      else:
        rows.append([row[0], row[1]])
        line_count += 1
  return JsonResponse({"data": rows})


def fetch_news():
  query_params = {
    "country": "us",
    "category": "business",
    "sortBy": "top",
    "apiKey": settings.NEWSAPI_KEY
  }
  main_url = "https://newsapi.org/v2/top-headlines?apiKey=191382c6a0c64a17b4f25390354c531b"
  # fetching data in json format
  res = requests.get(main_url, params=query_params)
  open_bbc_page = res.json()
  # getting all articles in a string article
  article = open_bbc_page["articles"]
  results = []
  for ar in article:
    results.append([ar["title"], ar["description"], ar["url"]])
  # Make news as 2 at a time to show on dashboard
  news = zip(results[::2], results[1::2])
  if len(results) % 2:
    news.append((results[-1], None))
  return news


def backtesting(request):
  print('Function Called')
  try:
    output = sp.check_output("quantdom", shell=True)
  except sp.CalledProcessError:
    output = 'No such command'
  return HttpResponse("Success")





#---------------------------------预测模型---------------------------------
# 准备时间序列数据函数
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)
  
  # LSTM 模型训练和预测视图
@csrf_exempt
def lstm_stock_prediction(request):
    if request.method == 'POST':
        try:
            # 获取请求中的股票代码和日期
            data = request.POST
            ticker = data.get('ticker')
            start_date = data.get('start_date')
            prediction_days = int(data.get('prediction_days'))

            # 下载股票数据
            df = yf.download(ticker, start=start_date, end=datetime.now().date())

            # 提取收盘价数据
            data = df['Adj Close'].values.reshape(-1, 1)

            # 数据归一化
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_normalized = scaler.fit_transform(data)

            # 设置时间步长
            n_steps = prediction_days
            X, y = prepare_data(data_normalized, n_steps)

            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # 构建 LSTM 模型
            LSTM_M = Sequential()
            LSTM_M.add(LSTM(units=50, return_sequences=False, input_shape=(n_steps, 1)))
            LSTM_M.add(Dropout(0.2))
            LSTM_M.add(Dense(units=1))
            LSTM_M.compile(optimizer='adam', loss='mean_squared_error')

            # 训练模型
            history = LSTM_M.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
                                callbacks=[EarlyStopping(monitor='val_loss', patience=25)], verbose=1)

            # 在测试集上进行预测
            y_pred = LSTM_M.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred)
            y_test_orig = scaler.inverse_transform(y_test)

            # 评估模型性能
            mse_lstm = mean_squared_error(y_test_orig, y_pred)

            # 返回 JSON 数据
            response_data = {
                'success': True,
                'message': 'LSTM model trained and evaluated successfully.',
                'mse_lstm': mse_lstm
                # 可以添加其他需要返回的数据，比如预测结果等
            }

            return JsonResponse(response_data)

        except Exception as e:
            response_data = {
                'success': False,
                'message': f'Error: {str(e)}'
            }
            return JsonResponse(response_data)

    return render(request, 'dashboard.html')