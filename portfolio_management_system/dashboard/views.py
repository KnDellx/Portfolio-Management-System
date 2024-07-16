
# -- coding: utf-8 --
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
from django.views.decorators.http import require_POST
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import datetime,timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd




# AlphaVantage API
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import subprocess as sp

# 平衡多个API密钥的使用
def get_alphavantage_key():
  """
  alphavantage_keys = [
    settings.ALPHAVANTAGE_KEY1,
    settings.ALPHAVANTAGE_KEY2,
    settings.ALPHAVANTAGE_KEY3,
    settings.ALPHAVANTAGE_KEY4,
    settings.ALPHAVANTAGE_KEY5,
    settings.ALPHAVANTAGE_KEY6,
    settings.ALPHAVANTAGE_KEY7,
  ]
  """
  alphavantage_key = "767ECK0JKR7YDLBP"
  # 随机选择
  return alphavantage_key

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
      # 返回该股票的完整每日时间序列数据
      data, meta_data = ts.get_daily(symbol=company_symbol, outputsize='full')
      # 获取当日收盘价
      buy_price = float(data[request.POST['date']]['4. close'])
      # 用于获取不同类型的财务数据，例如公司概览、收入声明、资产负债表和现金流量表等
      fd = FundamentalData(key=get_alphavantage_key(), output_format='json')
      data, meta_data = fd.get_company_overview(symbol=company_symbol)
      sector = data['Sector']

      # 检查是否已经持有该公司的股票
      found = False
      # 如果找到了该公司的股票，则将购买价值添加到购买价值列表中
      # 如果是已经有的公司的股票，则直接添加购买价值
      for c in holding_companies:
        if c.company_symbol == company_symbol:
          c.buying_value.append([buy_price, number_stocks])
          c.save()
          found = True
      # 如果没有找到该公司的股票，则创建一个新的StockHolding对象
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
  return render(request, 'backtesting/backtesting.html')
  """
  try:
    output = sp.check_output("quantdom", shell=True)
  except sp.CalledProcessError:
    output = 'No such command'
  return HttpResponse("Success")
  """




#---------------------------------预测模型---------------------------------
@csrf_exempt
def lstm_stock_prediction(request):
    if request.method == 'POST':
      
        """
        try:
            data = json.loads(request.body)
            ticker = data['stockCode']
            start_date = data['startDate']
            forecast_days = int(data['predictionDays'])
            # 解析日期字符串，假设格式为 YYYY-MM-DD
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        except (json.JSONDecodeError, KeyError) as e:
            return JsonResponse({'success': False, 'message': f'Invalid JSON data: {str(e)}'})
        end_date = datetime.now().date()
        # 下载股票数据
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            return JsonResponse({'success': False, 'message': 'Invalid stock code or insufficient data.'})
        
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
        # adf检验函数
        def adf_test(timeseries):
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(timeseries)
            # 返回p值
            return result[1]
        # 自动确定合适的拆分函数
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
        # 构建梯度提升树模型
        gbr_model = GradientBoostingRegressor(random_state=42)
        # 网格搜索法寻找最优参数
        grid_search = GridSearchCV(estimator=gbr_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train.reshape(-1, n_steps), y_train.ravel())
        gbr_best_model = grid_search.best_estimator_
        gbr_best_model.fit(X_train.reshape(-1, n_steps), y_train.ravel())
        # 在测试集上进行预测
        y_pred_rf = gbr_best_model.predict(X_test.reshape(-1, n_steps))
        y_pred_rf = scaler.inverse_transform(y_pred_rf.reshape(-1, 1))
        y_test_orig_rf = scaler.inverse_transform(y_test)
        mse_rf = mean_squared_error(y_test_orig_rf, y_pred_rf)

        # 模型择优
        min_mse = min(mse_lstm, mse_arima, mse_rf)
        if min_mse == mse_lstm:
          # 人工选择回溯步长
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
        # 生成正确的日期标签
        future_dates = [str((datetime.now().date() + timedelta(days=i)).isoformat()) for i in range(1, forecast_days + 1)]
        predicted_prices = predicted_prices.flatten().tolist()
        """
        # 使用测试数据
        with open("prediction_results.json", "r") as file:
            data = json.load(file)  
        return JsonResponse({
            'success': True,
            'model': data['model'],
            'mse': data['mse'],
            'dates': data['future_dates'],
            'predictions': data['predicted_prices']
        })
    else:
      print("Invalid request method.")
      return JsonResponse({'success': False, 'message': 'Invalid request method.'})
    



# ---------------------user preference---------------------
# Load group_results.csv into a pandas DataFrame
group_results = pd.read_csv('group_results.csv')

@require_POST
def recommendation_view(request):
    data = json.loads(request.body)
    
    # Extract user preferences
    risk_cluster = int(data.get('risk_tolerance', 0))
    return_cluster = int(data.get('return_preference', 0))
    price_quartile = int(data.get('fund_status', 0))
    investment_term = data.get('investment_term', 'long')  # default to 'long' if not provided

    # Filter group_results based on user preferences
    filtered_data = group_results[
        (group_results['Risk_Cluster'] == risk_cluster) &
        (group_results['Return_Cluster'] == return_cluster) &
        (group_results['Price_Quartile'] == price_quartile) &
        (group_results['Investment_Term'] == investment_term)
    ]

    if filtered_data.empty:
        return JsonResponse({'error': 'No recommendations found for the given preferences.'}, status=404)

    # Extract stock recommendations and number of stocks
    stock_count = filtered_data.iloc[0]['Stock_Count']
    stock_codes = filtered_data.iloc[0]['Stock_Codes'].strip('[]').replace("'", "").split(', ') if stock_count > 0 else []

    # Prepare response data
    response_data = {
        'stock_count': stock_count,
        'stock_codes': stock_codes
    }

    return JsonResponse(response_data)
    
