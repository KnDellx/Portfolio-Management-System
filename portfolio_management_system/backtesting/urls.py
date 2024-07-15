# backtesting/urls.py
from django.urls import path
from .views import backtesting_engine

urlpatterns = [
    path('backtesting_engine/', backtesting_engine, name='backtesting_engine'),
]
