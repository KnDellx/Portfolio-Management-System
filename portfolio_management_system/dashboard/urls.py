from django.urls import path
from . import views

urlpatterns = [
    path('dashboard', views.dashboard, name='dashboard'),
    path('company-list', views.send_company_list, name="company-list"),
    path('update-prices', views.update_values, name="update-prices"),
    path('get-financials', views.get_financials, name="update-prices"),
    path('add-holding', views.add_holding, name="add-holding"),
    path('get-portfolio-insights', views.get_portfolio_insights, name="get-portfolio-insights"),
    path('backtesting/', views.backtesting, name="backtesting"),
    path('lstm_stock_prediction/', views.lstm_stock_prediction, name='lstm_stock_prediction'),
    path('recommendation_view/', views.recommendation_view, name="recommendation_view"),
    path('delete-holding/', views.delete_holding, name="delete_holding"),
    path('multistock/', views.multistock, name="multistock"),
]