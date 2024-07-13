# 在 Django 的 urls.py 中配置路由
# 在 stockviewer 应用程序中的 urls.py 文件中
from django.urls import path
from . import views

urlpatterns = [
    path('get_stock_data/', views.get_stock_data, name='get_stock_data'),
]