from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('stock-prize/', views.stock_prize, name='stock-prize'),
]