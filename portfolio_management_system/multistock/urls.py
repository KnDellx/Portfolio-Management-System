# multistock/urls.py
from django.urls import path
from . import views
urlpatterns = [
    path('multistock_engine/', views.multistock_engine, name='multistock_engine'),
]
