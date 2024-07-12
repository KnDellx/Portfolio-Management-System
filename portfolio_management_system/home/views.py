from django.shortcuts import render

# render函数是一个非常有用的快捷函数, 封装一列操作
# 视图函数，接收web请求并返回web响应
def home(request):
  return render(request, 'home/index.html')