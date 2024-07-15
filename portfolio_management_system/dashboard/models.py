from django.db import models
from django.contrib.auth.models import User
from django.db.models import JSONField

class Portfolio(models.Model):
  # cascade说明删除User时，对应Portfolio也会被删除
  user = models.OneToOneField(User, on_delete=models.CASCADE)
  total_investment = models.FloatField(default=0)

  def update_investment(self):
    investment = 0
    holdings = StockHolding.objects.filter(portfolio=self)
    for c in holdings:
      investment += c.investment_amount
    self.total_investment = investment
    self.save()

  def __str__(self):
    return "Portfolio : " + str(self.user)

# models.Models是StockHolding的父类
class StockHolding(models.Model):
  portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE)
  # 存储公司股票代码，公司名称，公司所处领域，持有股票数量，投资金额，购买价格
  company_symbol = models.CharField(default='', max_length=25)
  company_name = models.CharField(max_length=100)
  sector = models.CharField(default='', max_length=50)
  number_of_shares = models.IntegerField(default=0)
  investment_amount = models.FloatField(default=0)
  # 存储关于购买价值的复杂数据
  buying_value = JSONField(default=list)

  def save(self, *args, **kwargs):
    inv_amount = 0.0
    num_shares = 0
    for price, quantity in self.buying_value:
      inv_amount += price * quantity
      num_shares += quantity
    self.investment_amount = inv_amount
    self.number_of_shares = num_shares
    # 调用父类的方法
    super(StockHolding, self).save(*args, **kwargs)

  def __str__(self):
    return str(self.portfolio) + " -> " + self.company_symbol + " " + str(self.number_of_shares)