#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
# 允许与文件系统交互，常见，删除文件或目录，获取环境变量，以及执行操作系统命令
import os
# 提供了访问与python解释器相关的变量和函数如
# sys.argv:获取命令行参数
# sys.exit():退出当前程序
import sys


def main():
    # os.envirom存储了所有的环境变量（可以被系统上任何程序访问）
    #setdefault()是将后面的键值对添加到环境变量中
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'portfolio_management_system.settings')
    try:
        # 函数用于解析命令行参数并执行相应的命令
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
