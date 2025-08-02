# config/__init__.py

"""
config 包的初始化文件。

这个文件使得我们可以直接从 'config' 包导入 Config 类，
而不是使用更长的 'config.settings.Config'。

用法:
from config import Config
"""

from .settings import Config

