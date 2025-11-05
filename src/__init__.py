"""
稳定币发展分析与预测系统
Stablecoin Development Analysis System

模块:
- config: 配置参数
- utils: 工具函数
- 01_build_monthly: 数据加载与月末主表构建
- 02_build_anchor_panel: 锚定面板构建
- 03_estimate_models: 模型估计
- 04_forecast_scenarios: 预测与情景分析
"""

__version__ = '1.0.0'
__author__ = 'Stablecoin Analysis Team'

# 导入主要类和函数（可选）
try:
    from .config import *
    from .utils import *
except ImportError:
    pass
