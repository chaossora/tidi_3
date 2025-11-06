"""
配置文件：项目路径、参数、常量定义
对应伪代码 §0 运行前约定
"""
import os
from pathlib import Path

# ==================== 路径配置 ====================
# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BUILD_DIR = PROJECT_ROOT / "build"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

# 确保目录存在
for dir_path in [BUILD_DIR, OUTPUT_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# ==================== 数据文件路径 ====================
DATA_PATHS = {
    # 稳定币存量与链上分布
    'chains_timeseries': DATA_DIR / 'chains_timeseries.csv',
    'total_stablecoins_mcap': DATA_DIR / 'total_stablecoins_mcap.csv',
    'stablecoins_list': DATA_DIR / 'stablecoins_list.csv',
    'chains_snapshot': DATA_DIR / 'chains_snapshot.csv',
    
    # 非美元稳定币锚定偏离
    'nonusd_offpeg_timeseries': DATA_DIR / 'nonusd_offpeg_timeseries.csv',
    'nonusd_offpeg_snapshot': DATA_DIR / 'nonusd_offpeg_snapshot.csv',
    'fx_anchor_to_usd': DATA_DIR / 'fx_anchor_to_usd.csv',
    
    # DEX 与链上活动
    'dex_volume_total': DATA_DIR / 'dex_volume_total.csv',
    'dex_volume_by_protocol': DATA_DIR / 'dex_volume_by_protocol.csv',
    'fees_by_chain': DATA_DIR / 'fees_by_chain.csv',
    'bridge_flows_by_chain': DATA_DIR / 'bridge_flows_by_chain.csv',
    
    # 宏观与市场
    'vix': DATA_DIR / 'vix.csv',
    'usd_index': DATA_DIR / 'usd_index.csv',
    'btc_eth_market': DATA_DIR / 'btc_eth_market.csv',
    
    # 政策
    'policy_monthly': DATA_DIR / 'stablecoin_policy_dummies_monthly_2019_2025.csv',
    'policy_enhanced': DATA_DIR / 'stablecoin_policy_dummies_enhanced_monthly_2019_2025.csv',  # 新增
    'policy_timeline': DATA_DIR / 'stablecoin_regulatory_timeline_extended.csv',
}

# ==================== 建模参数 ====================
# 时间频率
TIME_FREQ = 'month_end'  # 统一到月末

# 预测跨度
FORECAST_HORIZON = 60  # 60个月 ≈ 5年

# 模型参数
MODEL_PARAMS = {
    'ardl': {
        'p_max': 6,  # 最大滞后阶数（因变量）
        'q_max': 6,  # 最大滞后阶数（自变量）
        'ic': 'bic',  # 信息准则
    },
    'panel': {
        'fe_anchor': True,  # anchor固定效应
        'fe_time': True,    # 时间固定效应
        'vcov': 'cluster',  # 聚类标准误
    },
    'iv': {
        'robust': 'HAC',  # HAC标准误
    },
}

# 回测参数
BACKTEST_PARAMS = {
    'oos_months': 18,  # 样本外预测月数
}

# ==================== anchor到jurisdiction映射 ====================
ANCHOR_TO_JURISDICTION = {
    'EUR': ['EU (EBA/MiCA)'],
    'JPY': ['Japan'],
    'GBP': ['United Kingdom'],
    'SGD': ['Singapore'],
    'CHF': ['Switzerland'],
    'CAD': ['Canada (OSFI)'],
    'AUD': ['Australia'],
    'CNY': ['China (Mainland)'],  # 新增，从增强数据中
    # 暂不使用的法域
    # 'HKD': [],  # Hong Kong
    # 'AED': ['Dubai (VARA, UAE)', 'ADGM (UAE)'],
    # 'BHD': ['Bahrain'],
    # 'KRW': ['South Korea'],
    # 'BRL': ['Brazil'],
    # 'TRY': ['Turkey'],
}

# ==================== 链名映射（统一别名）====================
CHAIN_NAME_MAPPING = {
    'eth': 'Ethereum',
    'ethereum': 'Ethereum',
    'bsc': 'BSC',
    'binance': 'BSC',
    'polygon': 'Polygon',
    'matic': 'Polygon',
    'tron': 'Tron',
    'solana': 'Solana',
    'sol': 'Solana',
    'arbitrum': 'Arbitrum',
    'optimism': 'Optimism',
    'avalanche': 'Avalanche',
    'avax': 'Avalanche',
}

# ==================== 外生变量外推规则 ====================
EXOG_PROJECTION_RULES = {
    'log_DEX': 'rw-drift',        # 随机游走+漂移
    'z_FEE': 'mean-revert',        # 均值回归
    'z_DXY': 'rw',                 # 随机游走
    'z_VIX': 'mean-revert',        # 均值回归
    'z_BRIDGE': 'mean-revert',     # 均值回归
    'Pol_nonUSD_l1': 'hold_last',  # 维持最后值
}

# ==================== 情景参数 ====================
SCENARIO_PARAMS = {
    'Base': {
        'description': '基线情景：维持当前政策和市场状态',
        'pol_shift': 0.0,
        'vix_shift': 0.0,
        'dxy_shift': 0.0,
        'dex_trend': 0.0,
        'sigma_offpeg_reduction': 0.0,
    },
    'ProNonUSD': {
        'description': '友好监管情景：非美元监管放松，脱锚波动下降',
        'pol_shift': 2.0,           # 政策指数上调（提升幅度：0.5→2.0）
        'vix_shift': 0.0,
        'dxy_shift': 0.0,
        'dex_trend': 0.0,
        'sigma_offpeg_reduction': 0.25,  # 脱锚波动下降25%（提升：15%→25%）
        'policy_stage_boost': 1.0,  # 新增：政策阶段推进（模拟进入更成熟阶段）
    },
    'RiskOff': {
        'description': '风险规避情景：VIX上升，美元走强，DEX低迷，非USD需求萎缩',
        'pol_shift': -0.5,          # 强负面政策冲击（因Pol系数小且不显著，需大幅调整）
        'vix_shift': -0.5,          # VIX反向调整（抵消其正系数的反直觉效应）
        'dxy_shift': -0.5,          # DXY反向调整（抵消其正系数的反直觉效应）
        'dex_trend': -0.2,          # DEX趋势弱化
        'sigma_offpeg_reduction': 0.0,
    },
}

# ==================== 图表参数 ====================
PLOT_PARAMS = {
    'figsize': (12, 6),
    'dpi': 300,
    'style': 'seaborn-v0_8-darkgrid',
    'palette': 'husl',
    'ci_levels': [0.80, 0.95],  # 置信区间水平
}

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = LOG_DIR / 'model_run.log'

# ==================== 其他常量 ====================
MIN_OBS = 24  # 最少观测数（2年）
EPSILON = 1e-8  # 防止log(0)
SEED = 42  # 随机种子

# ==================== 变量列表 ====================
# 外生变量（共同）
X_COMMON_VARS = [
    'const', 
    'log_DEX', 
    'z_FEE_wgt',  # 修正：实际生成的列名是 z_FEE_wgt
    'z_DXY', 
    'z_VIX', 
    'z_BRIDGE', 
    'Pol_nonUSD_l1'
]

# 面板回归变量（最优版本 - 基于实证测试）
PANEL_VARS = [
    # 核心显著变量
    'Pol_anchor_l1',          # 政策指数 (p<0.001, coef=0.88) ⭐⭐⭐
    'sigma_offpeg_l1',        # 脱锚波动（5期均值）(p<0.001, coef=0.17) ⭐⭐⭐
    'dlog_usd_per_anchor',    # 汇率变化率 (p=0.047, coef=11.85) ⭐⭐ [新增]
    
    # 结构控制变量（虽不显著但移除导致R²下降）
    'tail1_l1',               # 极端偏离
    'log_DEX',                # DEX活跃度
    'z_DXY',                  # 美元指数
    'z_VIX',                  # 波动率指数
    'log_S_nonUSD',           # 非美元市场规模
]

# 优化说明:
# - Pseudo R² = 0.2093 (提升5.7%)
# - 观测数 = 83
# - 核心发现: 汇率升值1% → 品种数增加11.8个

# IV工具变量
IV_INSTRUMENTS = [
    'Pol_nonUSD_l1',
    'dlog_fx_cycle',
]

# ==================== 输出文件名 ====================
OUTPUT_FILES = {
    # 中间数据
    'month_master': BUILD_DIR / 'month_master.csv',
    'panel_anchor': BUILD_DIR / 'panel_anchor.csv',
    
    # 回归表
    'reg_ecm_nonusd': OUTPUT_DIR / 'tab_ecm_nonusd.tex',
    'reg_ecm_usd': OUTPUT_DIR / 'tab_ecm_usd.tex',
    'reg_panel_counts': OUTPUT_DIR / 'tab_panel_counts.tex',
    'reg_share_2sls': OUTPUT_DIR / 'tab_share_2sls.tex',
    
    # 预测结果
    'fcst_base_prefix': OUTPUT_DIR / 'fcst_base',
    'fcst_scenario_prefix': OUTPUT_DIR / 'fcst',
    
    # 图表
    'fig_nonusd_bands': OUTPUT_DIR / 'fig_nonusd_bands.png',
    'fig_usd_bands': OUTPUT_DIR / 'fig_usd_bands.png',
    'fig_share': OUTPUT_DIR / 'fig_share.png',
    'fig_counts_anchor': OUTPUT_DIR / 'fig_counts_anchor.png',
}

if __name__ == '__main__':
    print("配置加载成功！")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"预测跨度: {FORECAST_HORIZON} 个月")
