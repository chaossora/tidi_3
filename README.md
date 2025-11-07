# 稳定币发展分析与预测系统

## 项目简介

本项目是针对 2025 年第六届"大湾区杯"粤港澳金融数学建模竞赛 B 题开发的稳定币分析系统。项目基于计量经济学方法，对全球稳定币市场进行全面分析，并对未来 5 年（2025-2030）的发展趋势进行预测。

**核心目标：**
- 分析美元锚定稳定币（USD）与非美元锚定稳定币（非USD）的发展态势
- 预测不同锚定货币稳定币的数量增长趋势
- 评估监管政策、宏观经济和链上活动对稳定币市场的影响
- 进行多情景分析（基线、友好监管、风险规避）

## 项目特点

### 1. 完整的数据处理流程
- **21 个数据源**：涵盖稳定币存量、DEX 交易、跨链流动、宏观经济、监管政策等多维度数据
- **月度时间序列构建**：统一时间频率到月末，构建完整的分析数据集
- **锚定货币面板数据**：为 EUR、JPY、GBP、SGD、CHF 等非美元锚定货币构建面板数据

### 2. 多层次建模体系

#### A. ARDL/ECM 模型（时间序列）
- 对 `log_S_USD` 和 `log_S_nonUSD` 建立误差修正模型
- 自动选择最优滞后阶数（BIC 准则）
- 考虑长期协整关系和短期动态调整

#### B. 面板计数模型（Negative Binomial）
- 预测各锚定货币稳定币品种数量 `N_anchor`
- 包含锚定货币固定效应和时间固定效应
- 关键变量：政策指数、脱锚波动、汇率变化、市场规模

#### C. 2SLS/IV 模型（份额方程）
- 处理内生性问题的工具变量回归
- 预测美元稳定币市场份额 `share_USD`
- 工具变量：滞后政策指数、汇率周期

### 3. 情景分析框架

#### Base（基线情景）
- 外生变量按历史趋势外推
- 政策保持当前水平

#### ProNonUSD（友好监管情景）
- 非美元监管政策友好度提升（+2.0）
- 脱锚波动下降 25%
- 渐进式政策释放机制

#### RiskOff（风险规避情景）
- VIX、DXY 上升（避险情绪）
- DEX 活跃度下降 20%
- 渐进式负面冲击

### 4. 创新方法

- **渐进式政策释放**：政策效应非瞬时生效，而是在预测期内逐步释放
- **N_anchor 与 S_nonUSD 联动**：币种数增长直接反馈到市值预测
- **真实数据驱动**：使用 `total_stablecoins_mcap_by_peg.csv` 获取真实非 USD 市值
- **增强政策变量**：区分正面政策（policy_stage）和负面禁令（ban_dummy）

## 技术栈

### 核心依赖
```
Python >= 3.8
pandas >= 1.5.0
numpy >= 1.23.0
statsmodels >= 0.14.0  # ARDL, ECM, NegativeBinomial, 2SLS
scipy >= 1.10.0
matplotlib >= 3.6.0
```

### 主要模块
- **statsmodels.tsa.api.ARDL**：自回归分布滞后模型
- **statsmodels.discrete.count_model**：泊松/负二项回归
- **statsmodels.sandbox.regression.gmm.IV2SLS**：两阶段最小二乘法

## 项目结构

```
tidi_3/
├── main.py                          # 主入口脚本
├── requirements.txt                 # 依赖清单
├── README.md                        # 项目文档（本文件）
│
├── src/                             # 源代码目录
│   ├── __init__.py
│   ├── config.py                    # 配置文件（路径、参数）
│   ├── utils.py                     # 工具函数（日期处理、变换、统计）
│   ├── 01_build_monthly.py          # 数据加载与月末主表构建
│   ├── 02_build_anchor_panel.py     # 非美元锚定面板构建
│   ├── 03_estimate_models.py        # 模型估计（ARDL/ECM、计数、2SLS）
│   └── 04_forecast_scenarios.py     # 预测与情景分析
│
├── data/                            # 原始数据（21 个 CSV 文件）
│   ├── chains_timeseries.csv        # 链上稳定币存量时间序列
│   ├── total_stablecoins_mcap.csv   # 全球稳定币总市值
│   ├── total_stablecoins_mcap_by_peg.csv  # 按锚定类型分类的市值
│   ├── nonusd_offpeg_timeseries.csv # 非美元稳定币脱锚数据
│   ├── dex_volume_total.csv         # DEX 总交易量
│   ├── fees_by_chain.csv            # 各链交易费用
│   ├── vix.csv                      # VIX 波动率指数
│   ├── usd_index.csv                # 美元指数（DXY）
│   ├── stablecoin_policy_dummies_enhanced_monthly_2019_2025.csv  # 增强政策数据
│   └── ...                          # 其他数据文件
│
├── build/                           # 中间数据
│   ├── month_master.csv             # 月末主表（核心数据集）
│   └── panel_anchor.csv             # 锚定面板数据
│
├── outputs/                         # 输出结果
│   ├── fcst_base_*.csv              # 基线预测结果
│   ├── fcst_ProNonUSD_*.csv         # 友好监管情景预测
│   ├── fcst_RiskOff_*.csv           # 风险规避情景预测
│   ├── tab_ecm_*.tex                # 回归表（LaTeX 格式）
│   ├── fig_*.png                    # 可视化图表
│   └── ...
│
└── logs/                            # 运行日志
    └── model_run.log
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 运行完整分析

```powershell
python main.py
```

### 3. 单独运行各模块

```powershell
# 步骤1：构建月末主表
python src/01_build_monthly.py

# 步骤2：构建锚定面板
python src/02_build_anchor_panel.py

# 步骤3：估计模型
python src/03_estimate_models.py

# 步骤4：预测与情景分析
python src/04_forecast_scenarios.py
```

## 核心功能模块详解

### 模块 1：数据加载与月末主表构建 (`01_build_monthly.py`)

**主要功能：**
- 加载 21 个原始数据文件
- 统一日期格式并转换为月末
- 计算美元/非美元稳定币规模（`S_USD`, `S_nonUSD`）
- 构造宏观变量（VIX, DXY, DEX）
- 计算链上摩擦指标（加权费用、跨链流动）
- 生成变换变量（log, z-score, diff）

**关键类：**
- `RawDataLoader`：批量加载数据
- `MonthlyMasterBuilder`：构建月末主表

**输出：**
- `build/month_master.csv`（核心数据集）

### 模块 2：非美元锚定面板构建 (`02_build_anchor_panel.py`)

**主要功能：**
- 统计每月各锚定货币（EUR, JPY, GBP 等）的在市品类数量 `N_anchor`
- 计算脱锚波动指标（`sigma_offpeg`, `tail1`, `tail2`）
- 映射增强政策变量到各 anchor（区分正负面政策）
- 计算汇率周期影响 `dlog_usd_per_anchor`

**关键类：**
- `AnchorPanelBuilder`：面板数据构建器

**创新点：**
- 政策变量 `Pol_anchor = policy_stage - ban_dummy`（净政策友好度）
- 区分正面政策推进和负面禁令冲击

**输出：**
- `build/panel_anchor.csv`

### 模块 3：模型估计 (`03_estimate_models.py`)

**包含 3 类模型：**

#### A. ARDL/ECM 模型
```python
class ARDLEstimator:
    - 自动选择滞后阶数（p_max=6, q_max=6, ic='bic'）
    - 估计 ARDL 并转换为 ECM 形式
    - Bounds Test 协整检验
```

**估计方程：**
```
Δlog_S_t = α + β₁*ECT_{t-1} + Σγᵢ*Δlog_S_{t-i} + Σδⱼ*ΔX_{t-j} + εₜ
```

#### B. 面板计数模型
```python
class PanelCountEstimator:
    - Negative Binomial 回归（处理过度离散）
    - 包含 Anchor 固定效应
    - Pseudo R² ≈ 0.21
```

**关键变量：**
- `Pol_anchor_l1`：政策指数
- `sigma_offpeg_l1`：脱锚波动
- `dlog_usd_per_anchor`：汇率变化率

#### C. 2SLS/IV 模型
```python
class IVEstimator:
    - 第一阶段：预测 dlog_S_nonUSD
    - 第二阶段：预测 logit(share_USD)
    - 处理内生性问题
```

**输出：**
- `outputs/tab_ecm_usd.tex`（USD 模型回归表）
- `outputs/tab_ecm_nonusd.tex`（非 USD 模型回归表）
- `outputs/tab_panel_counts.tex`（计数模型回归表）
- `outputs/tab_share_2sls.tex`（份额模型回归表）

### 模块 4：预测与情景分析 (`04_forecast_scenarios.py`)

**预测流程：**

#### 步骤 1：外生变量外推（5 种规则）
```python
class ExogProjector:
    - rw: 随机游走
    - rw-drift: 随机游走+漂移
    - mean-revert: 均值回归（AR(1)）
    - hold_last: 维持最后值
    - ar1: 完整 AR(1) 模型
```

#### 步骤 2：递推 ECM 预测
```python
class ECMSimulator:
    - 使用 ARDL 模型进行动态递推预测
    - 计算 95% 置信区间（改进版：饱和函数限制 SE 增长）
    - 支持 Monte Carlo 模拟
```

#### 步骤 3：份额路径
```python
class ShareForecaster:
    - via_levels: 通过规模计算份额
    - via_IV: 通过 IV 模型预测份额
```

#### 步骤 4：面板计数预测
```python
class PanelCountForecaster:
    - 预测各 anchor 的 N_anchor
    - 聚合为 N_nonUSD
    - 支持渐进式政策释放
```

**情景分析创新：**

| 情景 | 关键调整 | 实现方式 |
|------|---------|---------|
| **Base** | 历史趋势外推 | 标准预测 |
| **ProNonUSD** | 政策友好 +2.0<br>脱锚波动 -25% | 渐进式释放（线性增长）<br>N_anchor↑ → S_nonUSD↑ |
| **RiskOff** | VIX/DXY 上升<br>DEX -20% | 渐进式负面冲击<br>投资意愿↓ → 新币种↓ |

**输出：**
- 18 个预测 CSV 文件（3 情景 × 6 指标）
- 4 个可视化图表（PNG 格式）

## 配置参数说明

### `src/config.py`

```python
# 预测跨度
FORECAST_HORIZON = 60  # 60 个月 ≈ 5 年

# 模型参数
MODEL_PARAMS = {
    'ardl': {
        'p_max': 6,      # 最大滞后阶数（因变量）
        'q_max': 6,      # 最大滞后阶数（自变量）
        'ic': 'bic',     # 信息准则
    },
    'panel': {
        'fe_anchor': True,  # anchor 固定效应
        'fe_time': True,    # 时间固定效应
    }
}

# 外生变量外推规则
EXOG_PROJECTION_RULES = {
    'log_DEX': 'rw-drift',        # DEX：随机游走+漂移
    'z_VIX': 'mean-revert',        # VIX：均值回归
    'z_DXY': 'rw',                 # DXY：随机游走
    'Pol_nonUSD_l1': 'hold_last',  # 政策：维持最后值
}

# 情景参数
SCENARIO_PARAMS = {
    'ProNonUSD': {
        'pol_shift': 2.0,                  # 政策友好度提升
        'sigma_offpeg_reduction': 0.25,    # 脱锚波动降低 25%
    },
    'RiskOff': {
        'vix_shift': -0.5,      # VIX 反向调整
        'dxy_shift': -0.5,      # DXY 反向调整
        'dex_trend': -0.2,      # DEX 趋势弱化
    }
}
```

## 输出文件说明

### 1. 预测结果 CSV

| 文件名 | 描述 |
|--------|------|
| `fcst_base_S_USD.csv` | 基线：USD 稳定币规模预测 |
| `fcst_base_S_nonUSD.csv` | 基线：非 USD 稳定币规模预测 |
| `fcst_base_share_levels.csv` | 基线：USD 市场份额（levels 法）|
| `fcst_base_share_IV.csv` | 基线：USD 市场份额（IV 法）|
| `fcst_base_N_anchor.csv` | 基线：各锚定货币品种数 |
| `fcst_base_N_nonUSD.csv` | 基线：非 USD 品种总数 |
| `fcst_ProNonUSD_*.csv` | 友好监管情景（同上 6 个指标）|
| `fcst_RiskOff_*.csv` | 风险规避情景（同上 6 个指标）|

### 2. 可视化图表

| 文件名 | 描述 |
|--------|------|
| `fig_nonusd_bands.png` | 非 USD 稳定币规模带状图（含 CI）|
| `fig_usd_bands.png` | USD 稳定币规模带状图（含 CI）|
| `fig_share.png` | USD 市场份额预测（双子图）|
| `fig_counts_anchor.png` | 各锚定货币品种数预测（2×2 分面图）|

### 3. 回归表（LaTeX）

- `tab_ecm_usd.tex`：USD 误差修正模型
- `tab_ecm_nonusd.tex`：非 USD 误差修正模型
- `tab_panel_counts.tex`：面板计数模型
- `tab_share_2sls.tex`：份额方程（2SLS）

## 核心发现

### 1. 政策影响显著
- **政策指数系数 = 0.88（p<0.001）**：政策友好度每提升 1 单位，稳定币品种数增加约 0.88 个
- 区分正面政策推进和负面禁令冲击至关重要

### 2. 汇率效应明显
- **汇率变化率系数 = 11.85（p=0.047）**：锚定货币升值 1%，该货币稳定币品种数增加约 11.8 个

### 3. 脱锚波动影响
- **脱锚波动系数 = 0.17（p<0.001）**：波动性增加反而促进品种数增加（可能反映市场活跃度）

### 4. 预测结果（2025-2030）

#### Base 情景
- USD 稳定币：持续增长但增速放缓
- 非 USD 稳定币：稳步增长
- USD 市场份额：从 97% 缓慢下降至约 95%

#### ProNonUSD 情景
- 非 USD 稳定币：加速增长（币种数显著增加）
- USD 市场份额：降至约 92-93%

#### RiskOff 情景
- 非 USD 稳定币：增长受抑制
- USD 市场份额：保持在 96-97%（避险效应）

## 关键技术亮点

### 1. 渐进式政策释放机制
```python
# 政策效应在预测期内线性增长
for i, date in enumerate(future_dates):
    progress = (i + 1) / n_periods  # 0.017 → 1.0
    pol_shift_progressive[date] = params['pol_shift'] * progress
```

**优势：**
- 更符合现实：政策生效需要时间
- 避免预测突变：消除阶跃冲击的不现实性
- 逻辑链完整：政策→币种数→市值→份额，层层传导

### 2. N_anchor 与 S_nonUSD 联动
```python
# ProNonUSD 情景：币种数增加 X 倍 → 市值增加 X 倍
N_ratio = N_anchor_pro / N_anchor_base
S_nonUSD_adjusted['level'] *= N_ratio
```

**逻辑：**
- 币种数是市值的先导指标
- 平均每个币种市值假设不变
- 市场扩张主要靠新币种进入

### 3. 改进的置信区间
```python
# 使用饱和函数限制 SE 增长
se_mult = sqrt(h)  # h <= 12
se_mult = sqrt(12) + 0.5 * log(h/12)  # h > 12
```

**优势：**
- 短期预测：标准误随时间平方根增长
- 长期预测：使用对数增长，避免 CI 过宽
- 转回原始规模后更合理

### 4. 增强政策变量
```python
# 净政策友好度 = 正面政策阶段 - 负面禁令
Pol_anchor = policy_stage - ban_dummy
# 范围：-1（纯禁令）到 3（生效且无禁令）
```

**优势：**
- 区分政策方向：正面推动 vs 负面压制
- 更精细刻画：4 个政策阶段（草案→通过→生效）
- 提升解释力：显著改善模型拟合度

## 依赖关系图

```
main.py
  ├─> 01_build_monthly.py
  │     └─> config.py, utils.py
  │
  ├─> 02_build_anchor_panel.py
  │     └─> config.py, utils.py, 01_build_monthly.py
  │
  ├─> 03_estimate_models.py
  │     └─> config.py, utils.py
  │
  └─> 04_forecast_scenarios.py
        └─> config.py, utils.py
```

## 常见问题

### Q1：为什么 ARDL 预测有时会失败？
**A：** statsmodels 的 ARDL.predict 对外生变量格式要求严格。项目实现了 `_manual_recursive_forecast` 作为 fallback，确保预测稳健性。

### Q2：ProNonUSD 情景为什么没有明显效果？
**A：** 早期版本使用瞬时冲击，导致预测不连续。改进版使用**渐进式政策释放**，效应在 60 个月内线性增长，更符合现实。

### Q3：如何调整预测跨度？
**A：** 修改 `config.py` 中的 `FORECAST_HORIZON` 参数（单位：月）。

### Q4：如何添加新的锚定货币？
**A：** 在 `config.py` 的 `ANCHOR_TO_JURISDICTION` 中添加映射关系，确保原始数据中包含该货币的相关记录。

### Q5：模型结果保存在哪里？
**A：** 
- 中间数据：`build/` 目录
- 最终结果：`outputs/` 目录
- 运行日志：`logs/model_run.log`

## 性能优化建议

### 1. 加速数据加载
```python
# 使用 pyarrow 引擎
df = pd.read_csv(path, engine='pyarrow')
```

### 2. 并行化模型估计
```python
from joblib import Parallel, delayed

# 并行估计多个 anchor 的模型
results = Parallel(n_jobs=-1)(
    delayed(estimate_model)(anchor_data) 
    for anchor_data in anchor_list
)
```

### 3. 缓存中间结果
```python
import pickle

# 保存模型对象
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# 加载模型对象
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```
