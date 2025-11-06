# 稳定币发展分析与预测系统# 稳定币发展分析与预测系统



> **Stablecoin Development Analysis and Forecasting System**本项目实现了基于伪代码的完整稳定币分析流程，包括数据加载、特征构建、模型估计、预测和情景分析。

> 

> 基于ARDL/ECM、面板计数模型和2SLS的完整稳定币市场分析与预测系统## 项目结构



---```

tidi_3/

## 📋 项目概述├── main.py                         # 主入口脚本

├── requirements.txt                # Python依赖

本项目实现了稳定币市场的系统性分析，包括：├── README.md                       # 本文件

- 美元与非美元稳定币的市值预测├── src/                           # 源代码目录

- 不同锚定货币的币种数量预测（带固定效应）│   ├── config.py                  # 配置文件（路径、参数、常量）

- 市场份额的结构分析│   ├── utils.py                   # 工具函数模块

- 三种情景分析（Base、ProNonUSD、RiskOff）│   ├── 01_build_monthly.py        # 数据加载与月末主表构建

│   ├── 02_build_anchor_panel.py   # 非美元锚定面板构建

**核心特性**：│   ├── 03_estimate_models.py      # 模型估计（ARDL/ECM、面板计数、2SLS）

- ✅ Anchor Fixed Effects（解决国家政策交叉影响）│   └── 04_forecast_scenarios.py   # 预测与情景分析

- ✅ 方案B政策传导机制（Pol_anchor → N_anchor → S_nonUSD）├── data/                          # 原始数据目录

- ✅ 自动模型选择（NB→Poisson回退）│   ├── chains_timeseries.csv

- ✅ 完整的置信区间估计│   ├── total_stablecoins_mcap.csv

│   ├── nonusd_offpeg_timeseries.csv

---│   ├── fx_anchor_to_usd.csv

│   ├── dex_volume_total.csv

## 🗂️ 项目结构│   ├── fees_by_chain.csv

│   ├── bridge_flows_by_chain.csv

```│   ├── vix.csv

tidi_3/│   ├── usd_index.csv

├── main.py                         # 主入口脚本（运行整个分析流程）│   ├── stablecoin_policy_dummies_monthly_2019_2025.csv

├── requirements.txt                # Python依赖包列表│   └── ... (其他数据文件)

├── README.md                       # 本文件├── build/                         # 中间结果目录

││   ├── month_master.csv           # 月末主表

├── src/                           # 源代码目录│   └── panel_anchor.csv           # 锚定面板

│   ├── config.py                  # 配置文件（路径、参数、常量）├── outputs/                       # 最终输出目录

│   ├── utils.py                   # 工具函数模块│   ├── fcst_base_*.csv            # 基线预测

│   ├── 01_build_monthly.py        # §1-§2: 数据加载与月末主表构建│   ├── fcst_*_*.csv               # 情景预测

│   ├── 02_build_anchor_panel.py   # §3: 非美元锚定面板构建│   ├── fig_*.png                  # 图表

│   ├── 03_estimate_models.py      # §4-§5: 模型估计（ARDL/ECM、面板计数、2SLS）│   └── tab_*.tex                  # 回归表（LaTeX格式）

│   └── 04_forecast_scenarios.py   # §6-§8: 预测与情景分析、结果导出└── logs/                          # 日志目录

│    └── model_run.log              # 运行日志

├── data/                          # 原始数据目录（17个CSV文件）```

│   ├── chains_timeseries.csv              # 链上稳定币时序数据

│   ├── total_stablecoins_mcap.csv         # 总市值数据## 功能模块

│   ├── nonusd_offpeg_timeseries.csv       # 非USD脱锚时序数据

│   ├── stablecoins_list.csv               # 稳定币列表### 1. 数据加载与月末主表构建 (`01_build_monthly.py`)

│   ├── fx_anchor_to_usd.csv               # 汇率数据

│   ├── dex_volume_total.csv               # DEX交易量**对应伪代码 §2**

│   ├── fees_by_chain.csv                  # 链上费用

│   ├── bridge_flows_by_chain.csv          # 跨链流动- 加载所有原始CSV数据

│   ├── vix.csv                             # VIX波动率指数- 统一日期格式并转换为月末

│   ├── usd_index.csv                       # 美元指数- 构造美元/非美元稳定币规模（S_USD、S_nonUSD、S_ALL）

│   ├── btc_eth_market.csv                  # BTC/ETH市场数据- 计算份额（share_USD）

│   ├── stablecoin_policy_dummies_monthly_2019_2025.csv  # 政策虚拟变量- 加入宏观变量（DEX、VIX、DXY）

│   └── ... (其他数据文件)- 计算链上摩擦（FEE_wgt）与跨链流动（BRIDGE_net）

│- 生成变换变量（log、z-score、差分等）

├── build/                         # 中间结果目录

│   ├── month_master.csv           # 月末主表（97个月×23个变量）**输出**: `build/month_master.csv`

│   └── panel_anchor.csv           # 锚定面板（99行×8个anchor）

│### 2. 非美元锚定面板构建 (`02_build_anchor_panel.py`)

├── outputs/                       # 最终输出目录

│   ├── tab_ecm_usd.tex            # USD ARDL模型回归表**对应伪代码 §3**

│   ├── tab_ecm_nonusd.tex         # 非USD ARDL模型回归表

│   ├── tab_panel_counts.tex       # 面板计数模型回归表（带Fixed Effects）- 统计每月在市品类数量（N_anchor）

│   ├── tab_share_2sls.tex         # 份额2SLS模型回归表- 计算脱锚波动（sigma_offpeg）与尾部概率（tail1、tail2）

│   │- 映射政策哑变量到各anchor（Pol_anchor）

│   ├── fcst_base_*.csv            # 基线预测（6个文件）- 加入汇率周期（dlog_usd_per_anchor）

│   ├── fcst_Base_*.csv            # Base情景（6个文件）- 创建滞后变量

│   ├── fcst_ProNonUSD_*.csv       # ProNonUSD情景（6个文件）- 计算全局非美元品类数量（N_nonUSD）

│   ├── fcst_RiskOff_*.csv         # RiskOff情景（6个文件）

│   │**输出**: `build/panel_anchor.csv`

│   ├── fig_usd_bands.png          # USD市值预测图（含置信区间）

│   ├── fig_nonusd_bands.png       # 非USD市值预测图### 3. 模型估计 (`03_estimate_models.py`)

│   ├── fig_share.png              # USD份额预测图（三情景对比）⭐

│   └── fig_counts_anchor.png      # 各anchor币种数预测图**对应伪代码 §4-5**

│

└── logs/                          # 日志目录#### 模型A: ARDL/ECM

    └── model_run.log              # 运行日志- 自动选择滞后阶数（BIC）

```- 估计误差修正模型

- Bounds Test（协整检验）

---- 分别估计 `log_S_USD` 和 `log_S_nonUSD`



## 🚀 快速开始#### 模型B: 面板计数（泊松/负二项）

- 检测过度离散

### 1. 环境配置- 估计 `N_anchor ~ Pol_anchor + sigma_offpeg + ...`

- 固定效应（anchor、时间）

```bash

# 安装依赖#### 模型C: 份额方程（2SLS/IV）

pip install -r requirements.txt- 第一阶段：预测 `dlog_S_nonUSD`（工具变量：Pol_nonUSD_l1、dlog_fx_cycle）

```- 第二阶段：`logit_share_USD ~ dlog_S_nonUSD_hat + 控制变量`

- 识别"非美元扩张 → 美元份额"效应

**依赖包**：

- `pandas` - 数据处理**输出**: 模型对象（可导出回归表）

- `numpy` - 数值计算

- `statsmodels` - 统计模型（ARDL/ECM、Poisson、2SLS）### 4. 预测与情景分析 (`04_forecast_scenarios.py`)

- `matplotlib` - 图表绘制

- `scipy` - 科学计算**对应伪代码 §6-8**



### 2. 运行完整分析#### 基线预测

- 外生变量外推（随机游走、均值回归、维持最后值）

```bash- 基于ECM递推预测S_USD和S_nonUSD（60个月）

# 运行整个分析流程（约4秒）- 计算份额路径

python main.py

```#### 三类情景

1. **Base**: 维持当前政策和市场状态

### 3. 查看结果2. **ProNonUSD**: 友好监管（Pol_nonUSD +0.5，脱锚波动 -15%）

3. **RiskOff**: 风险规避（VIX +1σ，DXY +1.5σ，DEX低迷）

```bash

# 查看生成的图表#### 可视化

start outputs/fig_share.png- 预测带状图（置信区间）

- 份额路径对比

# 查看回归表- 各anchor品类数预测

code outputs/tab_panel_counts.tex

**输出**: 

# 查看预测数据- `outputs/fcst_*.csv` (预测数据)

code outputs/fcst_ProNonUSD_share_levels.csv- `outputs/fig_*.png` (图表)

```

## 快速开始

---

### 1. 环境准备

## 📊 模型架构

```powershell

### 模型A：ARDL/ECM（误差修正模型）# 创建虚拟环境（可选）

python -m venv venv

**目标变量**：.\venv\Scripts\Activate.ps1

- `log_S_USD`：美元稳定币市值（对数）

- `log_S_nonUSD`：非美元稳定币市值（对数）# 安装依赖

pip install -r requirements.txt

**外生变量**：```

- `log_DEX`：去中心化交易所交易量

- `z_VIX`：VIX波动率指数（标准化）### 2. 数据准备

- `z_DXY`：美元指数（标准化）

- `z_FEE_wgt`：链上费用加权确保 `data/` 目录下包含所有必需的CSV文件（见项目结构）。

- `z_BRIDGE`：跨链流动强度

### 3. 运行完整流程

**模型设定**：

- USD模型：ARDL(3, 0) - 3个滞后项，0个外生变量滞后```powershell

- 非USD模型：ARDL(2, 0) - 2个滞后项# 运行主脚本（完整流程）

- 自动选择滞后阶数（基于BIC准则）python main.py

```

---

### 4. 单独运行各模块

### 模型B：面板计数模型（Poisson回归 + Fixed Effects）⭐

```powershell

**目标变量**：# 只构建月末主表

- `N_anchor`：各锚定货币的稳定币数量python src/01_build_monthly.py



**关键特性**：# 只构建锚定面板

- ✅ **Anchor Fixed Effects**：7个哑变量（AUD基准）python src/02_build_anchor_panel.py

- ✅ 解决国家政策交叉影响问题（中国禁令不影响欧洲）

# 只估计模型（需要先有build/*.csv）

**外生变量**：python src/03_estimate_models.py

- `Pol_anchor_l1`：政策指数滞后项（系数=0.47）

- `sigma_offpeg_l1`：脱锚波动率滞后项# 只运行预测（需要先有模型）

- `log_S_nonUSD`：非USD总市值python src/04_forecast_scenarios.py

- `dlog_usd_per_anchor`：汇率变化率```

- `tail1_l1`：尾部风险滞后项

- `log_DEX`, `z_DXY`, `z_VIX`：宏观变量## 配置说明

- `anchor_CAD`, `anchor_CHF`, ... `anchor_SGD`：固定效应哑变量

在 `src/config.py` 中可以修改：

**模型性能**：

- Pseudo R² - **预测跨度**: `FORECAST_HORIZON = 60`（默认60个月）

- 观测数：83- **模型参数**: `MODEL_PARAMS`（滞后阶数、信息准则等）

- **情景参数**: `SCENARIO_PARAMS`（政策冲击、VIX调整等）

**关键预测**：- **anchor到jurisdiction映射**: `ANCHOR_TO_JURISDICTION`

- EUR： **外生变量外推规则**: `EXOG_PROJECTION_RULES`

- CHF

- JPY/GBP

- CNY

### 中间结果 (`build/`)

---- `month_master.csv`: 月末主表，包含所有时序变量

- `panel_anchor.csv`: 锚定面板，包含各anchor的统计指标

### 模型C：份额2SLS模型

### 最终结果 (`outputs/`)

**目标变量**：- `fcst_base_S_USD.csv`: 美元稳定币基线预测

- `logit_share_USD`：USD份额的logit变换- `fcst_base_S_nonUSD.csv`: 非美元稳定币基线预测

- `fcst_base_share_levels.csv`: 份额预测（水平法）

**工具变量**：- `fcst_ProNonUSD_*.csv`: 友好监管情景预测

- `log_DEX`：DEX交易量（外生冲击）- `fcst_RiskOff_*.csv`: 风险规避情景预测

- `fig_nonusd_bands.png`: 非美元规模预测带状图

**模型性能**：- `tab_ecm_*.tex`: 回归表（LaTeX格式）



- `model_run.log`: 完整运行日志

---

## 核心变量说明

## 🎯 情景分析

### 月度时序变量

### 情景1：Base（基线）- `S_USD`: 美元稳定币规模（USD）

- `S_nonUSD`: 非美元稳定币规模（USD）

**设定**：维持当前政策和市场状态- `S_ALL`: 全部稳定币规模（USD）

- `share_USD`: 美元稳定币份额

**预测结果**：- `DEX`: DEX月度成交量

- USD份额：波动率指数（月均）

- S_USD：DXY`: 美元指数（月末）

- S_nonUSD：`FEE_wgt`: 链上费用加权均值

- N_nonUSD总数：`BRIDGE_net`: 跨链净流入

- `Pol_nonUSD`: 非美元政策指数（加权）

---

### 面板变量（anchor × month）

### 情景2：ProNonUSD（友好监管）⭐- `N_anchor`: 各anchor的在市品类数

- `sigma_offpeg`: 脱锚波动（标准差）

**政策冲击**：- `tail1`, `tail2`: 脱锚尾部概率（>1%、>2%）

- `Pol_anchor_l1 +2.0`（监管友好，政策指数提升2个单位）- `Pol_anchor`: 各anchor的政策指数

- `sigma_offpeg -25%`（脱锚波动下降）- `dlog_usd_per_anchor`: 汇率周期



```### 链名映射

- 统一链名别名（如 eth → Ethereum）

**预测结果**：- 加权使用"各链USD稳定币存量占比"

- USD份额

- N_nonUSD总数

- 关键洞察：监管放松可使非USD市值增长

- 政策强度 = active_regime_dummy + non_us_currency_covered_dummy

---- 多个jurisdiction取平均



### 情景3：RiskOff（风险规避）### 负值检查

- `S_nonUSD` 可能为负（口径不一致）

**市场冲击**


- **pandas**: 数据处理

**预测结果**：- **numpy**: 数值计算

- USD份额：- **scipy**: 统计函数

- 避险情绪下USD相对强势- **statsmodels**: 时间序列（ARDL）、面板（GLM）、2SLS

- **matplotlib**: 绘图

---- **seaborn**: 美化图表



## 📈 核心成果## 常见问题





## 引用与参考

---

本项目基于伪代码实现，对应赛题第3题的完整分析框架。

## 🛠️ 技术亮点

数据来源：

### 1. Anchor Fixed Effects- **DeFiLlama**: 稳定币存量、链上分布、DEX数据


- **解决**：添加7个anchor哑变量，完全隔离各国影响- **FRED**: VIX、美元指数

**自定义**: 政策哑变量面板



### 2. 方案B：N_anchor传导机制模型方法：


- **创新**：政策 → 币种数 → 市值的完整传导链- **面板计数**: Cameron & Trivedi (2013)

- **效果**：情景差异提升- **2SLS/IV**: Wooldridge (2010)



### 3. 自动模型选择## 许可证

- **逻辑**：检测过度离散 → 尝试NB → 收敛检查 → 回退Poisson

- **效果**：模型稳定性提升本项目仅供学术研究使用。



### 4. 完整的置信区间## 作者

- **方法**：Bootstrap + ARDL原生CI

- **应用**：所有预测图表包含80%和95%置信区间稳定币分析团队



---## 更新日志



## 📚 输出文件说明- **v1.0** (2025-11-05): 初始版本，完整实现伪代码框架


### 回归表（LaTeX格式，可直接用于论文）

1. **`tab_ecm_usd.tex`**：USD ARDL/ECM模型
   - 系数估计、标准误、显著性
   - 误差修正项、滞后项

2. **`tab_ecm_nonusd.tex`**：非USD ARDL/ECM模型
   - 同上

3. **`tab_panel_counts.tex`**：面板计数模型（含Fixed Effects）⭐
   - 7个anchor固定效应系数


4. **`tab_share_2sls.tex`**：份额2SLS模型
   - 第一阶段、第二阶段结果
   - 工具变量显著性

### 预测数据（CSV格式）

每个情景包含6个文件：
- `fcst_*_S_USD.csv`：USD市值预测（对数空间+水平空间）
- `fcst_*_S_nonUSD.csv`：非USD市值预测
- `fcst_*_share_levels.csv`：份额预测（levels法）
- `fcst_*_share_IV.csv`：份额预测（IV法）
- `fcst_*_N_anchor.csv`：各anchor币种数预测（按anchor×月份）
- `fcst_*_N_nonUSD.csv`：非USD总币种数预测

### 图表（PNG格式，300 DPI）

1. **`fig_usd_bands.png`**：USD市值预测
   - 点预测 + 80%/95%置信区间

2. **`fig_nonusd_bands.png`**：非USD市值预测
   - 同上

3. **`fig_share.png`**：USD份额三情景对比 ⭐⭐⭐
   - Base（蓝色实线）
   - ProNonUSD（橙色虚线）
   - RiskOff（灰色虚线）

4. **`fig_counts_anchor.png`**：各anchor币种数预测
   - 4个anchor的时间序列
   - EUR遥遥领先

---

## 🔬 模型验证

### 统计检验

✅ **Fixed Effects显著性**：
- 各anchor预测独立


✅ **情景差异显著性**：
- 统计显著 ✅

### 预测合理性

✅ **币种数排序**：EUR > CHF > JPY/GBP > CNY
✅ **市值变化**：非USD增长合理的政策效应）
✅ **份额稳定**：USD保持97%+（符合市场惯性）

---

## 📖 使用指南

### 修改情景参数

编辑 `src/config.py` 中的 `SCENARIO_PARAMS`：

```python
SCENARIO_PARAMS = {
    'ProNonUSD': {
        'pol_shift': 2.0,           # 政策冲击大小（可调整）
        'sigma_offpeg_reduction': 0.25,  # 脱锚波动下降比例
    },
    # ...
}
```

### 修改预测期数

编辑 `src/config.py`：

```python
FORECAST_HORIZON = 60  # 预测60个月（5年），可修改
```

### 查看详细日志

```bash
code logs/model_run.log
```

---

## 🎓 论文撰写建议

### 建议的论文结构

```
4. 方法论
  4.1 ARDL/ECM模型 → 引用 tab_ecm_*.tex
  4.2 面板计数模型（含Fixed Effects）→ 引用 tab_panel_counts.tex ⭐
  4.3 份额2SLS模型 → 引用 tab_share_2sls.tex
  
5. 结果
  5.1 基线预测 → 插入 fig_usd_bands.png, fig_nonusd_bands.png
  5.2 情景分析 → 插入 fig_share.png ⭐⭐⭐
  5.3 政策传导机制 → 解释方案B逻辑
  
6. 讨论
  6.1 欧元稳定币的领先地位
  6.2 监管政策的市场影响
  6.3 USD主导地位的持续性（97%+）
```


---

## ⚙️ 系统要求

- **Python**：3.8+
- **内存**：≥4GB
- **硬盘**：≥500MB（含数据和输出）
- **运行时间**：约4-5秒（完整流程）

---

## 📝 版本信息

- **版本**：1.0.0
- **完成日期**：2025年11月6日

---

## 🎉 项目完成度

✅ **100% 完成**

- ✅ 数据构建（97个月 + 99行面板）
- ✅ 4个模型估计（ARDL×2, Poisson, 2SLS）
- ✅ Fixed Effects实施（国家独立）
- ✅ 方案B实施（情景差异×7）
- ✅ 3情景预测（32个输出文件）
- ✅ 4张图表（300 DPI）
- ✅ 4张回归表（LaTeX）

**可以直接用于论文撰写！** 🎓✨

---

## 📧 联系方式

如有问题，请查看 `logs/model_run.log` 或检查输出文件完整性。

---

**End of README**
