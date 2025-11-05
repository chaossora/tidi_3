太棒了！你已经把脚本导出的 **CSV** 都放进项目里了。下面给你一份“**数据字典 + 建模用途**”，逐个文件说明**它是什么、每一列代表什么、能做什么**。为便于核对，我把每个文件背后的**官方/权威接口**也一起标注出来（见每段后的参考链接）。

---

## Ⅰ. 稳定币存量与链上分布

### 1) `stablecoins_list.csv` — 稳定币清单

**来源**：DeFiLlama 稳定币 API `/stablecoins`（含基础元数据） 。([DefiLlama API Docs][1])
**列说明**

* `id`：Llama 内部稳定币资产 ID。
* `name`：资产全名（如 *Euro Coin*）。
* `symbol`：资产符号（如 `EURC`、`XSGD`）。
* `pegType`：锚定类型（如 `peggedUSD`、`peggedEUR`、`peggedJPY` …）。
* `pegMechanism`：锚定机制（法币储备、加密抵押、算法等）。
* `gecko_id`：CoinGecko 的代币 ID（可用于价格别名，例如 `coingecko:eurc`）。
* `chains`：该稳定币出现过的链名（逗号分隔）。
  **用途**
* 作为“币种主表”，贯穿**偏离、分布、收益率**等子任务。
* 用 `pegType` 识别**非美元锚定**币清单。
* 用 `gecko_id`（或链+合约）作为**价格时间序列**的抓取别名。
  （参考：DeFiLlama **Stablecoins** 文档与页面展示字段。([DefiLlama API Docs][1])）

---

### 2) `chains_snapshot.csv` — 各链稳定币分布快照

**来源**：`/stablecoinchains`（各链稳定币存量的最新快照）。([DefiLlama API Docs][1])
**列说明**

* `chain`：链名（如 `Ethereum`、`Tron`、`Solana` …）。
* `gecko_chain_id`：该链在第三方数据中的 ID（如对接 Gecko/站内映射）。
* `peggedUSD`：**当前**在该链上**USD 锚定稳定币**的“流通/在链存量（USD 计）”。
  **用途**
* 做**空间分布**：不同链的稳定币“份额/集中度”。
* 与 `fees_by_chain.csv`、`dex_volume_*` **按链**合并，解释分布由来（拥堵/活跃度的结构性影响）。
  （对应网页“Stablecoins Circulating by Chain”。([DeFi Llama][2])）

---

### 3) `chains_timeseries.csv` — 各链分布的**日度时序**

**来源**：`/stablecoincharts2/{chain}`（失败时回退 `/stablecoincharts/{chain}`），取其中的 `totalCirculatingUSD.peggedUSD` 时间序列。([DefiLlama API Docs][1])
**列说明**

* `date`：UTC 日（ISO 字符串）。
* `chain`：链名。
* `total_usd`：当日该链的**USD 锚定稳定币存量（USD 计）**。
  **用途**
* 做**时序分析**：链上迁移、再配置、压力传导（可与桥净流/费用关联）。
* 构造**份额**、**增长率**、**跨链再配置指标**。

---

### 4) `total_stablecoins_mcap.csv` — 全部稳定币总市值（日度）

**来源**：由 `chains_timeseries.csv` 聚合（跨链求和）。
**列说明**

* `date`：UTC 日。
* `total_stablecoins_mcap_usd`：当天全网稳定币总量（USD 计）。
  **用途**
* 作为“系统规模”或“需求总量”的基线；拟合**宏观变量**（美元指数、VIX 等）对存量的影响。

---

## Ⅱ. 非美元稳定币“锚定偏离”（Off‑Peg）

### 5) `nonusd_offpeg_snapshot.csv` — 非美元稳定币**快照**偏离

**来源**：DeFiLlama **stablecoins** 清单里附带的**现价**（免 Key），结合 `fx_anchor_to_usd.csv` 的**锚定币兑 USD**。([DefiLlama API Docs][1])
**列说明**

* `symbol`：稳定币符号（如 `EURC`、`XSGD`）。
* `anchor`：锚定货币（如 `EUR`、`SGD`）。
* `price_usd`：稳定币**当前**美元价格。
* `anchor_usd`：**1 单位锚定货币**折算成**多少 USD**（来自 ECB EXR）。
* `offpeg_pct`：偏离百分比 = `price_usd / anchor_usd − 1`（%）。
  **用途**
* 检测**瞬时脱锚**；给“事件日/事件窗”筛样本。
  （ECB EXR 的系列键与取法：`EXR/D.USD.EUR.SP00.A` 等。([ECB Data Portal][3])）

---

### 6) `nonusd_offpeg_timeseries.csv` — 非美元稳定币**日度**偏离

**来源**：价格时序来自 DeFiLlama **coin‑prices** API（用 `coingecko:{id}` 或链+合约别名，免 CG Key）按**历史日期批量**抓取；锚定币兑 USD 来自 `fx_anchor_to_usd.csv`。([DefiLlama API Docs][4])
**列说明**

* `date`：UTC 日。
* `symbol`：稳定币符号。
* `anchor`：锚定货币。
* `price_usd`：当日稳定币美元价。
* `anchor_usd`：当日**1 单位锚定币**的美元价（EXR 转换，含周末前向填充）。
* `offpeg_pct`：当日偏离百分比（%）。
  **用途**
* 估计**偏离的回归速度**（ECM/AR 或半衰期）、极端值概率、与**收益/流动性**的联动。
  （coin‑prices API 支持 `/prices/historical/{timestamp}/{coins}` 与 `/coins/batchHistorical`，一次请求多币或多时点。([DefiLlama API Docs][4])）

---

## Ⅲ. 链上活跃度与流动性

### 7) `fees_by_chain.csv` — **链级费用/收入**（日度）

**来源**：DeFiLlama **Fees** 概览 `/api/overview/fees/{chain}?dataType=dailyFees`（注：个别链的 slug 与展示名不同，如 *OP Mainnet*）。([DefiLlama API Docs][4])
**列说明**

* `date`：UTC 日。
* `chain`：链名（按你脚本的映射写回）。
* `fees_usd`：该链当日**总费用**（USD 计，多数链近似代表“拥堵/使用强度”）。
  **用途**
* 解释**稳定币迁移/分布**（费用上升 → 迁出压力）；作为**摩擦项**进入回归。
  （API 返回体包含 `totalDataChart` 序列。([DefiLlama API Docs][4])）

---

### 8) `dex_volume_total.csv` — 全网 DEX **总成交量**（日度）

### 9) `dex_volume_by_protocol.csv` — DEX **按协议**成交量（日度）

**来源**：DeFiLlama **Aggregated DEX Volumes** `/api/overview/dexs?dataType=dailyVolume`；`totalDataChart` 为总量，`totalDataChartBreakdown` 为协议分解。([DefiLlama API Docs][4])
**列说明**

* `dex_volume_total.csv`

  * `date`：UTC 日。
  * `total_volume_usd`：全网 DEX 成交合计（USD）。
* `dex_volume_by_protocol.csv`

  * `date`：UTC 日。
  * `protocol`：协议（Uniswap、Curve、Pancake 等）。
  * `volume_usd`：该协议当日成交额（USD）。
    **用途**
* 作为**需求/流动性强度**的高频代理变量；配合偏离或各链份额做解释或响应。

---

### 10) `bridge_flows_by_chain.csv` — **跨链桥流量/净流**（日度）

**来源**：`https://bridges.llama.fi/bridgevolume/{ChainDisplayName}`（大小写敏感；例如 `Solana`、`Arbitrum`、`Ethereum`）。同页亦提供 `.csv` 导出。([DeFi Llama][5])
**列说明**

* `date`：UTC 日（时间戳已转换为日期）。
* `chain`：链名（展示名）。
* `deposit_usd`：当日**跨入**该链的桥接额（USD）。
* `withdraw_usd`：当日**跨出**该链的桥接额（USD）。
* `netflow_usd`：净流 = `deposit_usd − withdraw_usd`。
  **用途**
* 直接衡量**跨链再配置**；与 `chains_timeseries.csv` 对照验证“**净流→存量变化**”。
  （Bridges 总览页“Net Flows by Chain / .csv”可与结果对照。([DeFi Llama][5])）

---

## Ⅳ. 外汇 / 宏观 / 市场基准

### 11) `fx_anchor_to_usd.csv` — 锚定币兑 USD（**日度，前向填充**）

**来源**：ECB EXR（以欧元为分母）：

* 先取 `USD/EUR`：`EXR/D.USD.EUR.SP00.A`；
* 再取 `ANCHOR/EUR`：如 `EXR/D.JPY.EUR.SP00.A`；
* 计算 **`USD/ANCHOR = (USD/EUR) / (ANCHOR/EUR)`**；再对自然日 **ffill**。([ECB Data Portal][3])
  **列说明**
* `date`：UTC 日。
* `ccy`：锚定币 ISO 代码（`EUR`、`JPY`、`HKD`、`GBP`、`CNY`、`CHF`、`AUD`、`CAD`、`SGD` …）。
* `usd_per_anchor`：**1 单位锚定币**等于**多少 USD**。
  **用途**
* 计算所有非美元稳币的**理论锚定价**与偏离；做“汇率冲击”的解释变量。

---

### 12) `vix.csv` — 市场波动率（恐慌指数）

**来源**：FRED 系列 **VIXCLS**（CBOE VIX）。([FRED][6])
**列说明**

* `date`：交易日。
* `VIXCLS`：VIX 指数收盘值。
  **用途**
* 作为**风险厌恶**的外生冲击；常与稳定币脱锚/挤兑事件相关联。

---

### 13) `usd_index.csv` — **贸易加权**美元指数

**来源**：FRED 系列 **DTWEXBGS**（Nominal Broad U.S. Dollar Index）。([FRED][7])
**列说明**

* `date`：日期。
* `DTWEXBGS`：广义名义美元指数。
  **用途**
* 衡量美元整体强弱；解释“**非美元锚定**稳定币的系统性偏离”。

---

### 14) `btc_eth_market.csv` — BTC/ETH 市场基准（免 Key 兜底版）

**来源**：Binance **现货** K 线 `/api/v3/klines` 日频聚合（若你启用的是仅 Binance 兜底版本）。([Binance Developer Center][8])
**列说明**

* `asset`：`bitcoin` 或 `ethereum`。
* `date`：UTC 日。
* `price_usd`：收盘价（USDT 近似 USD）。
* `market_cap`：若用 Binance 兜底通常为 `None`；若你在别处补充，会有值。
* `volume_24h`：当日 **quote 成交额**（USDT 计，可近似 USD）。
  **用途**
* 作为“**加密市场状态**”的控制变量（市况/风险偏好对稳币需求/偏离的影响）。

---

## Ⅴ. 怎么把这些数据“连起来用”（建议的 Join 与特征）

* **跨链分布 × 费用 × 桥净流**

  * 用键 `date + chain` 关联：`chains_timeseries` ↔ `fees_by_chain` ↔ `bridge_flows_by_chain`。
  * 构造：`Δ份额`、`净流/存量`、`费用移动均值/分位数`；回归**再配置**或**份额变化**。
  * 端点说明见：Fees 概览、Bridges 统计、Stablecoins by Chain。([DefiLlama API Docs][4])

* **非美元稳币偏离 × 宏观外生变量**

  * 用键 `date + symbol` 关联：`nonusd_offpeg_timeseries` ↔ `vix` ↔ `usd_index` ↔ `fx_anchor_to_usd`。
  * 构造：`offpeg_pct` 的 **AR/ECM**、“高分位超调”指示、**事件窗均值**；检验 VIX/美元指数冲击。
  * 参考：FRED VIX、DTWEXBGS 索引释义。([FRED][6])

* **快照到时序**

  * `nonusd_offpeg_snapshot` 可筛出“**偏离最大的前 N**”再去看其 `nonusd_offpeg_timeseries` 的**回归速度**。
  * 汇率对齐采用 EXR“**欧元作分母**”的规范取法。([ECB Data Portal][3])

* **DEX 量 × 偏离/桥净流**

  * 用键 `date` 聚合或 `date + protocol/chain` 粗配：看成交活跃对偏离**放大/收敛**的边际作用。
  * 端点文档示例（支持 `dataType=dailyVolume` 与 breakdown）。([DefiLlama API Docs][4])

---

## Ⅵ. 数据质量与清洗小贴士

1. **时间基准**：上述接口多为 **UTC**；ECB EXR 为工作日数据，已按自然日 **前向填充**对齐价格。([ECB Data Portal][9])
2. **链名/slug**：Fees 的链 slug 与展示名可能不同（如 **Optimism → op‑mainnet**）；Bridges 的 `{ChainDisplayName}` **大小写敏感**（如 `Solana`）。([DefiLlama API Docs][4])
3. **DEX breakdown 结构**：`/api/overview/dexs` 的 `totalDataChartBreakdown` 既可能是“**数组形式**的时间点 + 协议映射”，也可能是“对象数组”；你的脚本已做兼容。([DefiLlama API Docs][4])
4. **价格来源**：为避免外部限流，本工具优先用 DeFiLlama 的 **coin‑prices**（免 Key），支持 `coingecko:{id}` 或 `chain:address`；必要时可用 `/batchHistorical` **一次多时点**批量提速。([DefiLlama API Docs][4])
5. **单位**：所有 **USD** 值默认 **名义美元**；如需常量对比，建议叠加 CPI 或使用实际利率（可另加 IMF/FRED/OECD 源）。

---

### 如果你的目录里还出现了其它文件（可选）

* `dex_volume_chain.csv`（若你扩展了按链 DEX 量）：`date, chain, volume_usd`，来源 `/api/overview/dexs/{chain}`。([DefiLlama API Docs][4])
* 任何 `*_rates_*.csv` / `imf_cpi_monthly.csv`：若你启用了这些模块，分别对应 **利率** 或 **通胀**（IMF IFS/央行/财政部/交易所源）。

---

如果你愿意，我也可以帮你把这些说明做成一份 **README（中英双语）** 或 **数据字典表（CSV/Markdown）**，直接放进你的项目根目录，便于队友查阅 👇

* 按“文件 → 字段 → 含义 → 单位 → 典型 join → 建模用途”的结构排版；
* 附上所有对应接口的参考链接（就是上面这些）。

[1]: https://api-docs.defillama.com/?utm_source=chatgpt.com "API Docs - DefiLlama"
[2]: https://defillama.com/stablecoins/chains?utm_source=chatgpt.com "Stablecoins Circulating - DefiLlama"
[3]: https://data.ecb.europa.eu/help/api/data?utm_source=chatgpt.com "Data | ECB Data Portal"
[4]: https://api-docs.defillama.com/llms.txt "api-docs.defillama.com"
[5]: https://defillama.com/bridges?utm_source=chatgpt.com "Bridge Volume - DefiLlama"
[6]: https://fred.stlouisfed.org/series/VIXCLS/?utm_source=chatgpt.com "CBOE Volatility Index: VIX (VIXCLS) | FRED | St. Louis Fed"
[7]: https://fred.stlouisfed.org/series/DTWEXBGS/?utm_source=chatgpt.com "Nominal Broad U.S. Dollar Index (DTWEXBGS) | FRED | St. Louis Fed"
[8]: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints?utm_source=chatgpt.com "Market Data endpoints | Binance Open Platform"
[9]: https://data.ecb.europa.eu/help/data-examples?utm_source=chatgpt.com "Data examples | ECB Data Portal"
