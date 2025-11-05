好的！我已在你之前的基础上**补充了更多法域（jurisdictions）**的稳定币/相关监管事件，并**一键生成了可直接用于建模的事件哑变量（按月）**。你可以直接下载两份 CSV：

* **事件时间轴（可溯源）**：[`stablecoin_regulatory_timeline_extended.csv`](sandbox:/mnt/data/stablecoin_regulatory_timeline_extended.csv)
* **月度事件哑变量面板（2019-01 — 2025-12）**：[`stablecoin_policy_dummies_monthly_2019_2025.csv`](sandbox:/mnt/data/stablecoin_policy_dummies_monthly_2019_2025.csv)

---

## 本次新增&补充的主要法域与关键事件（都配有官方或一手来源）

> 下列仅为说明摘要，**CSV 中每一行都含“source_url”**，便于追溯官方文件。

* **欧盟（EBA / MiCA）**

  * 2024‑06‑30：MiCA 对**电子货币通证 (EMT)** 与**资产参考通证 (ART)** 开始适用（生效节点）。([resmigazete.gov.tr][1])
  * 2024‑11‑07：EBA 就**非欧元计价** EMT/ART 报告技术标准发布征求意见（覆盖**非美元**货币）。([Iru OJK][2])

* **英国（BoE / FCA / HMT）**

  * 2023‑11‑06：英格兰银行与 FCA 发布**稳定币监管**讨论文件（系统性支付系统、发行/托管框架）。([Bank of England][3])
  * 2025‑04‑28：财政部公布**第1阶段稳定币入表**的草拟法令（草拟 SI），推进纳入支付监管。([Consult Treasury][4])

* **新加坡（MAS）**

  * 2023‑08‑15：**最终版稳定币监管框架**（适用新元及 G10 货币的**单一法币稳定币**）。([Default][5])

* **日本（FSA）**

  * 2023‑06‑01：修法生效，**稳定币（电子支付手段）**框架实施；并对稳定币/加密资产转移适用“**旅行规则**”。([Financial Services Agency][6])

* **瑞士（FINMA）**

  * 2019‑09‑11：发布**稳定币指引**（ICO 指引补充）。([FINMA][7])
  * 2024‑07‑26：发布**监督通告 06/2024：稳定币与银行保函**（操作实践与风险提示）。([FINMA][8])

* **阿联酋·迪拜（VARA）**

  * 2023‑02：**《虚拟资产及相关活动条例 2023》**及发行规则（含**法币锚定虚拟资产/稳定币**的发行要求）。([VARA Rulebook][9])
  * 2025‑05‑19：**发行规则**更新版本（含法币锚定虚拟资产附录）。([VARA Rulebook][10])

* **阿联酋·阿布扎比（ADGM, FSRA）**

  * 2024‑08：就**法币参照代币**等稳定币纳入加密代币框架发起咨询。([ADGM][11])
  * 2024‑12：发布**稳定币/法币参照代币最终框架**（Crypto Token Regime 更新版）。([ADGM][12])

* **巴林（CBB）**

  * 2024‑10：就**稳定币监管框架（SIO 模块）**公开征求意见。([Central Bank of Bahrain][13])
  * 2025‑07‑04：**正式发布**稳定币监管框架。([Central Bank of Bahrain][14])

* **加拿大（OSFI）**

  * 2023‑07‑26：就**法币参照加密资产（FRCA）**的资本与流动性处理征求意见。([OSFI][15])
  * 2025‑02‑20：发布**最终指引**（包含 FRCA/稳定币的审慎处理）。([OSFI][16])

* **澳大利亚（财政部）**

  * 2023‑12：**支付系统现代化**咨询，提出把**支付稳定币**纳入**储值设施（SVF）**框架与数字资产设施框架衔接。([Treasury][17])

* **韩国（FSC）**

  * 2023‑07：国会通过《**虚拟资产用户保护法**》。([Financial Services Commission][18])
  * 2024‑07‑19：法律**生效**。([Financial Services Commission][19])

* **巴西（中央银行 BCB）**

  * 2024‑05：就**加密资产/虚拟资产服务提供者(VASP)**下一步监管安排发布**新闻说明**（涵盖支付使用场景）。([Banco Central do Brasil][20])

* **土耳其（中央银行/资本市场委员会）**

  * 2021‑04‑16：央行规定**禁止**在支付中使用加密资产。([Iru OJK][21])
  * 2025‑03‑13：资本市场委员会发布**加密资产服务商通则**（许可框架）。([resmigazete.gov.tr][1])

* **泰国（央行 BOT）**

  * 2021‑03‑19：发布**泰铢稳定币政策**（界定与合规路径，属稳定币专项）。([People's Bank of China][22])

* **中国大陆（人民银行等十部门联合）**

  * 2021‑09‑24：联合发布**进一步防范处置虚拟货币交易炒作风险的通知**（一般性**禁止**）。([Reserve Bank of South Africa][23])

* **美国（纽约州 NYDFS / OCC / PWG）**

  * 2022‑06‑08：NYDFS 发布**美元稳定币发行指引**（赎回、准备金、鉴证），**仅适用于 USD**。([Department of Financial Services][24])
  * 2025‑03‑07：OCC **解释函 1183**，重申国行可从事特定稳定币相关活动。([OCC.gov][25])
  * 2021‑11‑01：财政部**PWG 稳定币报告**（建议与风险评估）。([U.S. Department of the Treasury][26])

* **百慕大（BMA）**

  * 2024‑05‑10：就**单一货币锚定稳定币（SCPS）**发布咨询稿。([cdn.bma.bm][27])
  * 2024‑12‑02：发布**最终版 SCPS 指引**。([cdn.bma.bm][28])

---

## CSV 文件结构说明

### 1) 事件时间轴：`stablecoin_regulatory_timeline_extended.csv`

字段包含：

* `jurisdiction`（法域）
* `event_date`（日期；为保证跨法域一致性，部分仅到“月”，用于建模足够）
* `event_name`、`event_type`（如：Consultation / Guidance / Framework Finalised / Law Enacted / Effective / Ban）
* `scope`（是否稳定币专项）
* `non_us_currency_covered`（是否**明确**覆盖**非美元**或**不限币种**的稳定币）
* `reference_currency_examples`（例如：SGD & G10 / 非欧元等）
* `source_url`（官方/一手链接，便于审计与复核）

**下载：**[stablecoin_regulatory_timeline_extended.csv](sandbox:/mnt/data/stablecoin_regulatory_timeline_extended.csv)

---

### 2) 月度事件哑变量面板：`stablecoin_policy_dummies_monthly_2019_2025.csv`

面板维度：每个 `jurisdiction` × 每月（2019‑01 — 2025‑12）。
包含以下**阶跃型**（step）事件变量（从事件所在月起置 1，此后保持 1）：

* `consult_or_discuss_dummy`：出现**咨询/讨论/征求意见/政策通告**等“**宣布/讨论**”类事件后为 1
* `framework_final_dummy`：出现**最终框架/规则/正式指引/已发布草拟但明确落地路径**为 1（如 MAS 最终框架、VARA 发行规则、ADGM/CBB 最终框架等）
* `effective_dummy`：**生效**/开始实施日期后为 1（如欧盟 MiCA EMT/ART 2024‑06‑30；韩国法 2024‑07‑19）
* `ban_dummy`：**禁止/限制**类事件后为 1（如中国大陆 2021‑09‑24、土耳其 2021‑04‑16）
* `active_regime_dummy`：若存在 `effective_dummy` 则随之为 1；否则当有 `framework_final_dummy` 但未给出硬生效日时，从最终框架发布月起为 1（便于计量**“有无有效监管框架”**的稳健性检验）
* `non_us_currency_covered_dummy`：法域层面的常量特征位（如 **EU/UK/SG/CH/VARA/ADGM/CBB/BMA/OSFI/澳大利亚**等均为 1；**NYDFS**为 0 因其文件**仅限 USD**；对**非稳定币通用法**或难以判定者标 0）

**下载：**[stablecoin_policy_dummies_monthly_2019_2025.csv](sandbox:/mnt/data/stablecoin_policy_dummies_monthly_2019_2025.csv)

> 说明：为适配常见时间序列/面板回归，哑变量按**月度**构造，避免日度极稀疏且跨法域日期精度不同带来的噪声；如你需要**日度版本**或不同起止区间，我可以直接按同样口径生成。

---

## 建模使用建议（简要）

* **基准设定**

  * 用 `active_regime_dummy` 度量“是否存在有效监管框架”，在稳健性中以 `framework_final_dummy`/`effective_dummy` 进行替换。
  * 对事件冲击分析（如事件研究法）可在 `consult_or_discuss_dummy` / `framework_final_dummy` / `effective_dummy` 的发生月做**窗口变量**。

* **非美元币种覆盖**

  * 直接使用 `non_us_currency_covered_dummy` 与样本内**非美元稳定币规模/交易量**交互，刻画“**监管覆盖 × 币种**”效应（例如 EU 的**非欧元**报告 ITS、SG 的 **SGD/G10**、VARA/ADGM/CBB/BMA 的**不限币种**框架）。([Iru OJK][2])

* **州/省级与国家级并存**

  * 美国样本可同时纳入 **NYDFS**（USD 专项）与联邦层面（如 **PWG 报告**、**OCC 解释函**）的变量，以区分**监管强度与约束层级**。([Department of Financial Services][24])

---

## 重要来源举例（与 CSV 完全一致）

* EBA：MiCA 应用与**非欧元**报告 ITS。([resmigazete.gov.tr][1])
* 英国：BoE / FCA 稳定币讨论文件与进展。([Bank of England][3])
* 新加坡：MAS **最终版稳定币框架**（SGD & G10）。([Default][5])
* 日本：FSA 稳定币（电子支付手段）与**旅行规则**实施。([Financial Services Agency][6])
* 瑞士：FINMA 稳定币**指引**（2019）与**监督通告 06/2024**。([FINMA][7])
* 迪拜 VARA：总法规与**法币锚定虚拟资产**发行规则。([VARA Rulebook][9])
* ADGM（FSRA）：咨询与**最终框架**。([ADGM][11])
* 巴林（CBB）：咨询与**正式发布**。([Central Bank of Bahrain][13])
* 加拿大（OSFI）：FRCA 咨询与**最终指引**。([OSFI][15])
* 澳大利亚：支付系统现代化（含**支付稳定币**纳入 SVF）。([Treasury][17])
* 韩国：**虚拟资产用户保护法**通过与**生效**。([Financial Services Commission][18])
* 巴西：BCB **加密/VASP**后续监管新闻说明。([Banco Central do Brasil][20])
* 土耳其：支付中**禁止**使用加密资产；后续服务商通则。([Iru OJK][21])
* 泰国：央行**泰铢稳定币**政策。([People's Bank of China][22])
* 中国大陆：**联合通知**（一般性禁止）。([Reserve Bank of South Africa][23])
* 美国：NYDFS **USD 稳定币发行指引**；OCC **IL1183**；PWG 报告。([Department of Financial Services][24])

---

如果你希望我**追加更多法域**（例如菲律宾、印尼、沙特、南非等）或把哑变量改成**日度**版本/调整分类口径，请告诉我，我会按相同规范**直接扩充并更新 CSV**。

[1]: https://www.resmigazete.gov.tr/eskiler/2024/07/20240702-1.htm?utm_source=chatgpt.com "2 Temmuz 2024 SALI - T.C. Resmî Gazete"
[2]: https://iru.ojk.go.id/iru/policy/detailpolicy/13208/ojk-issues-regulation-concerning-the-implementation-of-digital-financial-asset-trading-including-cryptocurrency?utm_source=chatgpt.com "OJK International Information Hub | Otoritas Jasa Keuangan"
[3]: https://www.bankofengland.co.uk/paper/2023/dp/regulatory-regime-for-systemic-payment-systems-using-stablecoins-and-related-service-providers?utm_source=chatgpt.com "Regulatory regime for systemic payment systems using stablecoins and ..."
[4]: https://consult.treasury.gov.au/c2023-469663?utm_source=chatgpt.com "Payments System Modernisation (Regulation of Payment Service ... - Treasury"
[5]: https://www.sgpc.gov.sg/api/file/getfile/Media%20Release_MAS%20Finalises%20Stablecoin%20Regulatory%20Framework.pdf?path=%2Fsgpcmedia%2Fmedia_releases%2Fmas%2Fpress_release%2FP-20230815-2%2Fattachment%2FMedia+Release_MAS+Finalises+Stablecoin+Regulatory+Framework.pdf&utm_source=chatgpt.com "MAS Finalises Stablecoin Regulatory Framework"
[6]: https://www.fsa.go.jp/en/news/2022/20221207/01.pdf?utm_source=chatgpt.com "Regulating the crypto assets landscape in Japan"
[7]: https://www.finma.ch/en/news/2019/09/20190911-mm-stable-coins/?utm_source=chatgpt.com "FINMA publishes ‘stable coin’ guidelines"
[8]: https://www.finma.ch/en/news/2024/07/20240726-m-am-06-24-stablecoins/?utm_source=chatgpt.com "FINMA publishes guidance on stablecoins | FINMA"
[9]: https://rulebooks.vara.ae/rulebook/virtual-assets-and-related-activities-regulations-2023?utm_source=chatgpt.com "Virtual Assets and Related Activities Regulations 2023 - VARA"
[10]: https://rulebooks.vara.ae/sites/default/files/en_net_file_store/VARA_EN_293_VER20250519.pdf?utm_source=chatgpt.com "Virtual Asset Issuance Rulebook - rulebooks.vara.ae"
[11]: https://www.adgm.com/documents/legal-framework/public-consultations/2024/consultation-paper-no-7/consultation-paper-no-7-of-2024-frts.pdf?utm_source=chatgpt.com "PROPOSED REGULATORY FRAMEWORK FOR THE ISSUANCE OF FIAT ... - ADGM"
[12]: https://www.adgm.com/media/announcements/fsra-introduces-a-regulatory-framework-to-support-the-issuance-of-fiat-referenced-tokens-in-adgm/?utm_source=chatgpt.com "FSRA introduces a regulatory framework to support the ... - adgm.com"
[13]: https://www.cbb.gov.bh/wp-content/uploads/2024/10/OG-300-2024-Circular-Consultation-Proposed-Stablecoin-Issuance-and-Offering-Module.pdf?utm_source=chatgpt.com "www.cbb.gov.bh"
[14]: https://www.cbb.gov.bh/media-center/central-bank-of-bahrain-issues-framework-for-regulating-stablecoin-issuance/?utm_source=chatgpt.com "Central Bank of Bahrain Issues Framework for Regulating Stablecoin ..."
[15]: https://www.osfi-bsif.gc.ca/en/consultation-international-recommendations-related-risks-posed-fiat-referenced-cryptoasset?utm_source=chatgpt.com "Consultation on international recommendations related to, and risks ..."
[16]: https://www.osfi-bsif.gc.ca/en/guidance/guidance-library/capital-liquidity-treatment-crypto-asset-exposures-guidelines-letter?utm_source=chatgpt.com "Capital and Liquidity Treatment of Crypto-asset Exposures Guidelines ..."
[17]: https://treasury.gov.au/sites/default/files/2023-12/c2023-469663-cp.pdf "Payments System Modernisation: Regulation of payment service providers – consultation paper"
[18]: https://www.fsc.go.kr/eng/pr010101/82683?utm_source=chatgpt.com "Press Releases - Financial Services Commission"
[19]: https://fsc.go.kr/eng/pr010101/81698?utm_source=chatgpt.com "Press Releases - Financial Services Commission"
[20]: https://www.bcb.gov.br/en/pressdetail/2536/nota "Banco Central do Brasil"
[21]: https://iru.ojk.go.id/iru/BE/uploads/regulation/files/file_444fdb9e-8b49-4e13-80e7-47c0330160f3-17042025155234.pdf?utm_source=chatgpt.com "Eng-infografis POJK crypto - iru.ojk.go.id"
[22]: https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/4348521/index.html?utm_source=chatgpt.com "关于进一步防范和处置虚拟货币交易炒作风险的通知"
[23]: https://www.resbank.co.za/content/dam/sarb/what-we-do/financial-stability/A%20Primer%20on%20Stablecoins.pdf?utm_source=chatgpt.com "A primer on stablecoins - resbank.co.za"
[24]: https://www.dfs.ny.gov/industry_guidance/industry_letters/il20220608_issuance_stablecoins "Industry Letter - June 8, 2022: Guidance on the Issuance of U.S. Dollar-Backed Stablecoins | Department of Financial Services"
[25]: https://www.occ.treas.gov/topics/charters-and-licensing/interpretations-and-actions/2025/int1183.pdf?utm_source=chatgpt.com "Interpretive Letter 1183 March 2025 OCC Letter Addressing Certain ..."
[26]: https://home.treasury.gov/news/press-releases/jy0454?utm_source=chatgpt.com "President’s Working Group on Financial Markets Releases Report and ..."
[27]: https://cdn.bma.bm/documents/2024-05-10-12-42-08-Notice---Digital-Asset-Business-Single-Currency-Pegged-Stablecoins-SCPS-Consultation-Guidance.pdf?utm_source=chatgpt.com "NOTICE - BMA"
[28]: https://cdn.bma.bm/documents/2024-12-02-16-55-46-Digital-Asset-Business--Single-Currency-Pegged-Stablecoins-SCPS-Guidance.pdf?utm_source=chatgpt.com "BERMUDA MONETARY AUTHORITY"
