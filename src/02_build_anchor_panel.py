"""
非美元锚定面板构建
对应伪代码 §3：非美元"锚定-面板"（按 anchor：EUR/JPY/HKD/...）

功能:
1. 统计每月在市品类数量
2. 计算脱锚波动与尾部概率
3. 映射政策哑变量到各anchor
4. 计算汇率周期影响
5. 生成面板数据用于后续模型
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from config import (
    DATA_PATHS, ANCHOR_TO_JURISDICTION, MIN_OBS, EPSILON
)
from utils import (
    to_month_end, parse_date_column, safe_log, zscore_transform,
    create_lags, check_negative_values, drop_na_min_obs
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnchorPanelBuilder:
    """非美元锚定面板构建器"""
    
    def __init__(self, raw_data: dict, month_data: pd.DataFrame):
        self.raw = raw_data
        self.month = month_data
        self.panel = None
    
    def build(self) -> dict:
        """构建锚定面板"""
        logger.info("=" * 60)
        logger.info("开始构建非美元锚定面板...")
        logger.info("=" * 60)
        
        # A) 统计每月在市品类数量与脱锚指标
        self.panel = self._build_anchor_stats()
        
        # B) 加入汇率周期
        self._add_fx_cycle()
        
        # C) 映射政策指数
        self._map_policy_to_anchor()
        
        # D) 创建滞后变量
        self._create_lags()
        
        # E) 计算全局非美元品类数量
        N_nonUSD = self._calculate_total_nonusd_count()
        
        # F) 清理
        # 只删除关键列的缺失值
        key_cols = ['anchor', 'month_end', 'N_anchor']
        self.panel = drop_na_min_obs(self.panel, min_obs=MIN_OBS // 2, subset=key_cols)
        
        logger.info(f"锚定面板构建完成: {len(self.panel)} 行")
        logger.info(f"唯一锚定货币: {self.panel['anchor'].nunique()}")
        
        return {
            'panel': self.panel,
            'month': self.month,  # 更新后的月表（含N_nonUSD）
            'N_nonUSD': N_nonUSD
        }
    
    def _build_anchor_stats(self) -> pd.DataFrame:
        """构建各anchor的统计指标"""
        logger.info("计算各anchor的在市数量与脱锚指标...")
        
        df = self.raw['nonusd_offpeg_timeseries'].copy()
        
        # 解析日期
        df['date'] = pd.to_datetime(df['date'], utc=True)
        df['month_end'] = df['date'].dt.to_period('M').dt.to_timestamp('M')
        
        # 按 anchor [X] month_end 分组统计
        stats_list = []
        
        for (anchor, month_end), group in df.groupby(['anchor', 'month_end']):
            stat = {
                'anchor': anchor,
                'month_end': month_end,
                'N_anchor': group['symbol'].nunique(),  # 当月独立品类数
                'sigma_offpeg': group['offpeg_pct'].std(),  # 脱锚波动
                'mean_offpeg': group['offpeg_pct'].mean(),  # 平均脱锚
                'tail1': (group['offpeg_pct'].abs() > 0.01).mean(),  # >1%的概率
                'tail2': (group['offpeg_pct'].abs() > 0.02).mean(),  # >2%的概率
            }
            stats_list.append(stat)
        
        panel = pd.DataFrame(stats_list)
        panel = panel.sort_values(['anchor', 'month_end']).reset_index(drop=True)
        
        logger.info(f"  统计完成: {len(panel)} 条记录")
        logger.info(f"  涵盖 {panel['anchor'].nunique()} 个锚定货币")
        
        return panel
    
    def _add_fx_cycle(self):
        """加入汇率周期（USD/anchor的月度变动）"""
        logger.info("加入汇率周期...")
        
        if 'fx_anchor_to_usd' not in self.raw:
            logger.warning("  未找到汇率数据，跳过")
            return
        
        fx_df = self.raw['fx_anchor_to_usd'].copy()
        
        # 转为月末
        fx_m = to_month_end(fx_df, date_col='date', 
                           value_cols='usd_per_anchor',
                           group_by='ccy', agg='last')
        
        # 按货币计算log变动
        fx_change_list = []
        for ccy in fx_m['ccy'].unique():
            ccy_data = fx_m[fx_m['ccy'] == ccy].copy()
            ccy_data = ccy_data.sort_values('month_end')
            ccy_data['log_usd_per_anchor'] = safe_log(ccy_data['usd_per_anchor'])
            ccy_data['dlog_usd_per_anchor'] = ccy_data['log_usd_per_anchor'].diff()
            fx_change_list.append(ccy_data[['ccy', 'month_end', 'dlog_usd_per_anchor']])
        
        fx_change = pd.concat(fx_change_list, ignore_index=True)
        
        # 合并到面板（anchor == ccy）
        self.panel = pd.merge(
            self.panel,
            fx_change.rename(columns={'ccy': 'anchor'}),
            on=['anchor', 'month_end'],
            how='left'
        )
        
        logger.info(f"  [OK] 汇率周期已加入")
    
    def _map_policy_to_anchor(self):
        """映射增强政策变量到各anchor"""
        logger.info("映射增强政策指数到各anchor...")
        
        # 尝试加载新的增强政策数据
        enhanced_policy_path = DATA_PATHS.get('policy_enhanced', 
            Path('data/stablecoin_policy_dummies_enhanced_monthly_2019_2025.csv'))
        
        try:
            policy_df = pd.read_csv(enhanced_policy_path)
            logger.info(f"  ✓ 加载增强政策数据: {len(policy_df)} 条记录")
            use_enhanced = True
        except FileNotFoundError:
            logger.warning(f"  未找到增强政策数据: {enhanced_policy_path}")
            if 'policy_monthly' in self.raw:
                policy_df = self.raw['policy_monthly'].copy()
                logger.info("  使用原有政策数据")
                use_enhanced = False
            else:
                logger.warning("  未找到任何政策数据，跳过")
                self.panel['Pol_anchor'] = 0
                self.panel['cum_events'] = 0
                self.panel['policy_stage'] = 0
                self.panel['any_impulse'] = 0
                return
        
        # 确保日期格式
        if 'month_end' in policy_df.columns:
            policy_df['month_end'] = pd.to_datetime(policy_df['month_end'])
        elif 'month' in policy_df.columns:
            policy_df['month_end'] = pd.to_datetime(policy_df['month'])
        elif 'date' in policy_df.columns:
            policy_df['month_end'] = pd.to_datetime(policy_df['date'])
        else:
            logger.warning(f"  政策数据列名: {list(policy_df.columns)}")
            logger.warning("  政策数据缺少日期列")
            self.panel['Pol_anchor'] = 0
            self.panel['cum_events'] = 0
            self.panel['policy_stage'] = 0
            self.panel['ban_dummy'] = 0  # 新增
            self.panel['any_impulse'] = 0
            return
        
        # 为每个anchor×月计算政策变量
        pol_list = []
        cum_events_list = []
        policy_stage_list = []
        ban_dummy_list = []  # 新增
        any_impulse_list = []
        
        for idx, row in self.panel.iterrows():
            anchor = row['anchor']
            month_end = row['month_end']
            
            # 查找对应的jurisdiction
            jurisdictions = ANCHOR_TO_JURISDICTION.get(anchor, [])
            
            if not jurisdictions:
                pol_list.append(0)
                cum_events_list.append(0)
                policy_stage_list.append(0)
                ban_dummy_list.append(0)  # 新增
                any_impulse_list.append(0)
                continue
            
            # 收集该anchor在该月的政策指标（多个jurisdiction取平均）
            pol_vals = []
            cum_vals = []
            stage_vals = []
            ban_vals = []  # 新增：禁令指标
            impulse_vals = []
            
            for juris in jurisdictions:
                # 使用年-月匹配（忽略具体日期）
                mask = ((policy_df['jurisdiction'] == juris) & 
                       (policy_df['month_end'].dt.year == month_end.year) &
                       (policy_df['month_end'].dt.month == month_end.month))
                if mask.sum() > 0:
                    pol_row = policy_df[mask].iloc[0]
                    
                    if use_enhanced:
                        # 使用增强政策变量
                        policy_stage = pol_row.get('policy_stage', 0)
                        ban_dummy = pol_row.get('ban_dummy', 0)
                        
                        # ✅ 关键修改：区分正面和负面政策
                        # Pol_anchor = 净政策友好度 (policy_stage - ban_dummy)
                        # 范围: -1 (纯禁令) 到 3 (生效且无禁令)
                        net_policy = policy_stage - ban_dummy
                        
                        pol_vals.append(net_policy)  # 主要政策指标使用净值
                        cum_vals.append(pol_row.get('cum_events', 0))  # 保留原始累计事件
                        stage_vals.append(policy_stage)  # 保留正面政策阶段
                        ban_vals.append(ban_dummy)  # 新增：禁令指标
                        impulse_vals.append(pol_row.get('any_impulse', 0))
                    else:
                        # 使用原有政策强度
                        pol_val = (
                            pol_row.get('active_regime_dummy', 0) + 
                            pol_row.get('non_us_currency_covered_dummy', 0)
                        )
                        pol_vals.append(pol_val)
                        cum_vals.append(0)
                        stage_vals.append(0)
                        ban_vals.append(0)
                        impulse_vals.append(0)
            
            # 取平均（如果有多个jurisdiction）
            pol_list.append(np.mean(pol_vals) if pol_vals else 0)
            cum_events_list.append(np.mean(cum_vals) if cum_vals else 0)
            policy_stage_list.append(np.mean(stage_vals) if stage_vals else 0)
            ban_dummy_list.append(np.max(ban_vals) if ban_vals else 0)  # 禁令用max（只要有一个禁令就算）
            any_impulse_list.append(np.max(impulse_vals) if impulse_vals else 0)  # 脉冲事件用max
        
        # 保存到面板
        self.panel['Pol_anchor'] = pol_list  # 现在是净政策友好度 (policy_stage - ban_dummy)
        if use_enhanced:
            self.panel['cum_events'] = cum_events_list
            self.panel['policy_stage'] = policy_stage_list
            self.panel['ban_dummy'] = ban_dummy_list  # 新增：禁令指标
            self.panel['any_impulse'] = any_impulse_list
            logger.info(f"  [OK] 增强政策指数已映射（区分正负面政策）")
            logger.info(f"     Pol_anchor（净政策友好度=stage-ban）均值: {np.mean(pol_list):.3f}, 范围: [{np.min(pol_list):.1f}, {np.max(pol_list):.1f}]")
            logger.info(f"     policy_stage 范围: {np.min(policy_stage_list):.0f} ~ {np.max(policy_stage_list):.0f}")
            logger.info(f"     ban_dummy 数量: {np.sum(ban_dummy_list):.0f} 条记录有禁令")
            logger.info(f"     any_impulse 事件数: {np.sum(any_impulse_list):.0f}")
        else:
            logger.info(f"  [OK] 政策指数已映射，均值: {np.mean(pol_list):.3f}")
    
    def _create_lags(self):
        """创建滞后变量"""
        logger.info("创建滞后变量...")
        
        # 基础滞后变量
        lag_vars = ['Pol_anchor', 'sigma_offpeg', 'mean_offpeg', 'tail1', 'tail2', 'N_anchor']
        
        # 如果使用增强政策数据，添加新变量
        if 'cum_events' in self.panel.columns:
            lag_vars.extend(['cum_events', 'policy_stage', 'ban_dummy'])  # 新增 ban_dummy
            # any_impulse不需要滞后（脉冲事件）
        
        for var in lag_vars:
            if var in self.panel.columns:
                self.panel = create_lags(self.panel, var, lags=1, group_by='anchor')
        
        logger.info(f"  [OK] 创建 {len([v for v in lag_vars if v in self.panel.columns])} 个变量的滞后项")
    
    def _calculate_total_nonusd_count(self) -> pd.DataFrame:
        """计算全局非美元品类数量（跨anchor求和）"""
        logger.info("计算全局非美元品类数量...")
        
        N_nonUSD = self.panel.groupby('month_end')['N_anchor'].sum().reset_index()
        N_nonUSD = N_nonUSD.rename(columns={'N_anchor': 'N_nonUSD'})
        
        # 合并到月表
        self.month = pd.merge(self.month, N_nonUSD, on='month_end', how='left')
        self.month['N_nonUSD'] = self.month['N_nonUSD'].fillna(0)
        
        logger.info(f"  [OK] N_nonUSD 范围: {N_nonUSD['N_nonUSD'].min():.0f} ~ {N_nonUSD['N_nonUSD'].max():.0f}")
        
        return N_nonUSD


def build_anchor_panel(raw_data: dict, month_data: pd.DataFrame) -> dict:
    """
    主函数：构建锚定面板
    
    参数:
        raw_data: 原始数据字典
        month_data: 月末主表
    
    返回:
        包含panel和更新后month的字典
    """
    builder = AnchorPanelBuilder(raw_data, month_data)
    result = builder.build()
    return result


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("测试：锚定面板构建")
    logger.info("=" * 60)
    
    # 加载数据
    from config import OUTPUT_FILES
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_monthly", Path(__file__).parent / "01_build_monthly.py")
    build_monthly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_monthly)
    
    loader = build_monthly.RawDataLoader(DATA_PATHS)
    raw = loader.load_all()
    
    # 构建月表
    month = build_monthly.build_monthly_master(raw)
    
    # 构建锚定面板
    result = build_anchor_panel(raw, month)
    panel = result['panel']
    month_updated = result['month']
    
    # 显示结果
    print("\n" + "=" * 60)
    print("锚定面板预览:")
    print("=" * 60)
    print(panel.head(15))
    print(f"\n形状: {panel.shape}")
    print(f"\n列名:\n{list(panel.columns)}")
    print(f"\n各anchor统计:")
    print(panel.groupby('anchor').agg({
        'N_anchor': ['mean', 'max'],
        'sigma_offpeg': 'mean',
        'Pol_anchor': 'mean'
    }))
    
    # 保存
    output_path = OUTPUT_FILES['panel_anchor']
    panel.to_csv(output_path, index=False)
    logger.info(f"\n[OK] 锚定面板已保存到: {output_path}")
    
    # 更新月表
    month_path = OUTPUT_FILES['month_master']
    month_updated.to_csv(month_path, index=False)
    logger.info(f"[OK] 月表已更新: {month_path}")
