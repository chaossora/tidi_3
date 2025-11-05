"""
数据加载与月末主表构建
对应伪代码 §2：原始数据加载与"月末主表"

功能:
1. 加载所有原始CSV数据
2. 统一日期格式并转换为月末
3. 构造美元/非美元稳定币规模
4. 计算宏观变量、链上摩擦、跨链流动等特征
5. 生成变换变量（log、z-score等）
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# 添加src到路径
sys.path.append(str(Path(__file__).parent))

from config import DATA_PATHS, MIN_OBS, EPSILON, CHAIN_NAME_MAPPING
from utils import (
    to_month_end, parse_date_column, safe_log, zscore_transform,
    create_lags, create_diffs, check_negative_values, drop_na_min_obs,
    add_constant, weighted_average, normalize_weights
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RawDataLoader:
    """原始数据加载器"""
    
    def __init__(self, data_paths: dict):
        self.data_paths = data_paths
        self.raw = {}
    
    def load_all(self) -> dict:
        """加载所有原始数据"""
        logger.info("开始加载原始数据...")
        
        for key, path in self.data_paths.items():
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    self.raw[key] = df
                    logger.info(f"[OK] 加载 {key}: {len(df)} 行, {len(df.columns)} 列")
                except Exception as e:
                    logger.error(f"[X] 加载 {key} 失败: {e}")
            else:
                logger.warning(f"[X] 文件不存在: {path}")
        
        logger.info(f"数据加载完成，共 {len(self.raw)} 个文件")
        return self.raw


class MonthlyMasterBuilder:
    """月末主表构建器"""
    
    def __init__(self, raw_data: dict):
        self.raw = raw_data
        self.month = None
        self.data_path = Path(__file__).parent.parent / 'data'  # 添加data路径
    
    def build(self) -> pd.DataFrame:
        """构建月末主表"""
        logger.info("=" * 60)
        logger.info("开始构建月末主表...")
        logger.info("=" * 60)
        
        # A) 美元锚定总规模
        S_USD = self._build_usd_scale()
        
        # B) 全部稳定币总规模
        S_ALL = self._build_total_scale()
        
        # C) 合并并计算非美元规模与份额
        self.month = self._merge_and_calculate_shares(S_USD, S_ALL)
        
        # D) 质量自检
        self._quality_check()
        
        # E) 加入宏观与市场变量
        self._add_macro_variables()
        
        # F) 加入链上摩擦与跨链流动
        self._add_onchain_metrics()
        
        # G) 生成常见变换
        self._create_transformations()
        
        # H) 清理与最终检查
        # 只删除关键列的缺失值
        key_cols = ['month_end', 'S_USD', 'S_nonUSD', 'share_USD']
        self.month = drop_na_min_obs(self.month, min_obs=MIN_OBS, subset=key_cols)
        
        logger.info(f"月末主表构建完成: {len(self.month)} 个月")
        logger.info(f"列: {list(self.month.columns)}")
        
        return self.month
    
    def _build_usd_scale(self) -> pd.DataFrame:
        """构建美元锚定稳定币规模（按月末）"""
        logger.info("构建美元稳定币规模 S_USD...")
        
        df = self.raw['chains_timeseries'].copy()
        
        # 按日期聚合各链的USD稳定币
        df_daily = df.groupby('date')['total_usd'].sum().reset_index()
        
        # 转为月末
        S_USD = to_month_end(df_daily, date_col='date', value_cols='total_usd', agg='last')
        S_USD = S_USD.rename(columns={'total_usd': 'S_USD'})
        
        logger.info(f"  S_USD 规模: {len(S_USD)} 个月, 范围: {S_USD['S_USD'].min():.2e} ~ {S_USD['S_USD'].max():.2e}")
        
        return S_USD
    
    def _build_total_scale(self) -> pd.DataFrame:
        """构建全部稳定币总规模（按月末）"""
        logger.info("构建总稳定币规模 S_ALL...")
        
        df = self.raw['total_stablecoins_mcap'].copy()
        
        # 转为月末
        S_ALL = to_month_end(df, date_col='date', 
                            value_cols='total_stablecoins_mcap_usd', agg='last')
        S_ALL = S_ALL.rename(columns={'total_stablecoins_mcap_usd': 'S_ALL'})
        
        logger.info(f"  S_ALL 规模: {len(S_ALL)} 个月, 范围: {S_ALL['S_ALL'].min():.2e} ~ {S_ALL['S_ALL'].max():.2e}")
        
        return S_ALL
    
    def _merge_and_calculate_shares(self, S_USD: pd.DataFrame, 
                                    S_ALL: pd.DataFrame) -> pd.DataFrame:
        """合并并计算非美元规模与份额（使用真实分peg数据）"""
        logger.info("合并数据并计算份额...")
        
        # 加载按锚定类型拆分的市值数据
        try:
            mcap_by_peg = pd.read_csv(self.data_path / 'total_stablecoins_mcap_by_peg.csv')
            mcap_by_peg['date'] = pd.to_datetime(mcap_by_peg['date'])
            
            # 分别聚合USD和非USD
            usd_daily = mcap_by_peg[mcap_by_peg['peg'] == 'peggedUSD'].copy()
            nonusd_daily = mcap_by_peg[mcap_by_peg['peg'] != 'peggedUSD'].groupby('date')['total_stablecoins_mcap_usd'].sum().reset_index()
            nonusd_daily.columns = ['date', 'S_nonUSD_daily']
            
            # 转为月末
            S_nonUSD = to_month_end(nonusd_daily, date_col='date', 
                                   value_cols='S_nonUSD_daily', agg='last')
            S_nonUSD = S_nonUSD.rename(columns={'S_nonUSD_daily': 'S_nonUSD'})
            
            logger.info(f"  ✅ 使用真实非USD市值数据")
            logger.info(f"  S_nonUSD数据: {len(S_nonUSD)} 个月")
            
        except FileNotFoundError:
            logger.warning("  ⚠️  未找到total_stablecoins_mcap_by_peg.csv，使用3%估算")
            S_nonUSD = None
        
        # 合并USD和ALL
        month = pd.merge(S_USD, S_ALL, on='month_end', how='outer')
        
        # 合并非USD（如果有真实数据）
        if S_nonUSD is not None:
            month = pd.merge(month, S_nonUSD, on='month_end', how='left')
        
        month = month.sort_values('month_end').reset_index(drop=True)
        
        # 填充缺失（前向填充）
        month['S_USD'] = month['S_USD'].fillna(method='ffill')
        month['S_ALL'] = month['S_ALL'].fillna(method='ffill')
        
        # 计算或填充S_nonUSD
        if S_nonUSD is not None and 'S_nonUSD' in month.columns:
            # 对于早期无数据的时期，使用3%估算作为fallback
            early_missing = month['S_nonUSD'].isna()
            if early_missing.any():
                month.loc[early_missing, 'S_nonUSD'] = month.loc[early_missing, 'S_ALL'] * 0.03
                logger.info(f"  早期{early_missing.sum()}个月使用3%估算补充")
            
            # 前向填充剩余缺失
            month['S_nonUSD'] = month['S_nonUSD'].fillna(method='ffill')
            
            logger.info(f"  S_nonUSD范围: {month['S_nonUSD'].min():.2e} ~ {month['S_nonUSD'].max():.2e}")
            logger.info(f"  S_nonUSD/S_ALL占比: {(month['S_nonUSD']/month['S_ALL']).mean():.2%} (均值)")
        else:
            # Fallback：使用3%估算
            month['S_nonUSD'] = month['S_ALL'] * 0.03
            logger.info(f"  使用市场占比（3%）估算S_nonUSD")
            logger.info(f"  S_nonUSD范围: {month['S_nonUSD'].min():.2e} ~ {month['S_nonUSD'].max():.2e}")
        
        # 重新计算S_ALL，确保 S_USD + S_nonUSD = S_ALL
        month['S_ALL'] = month['S_USD'] + month['S_nonUSD']
        
        # 计算份额
        month['share_USD'] = month['S_USD'] / month['S_ALL']
        month['share_nonUSD'] = month['S_nonUSD'] / month['S_ALL']
        
        return month
    
    def _quality_check(self):
        """质量自检"""
        logger.info("执行数据质量检查...")
        
        # 检查负值
        neg_counts = check_negative_values(self.month, ['S_nonUSD'], raise_error=False)
        
        if 'S_nonUSD' in neg_counts:
            neg_ratio = neg_counts['S_nonUSD'] / len(self.month)
            logger.warning(f"  ⚠ S_nonUSD 有 {neg_counts['S_nonUSD']} 个负值 ({neg_ratio:.1%})")
            logger.warning("  建议：检查 S_ALL 与 S_USD 数据源口径是否一致")
            
            # 简易修正：将负值设为小正数
            self.month.loc[self.month['S_nonUSD'] < 0, 'S_nonUSD'] = EPSILON
        
        # 检查份额范围
        share_out_of_range = ((self.month['share_USD'] < 0) | 
                             (self.month['share_USD'] > 1)).sum()
        if share_out_of_range > 0:
            logger.warning(f"  ⚠ share_USD 超出 [0,1] 范围: {share_out_of_range} 条")
            self.month['share_USD'] = self.month['share_USD'].clip(0, 1)
    
    def _add_macro_variables(self):
        """加入宏观与市场变量"""
        logger.info("加入宏观变量...")
        
        # DEX成交量（月度求和）
        if 'dex_volume_total' in self.raw:
            dex = to_month_end(self.raw['dex_volume_total'], 
                              date_col='date', value_cols='total_volume_usd', agg='sum')
            self.month = pd.merge(self.month, dex, on='month_end', how='left')
            self.month = self.month.rename(columns={'total_volume_usd': 'DEX'})
            logger.info(f"  [OK] DEX 成交量")
        
        # VIX（月均）
        if 'vix' in self.raw:
            vix = to_month_end(self.raw['vix'], 
                              date_col='date', value_cols='VIXCLS', agg='mean')
            self.month = pd.merge(self.month, vix, on='month_end', how='left')
            self.month = self.month.rename(columns={'VIXCLS': 'VIX'})
            logger.info(f"  [OK] VIX")
        
        # 美元指数（月末）
        if 'usd_index' in self.raw:
            dxy = to_month_end(self.raw['usd_index'], 
                              date_col='date', value_cols='DTWEXBGS', agg='last')
            self.month = pd.merge(self.month, dxy, on='month_end', how='left')
            self.month = self.month.rename(columns={'DTWEXBGS': 'DXY'})
            logger.info(f"  [OK] 美元指数 DXY")
    
    def _add_onchain_metrics(self):
        """加入链上摩擦与跨链流动指标"""
        logger.info("加入链上指标...")
        
        # 费用加权平均
        if 'fees_by_chain' in self.raw and 'chains_timeseries' in self.raw:
            fees = self._calculate_weighted_fees()
            self.month = pd.merge(self.month, fees, on='month_end', how='left')
            logger.info(f"  [OK] 费用加权")
        
        # 跨链净流动
        if 'bridge_flows_by_chain' in self.raw:
            bridge = self._calculate_bridge_flows()
            self.month = pd.merge(self.month, bridge, on='month_end', how='left')
            logger.info(f"  [OK] 跨链流动")
    
    def _calculate_weighted_fees(self) -> pd.DataFrame:
        """计算费用的加权平均（按链的USD稳定币存量权重）"""
        fees_df = self.raw['fees_by_chain'].copy()
        chains_df = self.raw['chains_timeseries'].copy()
        
        # 统一chain名称为首字母大写格式
        fees_df['chain'] = fees_df['chain'].str.title()
        chains_df['chain'] = chains_df['chain'].str.title()
        
        # 月末聚合
        fees_m = to_month_end(fees_df, date_col='date', 
                             value_cols='fees_usd',  # 修正列名
                             group_by='chain', agg='sum')
        
        chains_m = to_month_end(chains_df, date_col='date', 
                               value_cols='total_usd', 
                               group_by='chain', agg='last')
        
        # 合并（使用outer join保留所有月份）
        merged = pd.merge(chains_m, fees_m, on=['chain', 'month_end'], how='left')
        
        # 填充缺失的fees（早期月份可能没有fees数据）
        merged['fees_usd'] = merged['fees_usd'].fillna(0)
        
        # 计算权重（各链存量占比）
        total_by_month = merged.groupby('month_end')['total_usd'].sum()
        merged = pd.merge(merged, total_by_month.rename('total_all'), 
                         on='month_end', how='left')
        merged['weight'] = merged['total_usd'] / merged['total_all']
        
        # 加权求和
        merged['weighted_fees'] = merged['fees_usd'] * merged['weight']
        result = merged.groupby('month_end')['weighted_fees'].sum().reset_index()
        result = result.rename(columns={'weighted_fees': 'FEE_wgt'})
        
        return result
    
    def _calculate_bridge_flows(self) -> pd.DataFrame:
        """计算跨链净流动及强度"""
        bridge_df = self.raw['bridge_flows_by_chain'].copy()
        
        # 月度聚合
        bridge_m = to_month_end(bridge_df, date_col='date',
                               value_cols=['deposit_usd', 'withdraw_usd'],
                               group_by='chain', agg='sum')
        
        # 计算净流入
        bridge_m['net_flow'] = bridge_m['deposit_usd'] - bridge_m['withdraw_usd']
        
        # 按月求和
        result = bridge_m.groupby('month_end')['net_flow'].sum().reset_index()
        result = result.rename(columns={'net_flow': 'BRIDGE_net'})
        
        return result
    
    def _create_transformations(self):
        """创建变换变量"""
        logger.info("创建变换变量...")
        
        # Log变换
        self.month['log_S_USD'] = safe_log(self.month['S_USD'])
        self.month['log_S_nonUSD'] = safe_log(self.month['S_nonUSD'])
        self.month['log_S_ALL'] = safe_log(self.month['S_ALL'])
        
        if 'DEX' in self.month.columns:
            self.month['log_DEX'] = safe_log(self.month['DEX'])
        
        # 差分
        self.month = create_diffs(self.month, ['log_S_USD', 'log_S_nonUSD'], 
                                 periods=1, prefix='d')
        
        # Z-score标准化
        for col in ['VIX', 'DXY', 'FEE_wgt']:
            if col in self.month.columns:
                self.month[f'z_{col}'] = zscore_transform(self.month[col])
        
        # 跨链流动强度（相对于总规模）
        if 'BRIDGE_net' in self.month.columns:
            self.month['BRIDGE_intensity'] = (
                self.month['BRIDGE_net'] / self.month['S_ALL'].shift(1)
            )
            self.month['z_BRIDGE'] = zscore_transform(
                self.month['BRIDGE_intensity'].fillna(0)
            )
        
        # 添加常数项
        self.month = add_constant(self.month)
        
        logger.info(f"  [OK] 创建 {len([c for c in self.month.columns if c.startswith('log_') or c.startswith('z_') or c.startswith('d')])} 个变换变量")


def build_monthly_master(raw_data: dict) -> pd.DataFrame:
    """
    主函数：构建月末主表
    
    参数:
        raw_data: 原始数据字典
    
    返回:
        月末主表 DataFrame
    """
    builder = MonthlyMasterBuilder(raw_data)
    month = builder.build()
    return month


if __name__ == '__main__':
    # 测试运行
    logger.info("=" * 60)
    logger.info("测试：数据加载与月末主表构建")
    logger.info("=" * 60)
    
    # 加载数据
    loader = RawDataLoader(DATA_PATHS)
    raw = loader.load_all()
    
    # 构建月末主表
    month = build_monthly_master(raw)
    
    # 显示结果
    print("\n" + "=" * 60)
    print("月末主表预览:")
    print("=" * 60)
    print(month.head(10))
    print(f"\n形状: {month.shape}")
    print(f"\n列名:\n{list(month.columns)}")
    print(f"\n基本统计:\n{month.describe()}")
    
    # 保存
    from config import OUTPUT_FILES
    output_path = OUTPUT_FILES['month_master']
    month.to_csv(output_path, index=False)
    logger.info(f"\n[OK] 月末主表已保存到: {output_path}")
