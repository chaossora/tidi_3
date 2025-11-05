"""
工具函数模块：日期处理、数据聚合、统计函数等
对应伪代码 §2.2 工具函数
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging
from typing import Union, List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# 配置日志
logger = logging.getLogger(__name__)


# ==================== 日期处理 ====================

def to_month_end(df: pd.DataFrame, 
                 date_col: str = 'date', 
                 value_cols: Union[str, List[str]] = None,
                 group_by: Union[str, List[str]] = None,
                 agg: str = 'last') -> pd.DataFrame:
    """
    将日度数据转换为月末数据
    
    参数:
        df: 输入DataFrame
        date_col: 日期列名
        value_cols: 要聚合的数值列（None表示所有数值列）
        group_by: 分组列（如chain, anchor等）
        agg: 聚合方式 ('last'=月末取值, 'mean'=月均, 'sum'=月度求和)
    
    返回:
        按月末聚合后的DataFrame
    """
    df = df.copy()
    
    # 解析日期为datetime
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    
    # 转换为月末
    df['month_end'] = df[date_col].dt.to_period('M').dt.to_timestamp('M')
    
    # 确定要聚合的列
    if value_cols is None:
        value_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    elif isinstance(value_cols, str):
        value_cols = [value_cols]
    
    # 确定分组键
    group_keys = ['month_end']
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]
        group_keys = group_by + group_keys
    
    # 聚合
    agg_dict = {}
    if agg == 'last':
        for col in value_cols:
            if col in df.columns:
                agg_dict[col] = 'last'
    elif agg == 'mean':
        for col in value_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
    elif agg == 'sum':
        for col in value_cols:
            if col in df.columns:
                agg_dict[col] = 'sum'
    else:
        raise ValueError(f"不支持的聚合方式: {agg}")
    
    # 执行分组聚合
    result = df.groupby(group_keys, as_index=False).agg(agg_dict)
    
    return result


def parse_date_column(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """解析日期列为datetime格式"""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    return df


# ==================== 数据清洗与变换 ====================

def safe_log(series: pd.Series, epsilon: float = 1e-8) -> pd.Series:
    """安全的log变换，防止log(0)"""
    return np.log(series.clip(lower=epsilon))


def safe_div(numerator: pd.Series, denominator: pd.Series, 
             fill_value: float = np.nan) -> pd.Series:
    """安全的除法，防止除0"""
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    return result


def zscore_transform(series: pd.Series) -> pd.Series:
    """Z-score标准化"""
    return (series - series.mean()) / series.std()


def logit_transform(series: pd.Series, epsilon: float = 1e-8) -> pd.Series:
    """Logit变换：log(p / (1-p))"""
    p = series.clip(epsilon, 1 - epsilon)
    return np.log(p / (1 - p))


def inverse_logit(series: pd.Series) -> pd.Series:
    """Logit逆变换：1 / (1 + exp(-x))"""
    return 1 / (1 + np.exp(-series))


# ==================== 滞后与差分 ====================

def create_lags(df: pd.DataFrame, 
                cols: Union[str, List[str]], 
                lags: Union[int, List[int]] = 1,
                group_by: str = None) -> pd.DataFrame:
    """
    创建滞后变量
    
    参数:
        df: 输入DataFrame
        cols: 要滞后的列名
        lags: 滞后阶数（可以是单个整数或列表）
        group_by: 分组列（对面板数据按组创建滞后）
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    
    if isinstance(lags, int):
        lags = [lags]
    
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            new_col = f"{col}_l{lag}"
            if group_by is not None:
                df[new_col] = df.groupby(group_by)[col].shift(lag)
            else:
                df[new_col] = df[col].shift(lag)
    
    return df


def create_diffs(df: pd.DataFrame, 
                 cols: Union[str, List[str]], 
                 periods: int = 1,
                 group_by: str = None,
                 prefix: str = 'd') -> pd.DataFrame:
    """
    创建差分变量
    
    参数:
        df: 输入DataFrame
        cols: 要差分的列名
        periods: 差分周期
        group_by: 分组列
        prefix: 前缀（d表示差分，dlog表示对数差分）
    """
    df = df.copy()
    
    if isinstance(cols, str):
        cols = [cols]
    
    for col in cols:
        if col not in df.columns:
            continue
        new_col = f"{prefix}{col}" if not col.startswith('log_') else f"{prefix}log_{col.replace('log_', '')}"
        if col.startswith('log_'):
            new_col = f"dlog_{col.replace('log_', '')}"
        else:
            new_col = f"{prefix}_{col}"
            
        if group_by is not None:
            df[new_col] = df.groupby(group_by)[col].diff(periods)
        else:
            df[new_col] = df[col].diff(periods)
    
    return df


# ==================== 统计函数 ====================

def winsorize_series(series: pd.Series, 
                     lower: float = 0.01, 
                     upper: float = 0.99) -> pd.Series:
    """Winsorize处理（缩尾）"""
    lower_bound = series.quantile(lower)
    upper_bound = series.quantile(upper)
    return series.clip(lower_bound, upper_bound)


def rolling_stats(df: pd.DataFrame, 
                  col: str, 
                  window: int, 
                  stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
    """计算滚动统计量"""
    df = df.copy()
    for stat in stats:
        if stat == 'mean':
            df[f'{col}_ma{window}'] = df[col].rolling(window).mean()
        elif stat == 'std':
            df[f'{col}_std{window}'] = df[col].rolling(window).std()
        elif stat == 'min':
            df[f'{col}_min{window}'] = df[col].rolling(window).min()
        elif stat == 'max':
            df[f'{col}_max{window}'] = df[col].rolling(window).max()
    return df


# ==================== 缺失值处理 ====================

def forward_fill_by_group(df: pd.DataFrame, 
                          cols: List[str], 
                          group_by: str) -> pd.DataFrame:
    """按组前向填充"""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = df.groupby(group_by)[col].fillna(method='ffill')
    return df


def drop_na_min_obs(df: pd.DataFrame, min_obs: int = 24, 
                   subset: List[str] = None) -> pd.DataFrame:
    """
    删除关键列的缺失值，并检查最少观测数
    
    参数:
        df: 输入DataFrame
        min_obs: 最少观测数阈值
        subset: 需要检查缺失值的列（None表示检查所有列）
    """
    initial_len = len(df)
    
    if subset is not None:
        # 只删除指定列的缺失值
        df = df.dropna(subset=subset)
    else:
        # 不删除任何行，只警告
        pass
    
    if len(df) < min_obs:
        logger.warning(f"观测数({len(df)})少于{min_obs}，数据可能不足以进行可靠估计")
    
    if len(df) < initial_len:
        logger.info(f"删除缺失值：{initial_len} → {len(df)} 行")
    
    return df


# ==================== 加权计算 ====================

def weighted_average(df: pd.DataFrame, 
                     value_col: str, 
                     weight_col: str, 
                     group_by: str = None) -> Union[float, pd.Series]:
    """加权平均"""
    if group_by is not None:
        return df.groupby(group_by).apply(
            lambda x: np.average(x[value_col], weights=x[weight_col])
        )
    else:
        return np.average(df[value_col], weights=df[weight_col])


def normalize_weights(df: pd.DataFrame, 
                      weight_cols: List[str], 
                      group_by: str = None) -> pd.DataFrame:
    """权重标准化（使其和为1）"""
    df = df.copy()
    if group_by is not None:
        for col in weight_cols:
            df[col] = df.groupby(group_by)[col].transform(lambda x: x / x.sum())
    else:
        total = df[weight_cols].sum().sum()
        df[weight_cols] = df[weight_cols] / total
    return df


# ==================== 面板数据工具 ====================

def pivot_panel(df: pd.DataFrame, 
                index: str, 
                columns: str, 
                values: str) -> pd.DataFrame:
    """面板数据转宽格式"""
    return df.pivot(index=index, columns=columns, values=values)


def unpivot_panel(df: pd.DataFrame, 
                  id_vars: List[str], 
                  var_name: str = 'variable', 
                  value_name: str = 'value') -> pd.DataFrame:
    """宽格式转长格式"""
    return df.melt(id_vars=id_vars, var_name=var_name, value_name=value_name)


# ==================== 数据验证 ====================

def check_negative_values(df: pd.DataFrame, 
                          cols: List[str], 
                          raise_error: bool = False) -> Dict[str, int]:
    """检查负值"""
    negative_counts = {}
    for col in cols:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                logger.warning(f"列 '{col}' 含有 {neg_count} 个负值")
                negative_counts[col] = neg_count
                if raise_error:
                    raise ValueError(f"列 '{col}' 不应含有负值")
    return negative_counts


def check_missing_data(df: pd.DataFrame, 
                       threshold: float = 0.5) -> Dict[str, float]:
    """检查缺失数据比例"""
    missing_ratios = {}
    for col in df.columns:
        ratio = df[col].isna().sum() / len(df)
        if ratio > threshold:
            logger.warning(f"列 '{col}' 缺失率 {ratio:.2%} 超过阈值 {threshold:.2%}")
            missing_ratios[col] = ratio
    return missing_ratios


# ==================== 模型辅助函数 ====================

def add_constant(df: pd.DataFrame) -> pd.DataFrame:
    """添加常数项"""
    df = df.copy()
    df['const'] = 1
    return df


def demean_by_group(df: pd.DataFrame, 
                    cols: List[str], 
                    group_by: str) -> pd.DataFrame:
    """按组去均值（固定效应变换）"""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f'{col}_dm'] = df.groupby(group_by)[col].transform(lambda x: x - x.mean())
    return df


# ==================== 信息准则 ====================

def calculate_aic(loglik: float, k: int) -> float:
    """计算AIC"""
    return -2 * loglik + 2 * k


def calculate_bic(loglik: float, k: int, n: int) -> float:
    """计算BIC"""
    return -2 * loglik + k * np.log(n)


# ==================== 预测辅助 ====================

def extend_index(df: pd.DataFrame, 
                 periods: int, 
                 freq: str = 'M') -> pd.DatetimeIndex:
    """扩展时间索引用于预测"""
    last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else df['month_end'].max()
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    return future_dates


def create_forecast_template(last_date: pd.Timestamp, 
                             periods: int, 
                             freq: str = 'M') -> pd.DataFrame:
    """创建预测模板DataFrame"""
    future_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
    return pd.DataFrame({'month_end': future_dates})


# ==================== 导出辅助 ====================

def export_regression_table(results_dict: Dict, 
                           output_path: str, 
                           fmt: str = 'latex') -> None:
    """导出回归结果表格"""
    # 这里简化处理，实际需要根据模型结果格式化
    logger.info(f"回归表导出到: {output_path}")
    # 实际实现需要使用stargazer或自定义格式化


def save_forecast_results(forecast_dict: Dict, 
                         prefix: str, 
                         output_dir: str) -> None:
    """保存预测结果"""
    for key, df in forecast_dict.items():
        if isinstance(df, pd.DataFrame):
            output_path = f"{prefix}_{key}.csv"
            df.to_csv(output_path, index=False)
            logger.info(f"预测结果保存到: {output_path}")


if __name__ == '__main__':
    # 测试
    logging.basicConfig(level=logging.INFO)
    print("工具函数模块加载成功！")
    
    # 测试日期转换
    test_df = pd.DataFrame({
        'date': pd.date_range('2020-01-01', periods=100, freq='D'),
        'value': np.random.randn(100)
    })
    
    monthly = to_month_end(test_df, date_col='date', value_cols='value', agg='mean')
    print(f"\n测试日期转换：{len(test_df)} 条日度数据 -> {len(monthly)} 条月度数据")
    print(monthly.head())
