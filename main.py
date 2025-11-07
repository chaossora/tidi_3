"""
主入口脚本
对应伪代码 §1：项目主入口

功能：串联所有模块，完成完整的分析流程
"""
import sys
from pathlib import Path
import logging
import time

# 添加src到路径
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import (
    DATA_PATHS, OUTPUT_FILES, LOG_FILE, LOG_FORMAT, LOG_LEVEL
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """主函数：完整的分析流程"""
    
    start_time = time.time()
    
    print("=" * 80)
    print(" " * 20 + "稳定币发展分析与预测系统")
    print(" " * 25 + "Stablecoin Analysis")
    print("=" * 80)
    print()
    
    try:
        # ==================== 步骤1：加载数据并构建月末主表 ====================
        logger.info("\n" + "=" * 80)
        logger.info("步骤1：加载数据并构建月末主表")
        logger.info("=" * 80)
        
        import importlib
        build_monthly = importlib.import_module('src.01_build_monthly')
        RawDataLoader = build_monthly.RawDataLoader
        build_monthly_master = build_monthly.build_monthly_master
        
        loader = RawDataLoader(DATA_PATHS)
        raw_data = loader.load_all()
        
        month = build_monthly_master(raw_data)
        
        # 保存月末主表
        month.to_csv(OUTPUT_FILES['month_master'], index=False)
        logger.info(f"[OK] 月末主表已保存: {OUTPUT_FILES['month_master']}")
        
        # ==================== 步骤2：构建锚定面板 ====================
        logger.info("\n" + "=" * 80)
        logger.info("步骤2：构建非美元锚定面板")
        logger.info("=" * 80)
        
        build_anchor = importlib.import_module('src.02_build_anchor_panel')
        build_anchor_panel = build_anchor.build_anchor_panel
        
        panel_result = build_anchor_panel(raw_data, month)
        panel = panel_result['panel']
        month = panel_result['month']  
        
        # 保存面板
        panel.to_csv(OUTPUT_FILES['panel_anchor'], index=False)
        logger.info(f"[OK] 锚定面板已保存: {OUTPUT_FILES['panel_anchor']}")
        
        # 更新月表
        month.to_csv(OUTPUT_FILES['month_master'], index=False)
        logger.info(f"[OK] 月表已更新")
        
        # ==================== 步骤3：构建数据集 ====================
        logger.info("\n" + "=" * 80)
        logger.info("步骤3：构建模型数据集")
        logger.info("=" * 80)
        
        estimate_models_module = importlib.import_module('src.03_estimate_models')
        DatasetBuilder = estimate_models_module.DatasetBuilder
        
        dataset_builder = DatasetBuilder(month, panel)
        ds_dict = dataset_builder.build()
        
        logger.info(f"[OK] 数据集构建完成")
        logger.info(f"  - 月度数据: {len(ds_dict['ds'])} 个月")
        logger.info(f"  - 面板数据: {len(ds_dict['panel'])} 条记录")
        
        # ==================== 步骤4：估计模型 ====================
        logger.info("\n" + "=" * 80)
        logger.info("步骤4：估计模型（ARDL/ECM、面板计数、2SLS）")
        logger.info("=" * 80)
        
        estimate_models = estimate_models_module.estimate_models
        
        models = estimate_models(ds_dict)
        
        logger.info(f"[OK] 模型估计完成")
        logger.info(f"  - USD模型: {'[OK]' if models.get('USD') else '[X]'}")
        logger.info(f"  - 非USD模型: {'[OK]' if models.get('nonUSD') else '[X]'}")
        logger.info(f"  - 计数模型: {'[OK]' if models.get('counts') else '[X]'}")
        logger.info(f"  - 份额模型: {'[OK]' if models.get('share') else '[X]'}")
        
        # ==================== 步骤5：预测与情景分析 ====================
        logger.info("\n" + "=" * 80)
        logger.info("步骤5：基线预测与情景分析")
        logger.info("=" * 80)
        
        forecast_scenarios = importlib.import_module('src.04_forecast_scenarios')
        run_forecast_and_scenarios = forecast_scenarios.run_forecast_and_scenarios
        
        forecast_results = run_forecast_and_scenarios(
            models, 
            ds_dict['ds'], 
            ds_dict['panel']
        )
        
        logger.info(f"[OK] 预测与情景分析完成")
        logger.info(f"  - 基线预测: {list(forecast_results['baseline'].keys())}")
        logger.info(f"  - 情景数量: {len(forecast_results['scenarios'])}")
        
        # ==================== 完成 ====================
        elapsed_time = time.time() - start_time
        
        print("\n" + "=" * 80)
        print(" " * 30 + "分析完成！")
        print("=" * 80)
        print(f"\n总用时: {elapsed_time:.1f} 秒")
        print(f"\n结果输出目录: {OUTPUT_FILES['month_master'].parent}")
        print("\n主要输出文件:")
        print(f"  - 月末主表: {OUTPUT_FILES['month_master'].name}")
        print(f"  - 锚定面板: {OUTPUT_FILES['panel_anchor'].name}")
        print(f"  - 预测结果: outputs/fcst_*.csv")
        print(f"  - 图表: outputs/fig_*.png")
        print(f"  - 日志: {LOG_FILE}")
        print("\n" + "=" * 80)
        
        return {
            'raw': raw_data,
            'month': month,
            'panel': panel,
            'datasets': ds_dict,
            'models': models,
            'forecasts': forecast_results
        }
        
    except Exception as e:
        logger.error(f"\n运行出错: {e}", exc_info=True)
        print(f"\n[X] 运行失败: {e}")
        print(f"详细错误信息请查看日志文件: {LOG_FILE}")
        return None


if __name__ == '__main__':
    print("""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                      稳定币发展分析与预测系统                                ║
    ║                    Stablecoin Development Analysis                         ║
    ║                                                                            ║
    ║  本系统实现了基于伪代码的完整分析流程：                                        ║
    ║  1. 数据加载与月末主表构建                                                   ║
    ║  2. 非美元锚定面板构建                                                       ║
    ║  3. 模型数据集准备                                                          ║
    ║  4. ARDL/ECM、面板计数、2SLS模型估计                                         ║
    ║  5. 基线预测与情景分析（Base/ProNonUSD/RiskOff）                             ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    results = main()
    
    if results is not None:
        print("\n[OK] 所有结果已保存到 outputs/ 和 build/ 目录")
        print("[OK] 您可以查看生成的CSV文件和图表")
    else:
        print("\n[X] 分析未完成，请检查错误信息")
        sys.exit(1)
