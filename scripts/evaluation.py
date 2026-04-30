"""
模型评估脚本
包含离线评估和在线A/B测试
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, List

from src.utils.common import gauc, ndcg_at_k, diversity_score, coverage_score, print_metrics


def offline_evaluation(predictions: pd.DataFrame, 
                       item_features: pd.DataFrame) -> Dict[str, float]:
    """
    离线评估
    
    Args:
        predictions: 包含user_id, item_id, label, pred_score的DataFrame
        item_features: 物品特征DataFrame
        
    Returns:
        评估指标字典
    """
    print("开始离线评估...")
    
    metrics = {}
    
    # 1. AUC (整体)
    auc = roc_auc_score(predictions['label'], predictions['pred_score'])
    metrics['AUC'] = auc
    
    # 2. GAUC (分组AUC)
    gauc_score = gauc(
        predictions['user_id'].values,
        predictions['label'].values,
        predictions['pred_score'].values
    )
    metrics['GAUC'] = gauc_score
    
    # 3. Log Loss
    logloss = log_loss(predictions['label'], predictions['pred_score'])
    metrics['LogLoss'] = logloss
    
    # 4. NDCG@K
    ndcg_scores = []
    for user_id, group in predictions.groupby('user_id'):
        if len(group) < 10:
            continue
        y_true = group['label'].tolist()
        y_pred = group['pred_score'].tolist()
        ndcg = ndcg_at_k(y_true, y_pred, k=10)
        ndcg_scores.append(ndcg)
    
    metrics['NDCG@10'] = np.mean(ndcg_scores)
    
    # 5. Recall@K
    recall_scores = []
    for user_id, group in predictions.groupby('user_id'):
        # 按预测分数排序，取top-50
        group_sorted = group.sort_values('pred_score', ascending=False)
        top_k = group_sorted.head(50)
        
        # 计算召回率
        relevant_items = set(group[group['label'] == 1]['item_id'])
        recommended_items = set(top_k['item_id'])
        
        if len(relevant_items) > 0:
            recall = len(relevant_items & recommended_items) / len(relevant_items)
            recall_scores.append(recall)
    
    metrics['Recall@50'] = np.mean(recall_scores)
    
    # 6. 多样性 (基于推荐列表)
    recommendations = []
    for user_id, group in predictions.groupby('user_id'):
        top_items = group.sort_values('pred_score', ascending=False).head(10)['item_id'].tolist()
        recommendations.append(top_items)
    
    diversity = diversity_score(recommendations, item_features)
    metrics['Diversity'] = diversity
    
    # 7. 覆盖率
    total_items = item_features['item_id'].nunique()
    coverage = coverage_score(recommendations, total_items)
    metrics['Coverage'] = coverage
    
    # 8. 准确率指标
    # 将预测分数二值化
    predictions['pred_label'] = (predictions['pred_score'] > 0.5).astype(int)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics['Accuracy'] = accuracy_score(predictions['label'], predictions['pred_label'])
    metrics['Precision'] = precision_score(predictions['label'], predictions['pred_label'])
    metrics['Recall'] = recall_score(predictions['label'], predictions['pred_label'])
    metrics['F1'] = f1_score(predictions['label'], predictions['pred_label'])
    
    return metrics


def online_ab_test(control_group_data: pd.DataFrame,
                  treatment_group_data: pd.DataFrame) -> Dict[str, Dict]:
    """
    在线A/B测试评估
    
    Args:
        control_group_data: 对照组数据 (原策略)
        treatment_group_data: 实验组数据 (新策略)
        
    Returns:
        对比结果
    """
    print("开始A/B测试分析...")
    
    def calculate_online_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """计算在线指标"""
        metrics = {}
        
        # 1. CTR (点击率)
        metrics['CTR'] = data['is_click'].mean()
        
        # 2. 完播率
        if 'is_finish' in data.columns:
            metrics['FinishRate'] = data['is_finish'].mean()
        
        # 3. 人均观看时长
        if 'watch_time' in data.columns:
            metrics['AvgWatchTime'] = data.groupby('user_id')['watch_time'].sum().mean()
        
        # 4. 人均点击数
        metrics['AvgClicksPerUser'] = data.groupby('user_id')['is_click'].sum().mean()
        
        # 5. 互动率（点赞、分享等）
        if 'is_like' in data.columns:
            metrics['LikeRate'] = data['is_like'].mean()
        
        if 'is_share' in data.columns:
            metrics['ShareRate'] = data['is_share'].mean()
        
        # 6. 人均曝光数
        metrics['AvgImpressionPerUser'] = data.groupby('user_id').size().mean()
        
        return metrics
    
    # 计算两组指标
    control_metrics = calculate_online_metrics(control_group_data)
    treatment_metrics = calculate_online_metrics(treatment_group_data)
    
    # 计算提升
    comparison = {}
    for metric_name in control_metrics.keys():
        control_value = control_metrics[metric_name]
        treatment_value = treatment_metrics[metric_name]
        
        if control_value > 0:
            lift = (treatment_value - control_value) / control_value * 100
        else:
            lift = 0
        
        comparison[metric_name] = {
            'control': control_value,
            'treatment': treatment_value,
            'lift_%': lift
        }
    
    # 统计显著性检验
    from scipy import stats
    
    # CTR的卡方检验
    control_clicks = control_group_data['is_click'].sum()
    control_total = len(control_group_data)
    treatment_clicks = treatment_group_data['is_click'].sum()
    treatment_total = len(treatment_group_data)
    
    contingency_table = [
        [control_clicks, control_total - control_clicks],
        [treatment_clicks, treatment_total - treatment_clicks]
    ]
    
    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    
    comparison['statistical_test'] = {
        'chi2': chi2,
        'p_value': p_value,
        'is_significant': p_value < 0.05
    }
    
    return comparison


def print_ab_test_results(comparison: Dict[str, Dict]):
    """打印A/B测试结果"""
    print("\n" + "="*70)
    print("A/B测试结果对比")
    print("="*70)
    print(f"{'指标':<20} {'对照组':>15} {'实验组':>15} {'提升':>15}")
    print("-"*70)
    
    for metric_name, values in comparison.items():
        if metric_name == 'statistical_test':
            continue
        
        control = values['control']
        treatment = values['treatment']
        lift = values['lift_%']
        
        print(f"{metric_name:<20} {control:>15.4f} {treatment:>15.4f} {lift:>14.2f}%")
    
    print("-"*70)
    
    # 统计显著性
    stat_test = comparison.get('statistical_test', {})
    p_value = stat_test.get('p_value', 1.0)
    is_significant = stat_test.get('is_significant', False)
    
    print(f"\n统计显著性检验: p-value = {p_value:.4f}")
    if is_significant:
        print("✓ 实验组与对照组存在显著差异 (p < 0.05)")
    else:
        print("✗ 实验组与对照组无显著差异 (p >= 0.05)")
    
    print("="*70 + "\n")


from src.data.kuairand_loader import (
    load_kuairand_splits, 
    load_user_features, 
    load_video_features
)


def main():
    """主评估流程"""
    
    print("="*55)
    print("加载 KuaiRand-1K 数据进行评估演示")
    print("="*55)
    
    # 1. 加载数据
    user_features = load_user_features()
    # 采样一部分视频特征，避免内存占用过大
    item_features = load_video_features(sample_n=100000).reset_index()
    item_features = item_features.rename(columns={'video_id': 'item_id'})
    
    train_log, val_log, test_log = load_kuairand_splits(
        use_early_log_as_train=False,
        val_ratio=0.2,
        verbose=True
    )
    
    # 2. 离线评估示例 (使用测试集作为模拟预测结果)
    # 实际场景中，这里应该是模型对测试集的预测分数
    print("\n[示例] 对测试集(log_random)进行离线指标计算...")
    predictions = test_log.copy()
    # 模拟一个预测分数 (在真实标签基础上加点噪声)
    predictions['pred_score'] = predictions['is_click'] * 0.6 + np.random.beta(2, 5, len(predictions)) * 0.4
    predictions['label'] = predictions['is_click']
    
    offline_metrics = offline_evaluation(predictions, item_features)
    print_metrics(offline_metrics)
    
    # 3. 在线A/B测试分析示例
    print("\n[示例] A/B测试对比分析 (log_standard vs log_random)...")
    # 这里只是演示逻辑，通常是对比两个实验组
    comparison = online_ab_test(val_log, test_log)
    print_ab_test_results(comparison)
    
    print("\n评估演示完成！")


if __name__ == '__main__':
    main()
