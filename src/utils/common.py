"""
工具函数模块
包含数据处理、特征工程等通用函数
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import hashlib
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def hash_bucket(key: str, bucket_size: int) -> int:
    """哈希分桶"""
    hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return hash_value % bucket_size


def split_by_user(df: pd.DataFrame, 
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按用户划分训练、验证、测试集
    保证同一用户的数据不会跨集合
    """
    users = df['user_id'].unique()
    np.random.shuffle(users)
    
    n_users = len(users)
    train_end = int(n_users * train_ratio)
    val_end = train_end + int(n_users * val_ratio)
    
    train_users = users[:train_end]
    val_users = users[train_end:val_end]
    test_users = users[val_end:]
    
    train_df = df[df['user_id'].isin(train_users)]
    val_df = df[df['user_id'].isin(val_users)]
    test_df = df[df['user_id'].isin(test_users)]
    
    return train_df, val_df, test_df


def split_by_time(df: pd.DataFrame,
                  time_col: str = 'timestamp',
                  train_days: int = 7,
                  val_days: int = 1,
                  test_days: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间划分训练、验证、测试集
    模拟真实在线场景
    """
    df = df.sort_values(time_col)
    
    total_days = train_days + val_days + test_days
    min_time = df[time_col].min()
    max_time = df[time_col].max()
    time_span = (max_time - min_time).total_seconds()
    
    train_end = min_time + pd.Timedelta(seconds=time_span * train_days / total_days)
    val_end = train_end + pd.Timedelta(seconds=time_span * val_days / total_days)
    
    train_df = df[df[time_col] < train_end]
    val_df = df[(df[time_col] >= train_end) & (df[time_col] < val_end)]
    test_df = df[df[time_col] >= val_end]
    
    return train_df, val_df, test_df


def negative_sampling(positive_samples: pd.DataFrame,
                     item_pool: List[int],
                     neg_ratio: int = 4) -> pd.DataFrame:
    """
    负采样
    为每个正样本生成neg_ratio个负样本
    """
    negative_samples = []
    
    for _, row in positive_samples.iterrows():
        user_id = row['user_id']
        # 随机采样负样本
        neg_items = np.random.choice(item_pool, size=neg_ratio, replace=False)
        
        for item_id in neg_items:
            neg_sample = row.copy()
            neg_sample['item_id'] = item_id
            neg_sample['label'] = 0
            negative_samples.append(neg_sample)
    
    negative_df = pd.DataFrame(negative_samples)
    return pd.concat([positive_samples, negative_df], ignore_index=True)


def gauc(user_ids: np.ndarray, 
         labels: np.ndarray, 
         preds: np.ndarray) -> float:
    """
    计算Group AUC
    按用户分组计算AUC，然后加权平均
    """
    from sklearn.metrics import roc_auc_score
    
    df = pd.DataFrame({
        'user_id': user_ids,
        'label': labels,
        'pred': preds
    })
    
    gauc_sum = 0
    total_weight = 0
    
    for user_id, group in df.groupby('user_id'):
        if len(group['label'].unique()) < 2:
            # 该用户只有一个类别，跳过
            continue
        
        try:
            auc = roc_auc_score(group['label'], group['pred'])
            weight = len(group)
            gauc_sum += auc * weight
            total_weight += weight
        except:
            continue
    
    return gauc_sum / total_weight if total_weight > 0 else 0


def ndcg_at_k(y_true: List[float], 
              y_pred: List[float], 
              k: int = 10) -> float:
    """
    计算NDCG@K
    """
    # 按预测分数排序
    order = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[order[:k]]
    
    # DCG
    dcg = np.sum(y_true_sorted / np.log2(np.arange(2, k + 2)))
    
    # IDCG
    y_true_ideal = np.sort(y_true)[::-1][:k]
    idcg = np.sum(y_true_ideal / np.log2(np.arange(2, k + 2)))
    
    return dcg / idcg if idcg > 0 else 0


def diversity_score(recommendations: List[List[int]], 
                    item_features: pd.DataFrame) -> float:
    """
    计算推荐列表的多样性
    基于物品类别的多样性
    """
    diversity_scores = []
    
    for rec_list in recommendations:
        categories = item_features[item_features['item_id'].isin(rec_list)]['category'].values
        unique_ratio = len(set(categories)) / len(categories) if len(categories) > 0 else 0
        diversity_scores.append(unique_ratio)
    
    return np.mean(diversity_scores)


def coverage_score(recommendations: List[List[int]], 
                  total_items: int) -> float:
    """
    计算推荐覆盖率
    """
    recommended_items = set()
    for rec_list in recommendations:
        recommended_items.update(rec_list)
    
    return len(recommended_items) / total_items


def build_user_behavior_sequence(df: pd.DataFrame,
                                 user_col: str = 'user_id',
                                 item_col: str = 'item_id',
                                 time_col: str = 'timestamp',
                                 max_seq_len: int = 50) -> Dict[int, List[int]]:
    """
    构建用户行为序列
    """
    df = df.sort_values([user_col, time_col])
    
    user_sequences = {}
    for user_id, group in df.groupby(user_col):
        seq = group[item_col].tolist()[-max_seq_len:]  # 只保留最近的行为
        user_sequences[user_id] = seq
    
    return user_sequences


def exponential_decay(publish_time: pd.Series,
                     current_time: pd.Timestamp,
                     decay_hours: int = 72,
                     decay_rate: float = 0.8) -> pd.Series:
    """
    指数衰减函数
    用于计算内容时效性得分
    """
    time_diff_hours = (current_time - publish_time).dt.total_seconds() / 3600
    decay_factor = decay_rate ** (time_diff_hours / decay_hours)
    return decay_factor.clip(0, 1)


class ABTestSplitter:
    """A/B测试分流器"""
    
    def __init__(self, experiment_name: str, traffic_split: Dict[str, float]):
        self.experiment_name = experiment_name
        self.traffic_split = traffic_split
        
        # 计算分流边界
        self.boundaries = {}
        cumsum = 0
        for group, ratio in traffic_split.items():
            self.boundaries[group] = (cumsum, cumsum + ratio)
            cumsum += ratio
    
    def get_group(self, user_id: int) -> str:
        """获取用户所属实验组"""
        hash_value = hash_bucket(f"{self.experiment_name}_{user_id}", 10000) / 10000
        
        for group, (lower, upper) in self.boundaries.items():
            if lower <= hash_value < upper:
                return group
        
        return list(self.traffic_split.keys())[0]  # 默认返回第一组


def print_metrics(metrics: Dict[str, float]):
    """打印评估指标"""
    print("\n" + "="*50)
    print("评估指标:")
    print("="*50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("="*50 + "\n")
