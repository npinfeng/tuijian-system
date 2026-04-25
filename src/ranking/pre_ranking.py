"""
粗排模块 (Pre-Ranking)

从召回的 500 个候选集快速筛选到 200 个，送入精排。
使用轻量级逻辑回归模型，特点：
- 计算开销极低（线性模型）
- 使用统计特征（CTR、完播率等），无需深度模型推理
- 支持在线增量更新（特征实时化）

在工业级推荐系统中，粗排的目标是用极低计算成本淘汰不相关候选，
降低精排（DIN 等深度模型）的计算压力。
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


class PreRankingFeatureExtractor:
    """粗排特征提取器（使用轻量统计特征）"""

    def __init__(self):
        # 物品统计特征缓存
        self.item_stats: Dict[int, Dict] = {}
        # 用户统计特征缓存
        self.user_stats: Dict[int, Dict] = {}

    def fit(self, interactions: pd.DataFrame,
            items: Optional[pd.DataFrame] = None,
            window_days: int = 7):
        """
        从历史交互中计算统计特征
        Args:
            interactions: 交互记录
            items: 物品元信息
            window_days: 统计窗口（天）
        """
        print("计算粗排统计特征...")

        # 物品统计：CTR、完播率、点赞率、分享率
        item_agg = {}
        for col in ['is_click', 'is_finish', 'is_like', 'is_share']:
            if col in interactions.columns:
                item_agg[col] = 'mean'
        item_agg['user_id'] = 'count'

        item_group = interactions.groupby('item_id').agg(item_agg)
        item_group = item_group.rename(columns={'user_id': 'impression_cnt'})

        for item_id, row in item_group.iterrows():
            self.item_stats[int(item_id)] = {
                'ctr': float(row.get('is_click', 0.15)),
                'play_rate': float(row.get('is_finish', 0.3)),
                'like_rate': float(row.get('is_like', 0.05)),
                'share_rate': float(row.get('is_share', 0.02)),
                'impression_cnt': int(row['impression_cnt']),
            }

        # 用户统计：活跃度、平均 CTR
        user_agg = {'item_id': 'count'}
        if 'is_click' in interactions.columns:
            user_agg['is_click'] = 'mean'

        user_group = interactions.groupby('user_id').agg(user_agg)
        user_group = user_group.rename(columns={'item_id': 'action_cnt'})

        for user_id, row in user_group.iterrows():
            self.user_stats[int(user_id)] = {
                'action_cnt': int(row['action_cnt']),
                'avg_ctr': float(row.get('is_click', 0.15)),
            }

        print(f"粗排特征计算完成: {len(self.item_stats)} 个物品, {len(self.user_stats)} 个用户")

    def extract(self, user_id: int, item_ids: List[int]) -> np.ndarray:
        """
        提取 (用户, 物品) 对的特征
        Args:
            user_id: 用户 ID
            item_ids: 候选物品 ID 列表
        Returns:
            特征矩阵, shape (len(item_ids), n_features)
        """
        user_feat = self.user_stats.get(user_id, {
            'action_cnt': 0, 'avg_ctr': 0.15
        })

        features = []
        for item_id in item_ids:
            item_feat = self.item_stats.get(item_id, {
                'ctr': 0.1, 'play_rate': 0.25,
                'like_rate': 0.04, 'share_rate': 0.015,
                'impression_cnt': 0
            })

            feat_vec = [
                # 物品统计特征
                item_feat['ctr'],
                item_feat['play_rate'],
                item_feat['like_rate'],
                item_feat['share_rate'],
                np.log1p(item_feat['impression_cnt']),
                # 用户统计特征
                np.log1p(user_feat['action_cnt']),
                user_feat['avg_ctr'],
                # 交叉特征
                item_feat['ctr'] * user_feat['avg_ctr'],
            ]
            features.append(feat_vec)

        return np.array(features, dtype=np.float32)


class PreRanking:
    """
    粗排模型（逻辑回归）
    用于快速筛选召回候选集：500 -> 200
    """

    def __init__(self, top_k: int = 200):
        """
        Args:
            top_k: 粗排后保留的候选数量
        """
        self.top_k = top_k
        self.lr_model = LogisticRegression(
            C=1.0,
            max_iter=200,
            solver='lbfgs',
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_extractor = PreRankingFeatureExtractor()
        self.is_fitted = False

    def fit(self, interactions: pd.DataFrame,
            items: Optional[pd.DataFrame] = None):
        """
        训练粗排模型
        Args:
            interactions: 需含 user_id, item_id, is_click
            items: 物品信息（可选）
        """
        print("训练粗排模型...")

        if 'is_click' not in interactions.columns:
            print("警告: interactions 缺少 is_click 列，跳过粗排训练")
            return

        # 提取统计特征
        self.feature_extractor.fit(interactions, items)

        # 构建训练样本
        print("构建粗排训练样本...")
        X_list, y_list = [], []

        # 采样部分数据（避免内存溢出）
        sample_df = interactions.sample(
            n=min(len(interactions), 50000), random_state=42)

        for _, row in sample_df.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            label = int(row['is_click'])

            feat = self.feature_extractor.extract(uid, [iid])[0]
            X_list.append(feat)
            y_list.append(label)

        X = np.array(X_list)
        y = np.array(y_list)

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练
        self.lr_model.fit(X_scaled, y)
        self.is_fitted = True

        # 评估
        y_pred = self.lr_model.predict_proba(X_scaled)[:, 1]
        auc = roc_auc_score(y, y_pred)
        print(f"粗排模型训练完成! 训练集 AUC: {auc:.4f}")

    def rank(self, user_id: int,
             candidates: List[Tuple[int, float]],
             top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        粗排打分并筛选
        Args:
            user_id: 用户 ID
            candidates: [(item_id, recall_score), ...]
            top_k: 保留数量，默认使用 self.top_k
        Returns:
            [(item_id, pre_ranking_score), ...]
        """
        if not candidates:
            return []

        k = top_k or self.top_k
        item_ids = [item_id for item_id, _ in candidates]
        recall_scores = {item_id: score for item_id, score in candidates}

        if self.is_fitted:
            # 模型打分
            X = self.feature_extractor.extract(user_id, item_ids)
            X_scaled = self.scaler.transform(X)
            model_scores = self.lr_model.predict_proba(X_scaled)[:, 1]

            # 融合召回分和模型分（0.3 * recall + 0.7 * model）
            scored_items = []
            for i, item_id in enumerate(item_ids):
                recall_score = recall_scores.get(item_id, 0.0)
                # 归一化召回分到 [0, 1]
                normalized_recall = min(1.0, max(0.0, recall_score))
                final_score = 0.3 * normalized_recall + 0.7 * float(model_scores[i])
                scored_items.append((item_id, final_score))
        else:
            # 未训练时直接使用召回分
            scored_items = candidates

        # 排序取 top-k
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:k]

    def save(self, filepath: str):
        data = {
            'top_k': self.top_k,
            'lr_model': self.lr_model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'is_fitted': self.is_fitted,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"PreRanking 模型已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'PreRanking':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(top_k=data['top_k'])
        model.lr_model = data['lr_model']
        model.scaler = data['scaler']
        model.feature_extractor = data['feature_extractor']
        model.is_fitted = data['is_fitted']
        print(f"PreRanking 模型已从 {filepath} 加载")
        return model
