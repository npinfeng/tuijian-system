"""
ItemCF (基于物品的协同过滤) 召回实现
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pickle


class ItemCF:
    """
    ItemCF协同过滤算法
    基于物品相似度进行推荐
    """
    
    def __init__(self, similarity_type: str = 'cosine', top_k_similar: int = 100):
        """
        Args:
            similarity_type: 相似度计算方式 (cosine, jaccard)
            top_k_similar: 每个物品保留的最相似物品数量
        """
        self.similarity_type = similarity_type
        self.top_k_similar = top_k_similar
        
        # 物品相似度矩阵
        self.item_similarity_matrix = None
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        # 用户历史行为
        self.user_history = defaultdict(set)
    
    def fit(self, interactions: pd.DataFrame):

        print("开始训练ItemCF模型...")
        
        # 构建用户历史
        for _, row in interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            self.user_history[user_id].add(item_id)
        
        # 构建物品索引
        unique_items = interactions['item_id'].unique()
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        n_items = len(unique_items)
        n_users = interactions['user_id'].nunique()
        
        print(f"用户数: {n_users}, 物品数: {n_items}")
        
        # 构建用户-物品交互矩阵
        user_to_idx = {user: idx for idx, user in enumerate(interactions['user_id'].unique())}
        
        rows = []
        cols = []
        data = []
        
        for _, row in interactions.iterrows():
            user_idx = user_to_idx[row['user_id']]
            item_idx = self.item_to_idx[row['item_id']]
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1)  # 可以替换为评分或权重
        
        user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        )
        
        # 计算物品相似度矩阵
        print("计算物品相似度矩阵...")
        if self.similarity_type == 'cosine':
            # 转置后计算，得到item-item相似度
            item_similarity = cosine_similarity(user_item_matrix.T, dense_output=False)
        else:  # jaccard
            item_similarity = self._jaccard_similarity(user_item_matrix)
        
        # 只保留top-k相似物品，节省内存
        print(f"保留每个物品的top-{self.top_k_similar}相似物品...")
        self.item_similarity_matrix = self._keep_top_k(item_similarity, self.top_k_similar)
        
        print("ItemCF模型训练完成！")
    
    def _jaccard_similarity(self, matrix: csr_matrix) -> csr_matrix:
        """计算Jaccard相似度"""
        # 转置: item x user
        item_user_matrix = matrix.T
        
        # 计算交集
        intersection = item_user_matrix.dot(item_user_matrix.T)
        
        # 计算并集
        item_counts = np.array(item_user_matrix.sum(axis=1)).flatten()
        union = item_counts[:, None] + item_counts[None, :] - intersection.toarray()
        
        # 避免除零
        union[union == 0] = 1
        
        similarity = intersection.toarray() / union
        return csr_matrix(similarity)
    
    def _keep_top_k(self, similarity_matrix: csr_matrix, k: int) -> Dict[int, List[Tuple[int, float]]]:
        """
        保留每个物品的top-k相似物品
        Returns:
            Dict: {item_idx: [(similar_item_idx, similarity_score), ...]}
        """
        top_k_similar = {}
        
        for item_idx in range(similarity_matrix.shape[0]):
            # 获取该物品与所有物品的相似度
            similarities = similarity_matrix[item_idx].toarray().flatten()
            
            # 排除自己
            similarities[item_idx] = -1
            
            # 找到top-k
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_scores = similarities[top_k_indices]
            
            # 只保留相似度>0的
            valid_mask = top_k_scores > 0
            top_k_indices = top_k_indices[valid_mask]
            top_k_scores = top_k_scores[valid_mask]
            
            top_k_similar[item_idx] = list(zip(top_k_indices, top_k_scores))
        
        return top_k_similar
    
    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """
        为用户推荐物品
        Args:
            user_id: 用户ID
            n: 推荐数量
        Returns:
            推荐列表: [(item_id, score), ...]
        """
        # 获取用户历史行为
        user_items = self.user_history.get(user_id, set())
        
        if not user_items:
            return []
        
        # 计算候选物品得分
        candidate_scores = defaultdict(float)
        
        for item_id in user_items:
            if item_id not in self.item_to_idx:
                continue
            
            item_idx = self.item_to_idx[item_id]
            
            # 获取相似物品
            similar_items = self.item_similarity_matrix.get(item_idx, [])
            
            for similar_item_idx, similarity in similar_items:
                similar_item_id = self.idx_to_item[similar_item_idx]
                
                # 过滤已交互物品
                if similar_item_id in user_items:
                    continue
                
                # 累加得分
                candidate_scores[similar_item_id] += similarity
        
        # 排序并返回top-n
        recommendations = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return recommendations
    
    def batch_recommend(self, 
                       user_ids: List[int], 
                       n: int = 10) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量推荐
        Args:
            user_ids: 用户ID列表
            n: 每个用户推荐数量
        Returns:
            Dict: {user_id: [(item_id, score), ...]}
        """
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend(user_id, n)
        return results
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'similarity_type': self.similarity_type,
            'top_k_similar': self.top_k_similar,
            'item_similarity_matrix': self.item_similarity_matrix,
            'item_to_idx': self.item_to_idx,
            'idx_to_item': self.idx_to_item,
            'user_history': dict(self.user_history)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"模型已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ItemCF':
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(
            similarity_type=model_data['similarity_type'],
            top_k_similar=model_data['top_k_similar']
        )
        
        model.item_similarity_matrix = model_data['item_similarity_matrix']
        model.item_to_idx = model_data['item_to_idx']
        model.idx_to_item = model_data['idx_to_item']
        model.user_history = defaultdict(set, model_data['user_history'])
        
        print(f"模型已从 {filepath} 加载")
        return model


class UserCF:
    """
    UserCF协同过滤算法
    基于用户相似度进行推荐
    """
    
    def __init__(self, similarity_type: str = 'cosine', top_k_similar: int = 50):
        """
        Args:
            similarity_type: 相似度计算方式
            top_k_similar: 每个用户保留的最相似用户数量
        """
        self.similarity_type = similarity_type
        self.top_k_similar = top_k_similar
        
        self.user_similarity_matrix = None
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.user_history = defaultdict(set)
        self.item_users = defaultdict(set)  # 物品被哪些用户交互过
    
    def fit(self, interactions: pd.DataFrame):
        """训练UserCF模型"""
        print("开始训练UserCF模型...")
        
        # 构建用户-物品映射
        for _, row in interactions.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            self.user_history[user_id].add(item_id)
            self.item_users[item_id].add(user_id)
        
        # 构建用户索引
        unique_users = interactions['user_id'].unique()
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        
        n_users = len(unique_users)
        n_items = interactions['item_id'].nunique()
        
        print(f"用户数: {n_users}, 物品数: {n_items}")
        
        # 构建用户-物品矩阵
        item_to_idx = {item: idx for idx, item in enumerate(interactions['item_id'].unique())}
        
        rows, cols, data = [], [], []
        for _, row in interactions.iterrows():
            user_idx = self.user_to_idx[row['user_id']]
            item_idx = item_to_idx[row['item_id']]
            rows.append(user_idx)
            cols.append(item_idx)
            data.append(1)
        
        user_item_matrix = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        
        # 计算用户相似度
        print("计算用户相似度矩阵...")
        user_similarity = cosine_similarity(user_item_matrix, dense_output=False)
        
        # 保留top-k
        print(f"保留每个用户的top-{self.top_k_similar}相似用户...")
        self.user_similarity_matrix = self._keep_top_k(user_similarity, self.top_k_similar)
        
        print("UserCF模型训练完成！")
    
    def _keep_top_k(self, similarity_matrix: csr_matrix, k: int) -> Dict[int, List[Tuple[int, float]]]:
        """保留top-k相似用户"""
        top_k_similar = {}
        
        for user_idx in range(similarity_matrix.shape[0]):
            similarities = similarity_matrix[user_idx].toarray().flatten()
            similarities[user_idx] = -1  # 排除自己
            
            top_k_indices = np.argsort(similarities)[-k:][::-1]
            top_k_scores = similarities[top_k_indices]
            
            valid_mask = top_k_scores > 0
            top_k_indices = top_k_indices[valid_mask]
            top_k_scores = top_k_scores[valid_mask]
            
            top_k_similar[user_idx] = list(zip(top_k_indices, top_k_scores))
        
        return top_k_similar
    
    def recommend(self, user_id: int, n: int = 10) -> List[Tuple[int, float]]:
        """为用户推荐物品"""
        if user_id not in self.user_to_idx:
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_items = self.user_history[user_id]
        
        # 找到相似用户
        similar_users = self.user_similarity_matrix.get(user_idx, [])
        
        # 计算候选物品得分
        candidate_scores = defaultdict(float)
        
        for similar_user_idx, similarity in similar_users:
            similar_user_id = self.idx_to_user[similar_user_idx]
            similar_user_items = self.user_history[similar_user_id]
            
            for item_id in similar_user_items:
                if item_id in user_items:
                    continue
                candidate_scores[item_id] += similarity
        
        # 排序返回
        recommendations = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return recommendations
