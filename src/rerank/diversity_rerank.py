"""
MMR (Maximal Marginal Relevance) 多样性重排算法
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd


class MMRReranker:
    """
    MMR重排算法
    在保证相关性的同时，增加推荐结果的多样性
    """
    
    def __init__(self, lambda_param: float = 0.5):
        """
        Args:
            lambda_param: 相关性和多样性的平衡参数
                         =1.0: 完全相关性
                         =0.0: 完全多样性
                         =0.5: 平衡
        """
        self.lambda_param = lambda_param
    
    def rerank(self,
               items: List[int],
               scores: List[float],
               item_features,
               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        MMR重排
        Args:
            items: 候选物品ID列表
            scores: 候选物品的相关性得分
            item_features: 物品特征信息
            top_k: 返回top-k个物品
        Returns:
            重排后的推荐列表
        """
        if len(items) == 0:
            return []
        
        # 获取物品embeddings
        item_feature_dict = {}
        if isinstance(item_features, pd.DataFrame):
            for _, row in item_features.iterrows():
                item_feature_dict[row['item_id']] = row['embedding']
        else:
            for item in item_features:
                if 'embedding' in item:
                    item_feature_dict[item['item_id']] = item['embedding']
        
        # 初始化
        selected_items = []
        selected_scores = []
        remaining_items = list(zip(items, scores))
        
        # 选择第一个物品（相关性最高）
        remaining_items.sort(key=lambda x: x[1], reverse=True)
        first_item, first_score = remaining_items.pop(0)
        selected_items.append(first_item)
        selected_scores.append(first_score)
        
        # 迭代选择剩余物品
        while len(selected_items) < top_k and remaining_items:
            max_mmr_score = -float('inf')
            max_mmr_idx = -1
            
            for idx, (item_id, relevance_score) in enumerate(remaining_items):
                # 计算与已选物品的最大相似度
                max_similarity = 0
                item_emb = item_feature_dict.get(item_id)
                
                if item_emb is not None:
                    for selected_item in selected_items:
                        selected_emb = item_feature_dict.get(selected_item)
                        if selected_emb is not None:
                            similarity = self._cosine_similarity(item_emb, selected_emb)
                            max_similarity = max(max_similarity, similarity)
                
                # 计算MMR得分
                mmr_score = (self.lambda_param * relevance_score - 
                            (1 - self.lambda_param) * max_similarity)
                
                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_idx = idx
            
            # 添加得分最高的物品
            if max_mmr_idx >= 0:
                selected_item, selected_score = remaining_items.pop(max_mmr_idx)
                selected_items.append(selected_item)
                selected_scores.append(selected_score)
        
        return list(zip(selected_items, selected_scores))
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)


class DiversityReranker:
    """
    综合多样性重排器
    考虑类目多样性、作者多样性、时效性等
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 重排配置
        """
        self.config = config
        self.diversity_config = config.get('diversity', {})
        self.freshness_config = config.get('freshness', {})
        self.business_rules = config.get('business_rules', {})
    
    def rerank(self,
               items: List[int],
               scores: List[float],
               item_features,
               top_k: int = 10) -> List[Tuple[int, float]]:
        """
        多样性重排
        """
        if len(items) == 0:
            return []
        
        # 创建item字典
        item_dict = {}
        if isinstance(item_features, pd.DataFrame):
            for _, row in item_features.iterrows():
                item_dict[row['item_id']] = row.to_dict()
        else:
            for item in item_features:
                item_dict[item['item_id']] = item
        
        # 初始化结果
        selected_items = []
        selected_scores = []
        remaining_items = list(zip(items, scores))
        
        # 记录已选择的类目和作者
        selected_categories = []
        selected_authors = []
        
        import datetime
        current_time = datetime.datetime.now()
        
        # 预计算时效性和新内容加权得分，避免在循环中重复计算
        item_static_multiplier = {}
        for item_id, _ in remaining_items:
            item_info = item_dict.get(item_id)
            if item_info is None:
                item_static_multiplier[item_id] = 1.0
                continue
                
            multiplier = 1.0
            if self.freshness_config.get('enabled', True):
                multiplier *= self._calculate_freshness(item_info, current_time)
                
            new_item_config = self.business_rules.get('new_item_boost', {})
            if new_item_config.get('enabled', True):
                if self._is_new_item(item_info, new_item_config.get('new_threshold_hours', 24), current_time):
                    multiplier *= new_item_config.get('boost_factor', 1.2)
                    
            item_static_multiplier[item_id] = multiplier
        
        # 迭代选择
        while len(selected_items) < top_k and remaining_items:
            best_item = None
            best_score = -float('inf')
            best_idx = -1
            
            for idx, (item_id, base_score) in enumerate(remaining_items):
                item_info = item_dict.get(item_id)
                if item_info is None:
                    continue
                
                # 计算综合得分
                final_score = base_score
                
                # 1. 多样性惩罚
                if self.diversity_config.get('enabled', True):
                    category = item_info.get('category')
                    author = item_info.get('author_id')
                    
                    # 类目多样性
                    category_count = selected_categories.count(category)
                    category_penalty = category_count * self.diversity_config.get('category_diversity_weight', 0.3)
                    
                    # 作者多样性
                    author_count = selected_authors.count(author)
                    author_penalty = author_count * self.diversity_config.get('author_diversity_weight', 0.3)
                    
                    final_score -= (category_penalty + author_penalty)
                
                # 应用预计算的乘数
                final_score *= item_static_multiplier.get(item_id, 1.0)
                
                # 更新最佳物品
                if final_score > best_score:
                    best_score = final_score
                    best_item = (item_id, base_score)
                    best_idx = idx
            
            # 添加最佳物品
            if best_idx >= 0:
                item_id, original_score = remaining_items.pop(best_idx)
                selected_items.append(item_id)
                selected_scores.append(best_score)
                
                # 更新已选类目和作者
                item_info = item_dict[item_id]
                selected_categories.append(item_info.get('category'))
                selected_authors.append(item_info.get('author_id'))
        
        # 应用业务规则
        final_results = self._apply_business_rules(
            list(zip(selected_items, selected_scores)),
            item_dict
        )
        
        return final_results
    
    def _calculate_freshness(self, item_info: Dict, now=None) -> float:
        """计算时效性得分"""
        import datetime
        
        publish_time = item_info.get('publish_time')
        if publish_time is None:
            return 1.0
        
        # 计算发布时长（小时）
        if now is None:
            now = datetime.datetime.now()
        if isinstance(publish_time, str):
            publish_time = datetime.datetime.fromisoformat(publish_time)
        
        hours_since_publish = (now - publish_time).total_seconds() / 3600
        
        # 指数衰减
        decay_hours = self.freshness_config.get('decay_hours', 72)
        decay_rate = self.freshness_config.get('decay_rate', 0.8)
        
        freshness = decay_rate ** (hours_since_publish / decay_hours)
        
        return max(0.1, min(1.0, freshness))  # 限制在[0.1, 1.0]范围
    
    def _is_new_item(self, item_info: Dict, threshold_hours: int, now=None) -> bool:
        """判断是否为新内容"""
        import datetime
        
        publish_time = item_info.get('publish_time')
        if publish_time is None:
            return False
        
        if now is None:
            now = datetime.datetime.now()
        if isinstance(publish_time, str):
            publish_time = datetime.datetime.fromisoformat(publish_time)
        
        hours_since_publish = (now - publish_time).total_seconds() / 3600
        
        return hours_since_publish <= threshold_hours
    
    def _apply_business_rules(self,
                             recommendations: List[Tuple[int, float]],
                             item_dict: Dict) -> List[Tuple[int, float]]:
        """
        应用业务规则
        1. 同一作者去重
        2. 同一类目限制
        3. 内容安全过滤
        """
        author_dedup_config = self.business_rules.get('author_dedup', {})
        category_control_config = self.business_rules.get('category_control', {})
        
        if not author_dedup_config.get('enabled', True):
            return recommendations
        
        filtered_results = []
        author_counts = {}
        category_counts = {}
        
        max_same_author = author_dedup_config.get('max_same_author', 2)
        max_same_category = category_control_config.get('max_same_category', 3)
        
        for item_id, score in recommendations:
            item_info = item_dict.get(item_id)
            if item_info is None:
                continue
            
            author_id = item_info.get('author_id')
            category = item_info.get('category')
            
            # 检查作者数量
            if author_counts.get(author_id, 0) >= max_same_author:
                continue
            
            # 检查类目数量
            if category_counts.get(category, 0) >= max_same_category:
                continue
            
            # 通过检查，添加到结果
            filtered_results.append((item_id, score))
            author_counts[author_id] = author_counts.get(author_id, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
        
        return filtered_results
