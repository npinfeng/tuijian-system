"""
特征工程模块
包含实时特征和离线特征的处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import json
import time


class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config: Dict):
        self.config = config
    
    def extract_user_features(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取用户特征
        """
        features = user_data.copy()
        
        # 1. 基础特征
        # 注册天数
        if 'register_date' in features.columns:
            features['register_days'] = (
                datetime.now() - pd.to_datetime(features['register_date'])
            ).dt.days
        
        # 年龄分桶
        if 'age' in features.columns:
            features['age_group'] = pd.cut(
                features['age'],
                bins=[0, 18, 25, 35, 45, 100],
                labels=['<18', '18-25', '25-35', '35-45', '45+']
            )
        
        # 2. 设备特征
        if 'device_type' in features.columns:
            features['is_ios'] = (features['device_type'] == 'iOS').astype(int)
            features['is_android'] = (features['device_type'] == 'Android').astype(int)
        
        return features
    
    def extract_item_features(self, item_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取物品特征
        """
        features = item_data.copy()
        
        # 1. 时间特征
        if 'publish_time' in features.columns:
            publish_time = pd.to_datetime(features['publish_time'])
            
            # 发布时长（小时）
            features['hours_since_publish'] = (
                datetime.now() - publish_time
            ).dt.total_seconds() / 3600
            
            # 发布时段
            features['publish_hour'] = publish_time.dt.hour
            features['publish_weekday'] = publish_time.dt.weekday
            features['is_weekend'] = (publish_time.dt.weekday >= 5).astype(int)
        
        # 2. 内容特征
        if 'duration' in features.columns:
            # 时长分桶
            features['duration_bucket'] = pd.cut(
                features['duration'],
                bins=[0, 30, 60, 120, 300, 1000],
                labels=['<30s', '30s-1m', '1m-2m', '2m-5m', '5m+']
            )
        
        # 3. 标签特征
        if 'tag_list' in features.columns:
            # 标签数量
            features['tag_count'] = features['tag_list'].apply(
                lambda x: len(x.split(',')) if isinstance(x, str) else 0
            )
        
        return features
    
    def extract_context_features(self, context_data: pd.DataFrame) -> pd.DataFrame:
        """
        提取上下文特征
        """
        features = context_data.copy()
        
        # 时间特征
        current_time = datetime.now()
        features['hour'] = current_time.hour
        features['weekday'] = current_time.weekday()
        features['is_weekend'] = int(current_time.weekday() >= 5)
        
        # 时段特征
        hour = current_time.hour
        if 6 <= hour < 12:
            features['time_period'] = 'morning'
        elif 12 <= hour < 18:
            features['time_period'] = 'afternoon'
        elif 18 <= hour < 24:
            features['time_period'] = 'evening'
        else:
            features['time_period'] = 'night'
        
        return features
    
    def extract_statistical_features(self, 
                                    interactions: pd.DataFrame,
                                    window_days: int = 7) -> Dict[str, pd.DataFrame]:
        """
        提取统计特征
        包括用户统计、物品统计、交叉统计
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_interactions = interactions[
            pd.to_datetime(interactions['timestamp']) >= cutoff_date
        ]
        
        # 1. 用户统计特征
        user_stats = recent_interactions.groupby('user_id').agg({
            'item_id': 'count',  # 点击次数
            'is_click': 'sum',   # 点击数
            'is_play': 'sum',    # 播放数
            'play_duration': 'mean',  # 平均播放时长
        }).rename(columns={
            'item_id': f'user_action_{window_days}d',
            'is_click': f'user_click_{window_days}d',
            'is_play': f'user_play_{window_days}d',
            'play_duration': f'user_avg_duration_{window_days}d'
        })
        
        # 2. 物品统计特征
        item_stats = recent_interactions.groupby('item_id').agg({
            'user_id': 'count',  # 曝光次数
            'is_click': ['sum', 'mean'],  # 点击数和点击率
            'is_play': ['sum', 'mean'],   # 播放数和播放率
            'play_duration': 'mean',
        })
        item_stats.columns = [
            f'item_impression_{window_days}d',
            f'item_click_{window_days}d',
            f'item_ctr_{window_days}d',
            f'item_play_{window_days}d',
            f'item_play_rate_{window_days}d',
            f'item_avg_duration_{window_days}d'
        ]
        
        # 3. 用户-类目交叉特征
        if 'category' in recent_interactions.columns:
            user_category_stats = recent_interactions.groupby(
                ['user_id', 'category']
            )['is_click'].sum().reset_index()
            user_category_stats.columns = ['user_id', 'category', 'user_category_click']
        else:
            user_category_stats = None
        
        # 4. 用户-作者交叉特征
        if 'author_id' in recent_interactions.columns:
            user_author_stats = recent_interactions.groupby(
                ['user_id', 'author_id']
            )['is_click'].sum().reset_index()
            user_author_stats.columns = ['user_id', 'author_id', 'user_author_click']
        else:
            user_author_stats = None
        
        return {
            'user_stats': user_stats,
            'item_stats': item_stats,
            'user_category_stats': user_category_stats,
            'user_author_stats': user_author_stats
        }
    
    def build_cross_features(self, 
                            user_features: pd.DataFrame,
                            item_features: pd.DataFrame) -> pd.DataFrame:
        """
        构建交叉特征
        """
        # 笛卡尔积
        cross_features = user_features.merge(
            item_features,
            how='cross'
        )
        
        # 手动交叉特征
        if 'age' in cross_features.columns and 'category' in cross_features.columns:
            cross_features['age_category'] = (
                cross_features['age'].astype(str) + '_' + 
                cross_features['category'].astype(str)
            )
        
        if 'gender' in cross_features.columns and 'category' in cross_features.columns:
            cross_features['gender_category'] = (
                cross_features['gender'].astype(str) + '_' + 
                cross_features['category'].astype(str)
            )
        
        return cross_features


class OfflineFeatureStore:
    """
    离线特征存储
    基于本地内存字典 (用于面试项目和本地测试的离线特征检索)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.store = {}
        self.ttl = config.get('realtime', {}).get('ttl', 86400)  # 24小时
    
    def _is_expired(self, key: str) -> bool:
        if key in self.store:
            data = self.store[key]
            if time.time() > data['expire_at']:
                self.store.pop(key, None)
                return True
            return False
        return True

    def get_user_features(self, user_id: int) -> Dict:
        """获取用户离线推断特征"""
        key = f"user_features:{user_id}"
        if not self._is_expired(key):
            return json.loads(self.store[key]['value'])
        return {}
    
    def set_user_features(self, user_id: int, features: Dict):
        """设置用户离线特征"""
        key = f"user_features:{user_id}"
        self.store[key] = {
            'value': json.dumps(features),
            'expire_at': time.time() + self.ttl
        }
    
    def get_item_features(self, item_id: int) -> Dict:
        """获取物品离线推断特征"""
        key = f"item_features:{item_id}"
        if not self._is_expired(key):
            return json.loads(self.store[key]['value'])
        return {}
    
    def set_item_features(self, item_id: int, features: Dict):
        """设置物品离线特征"""
        key = f"item_features:{item_id}"
        self.store[key] = {
            'value': json.dumps(features),
            'expire_at': time.time() + self.ttl
        }
    
    def increment_counter(self, key: str, amount: int = 1):
        """增加计数器"""
        if self._is_expired(key):
            self.store[key] = {'value': str(amount), 'expire_at': time.time() + self.ttl}
        else:
            current_value = int(self.store[key]['value'])
            self.store[key]['value'] = str(current_value + amount)
            self.store[key]['expire_at'] = time.time() + self.ttl
    
    def get_counter(self, key: str) -> int:
        """获取计数器值"""
        if not self._is_expired(key):
            return int(self.store[key]['value'])
        return 0
    
    def update_item_ctr(self, item_id: int, is_click: int):
        """
        更新物品点击率
        使用滑动窗口统计
        """
        impression_key = f"item_impression:{item_id}"
        click_key = f"item_click:{item_id}"
        
        self.increment_counter(impression_key)
        if is_click:
            self.increment_counter(click_key)
    
    def get_item_ctr(self, item_id: int) -> float:
        """获取物品点击率"""
        impression_key = f"item_impression:{item_id}"
        click_key = f"item_click:{item_id}"
        
        impressions = self.get_counter(impression_key)
        clicks = self.get_counter(click_key)
        
        if impressions == 0:
            return 0.0
        
        return clicks / impressions
