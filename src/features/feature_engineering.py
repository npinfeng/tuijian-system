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
    """
    特征工程类 (严格对齐召回与排序模型输入)
    基于 KuaiRand-1K 数据集，生成的特征格式严格匹配:
    - TwoTowerModel (双塔召回)
    - DIN (深度兴趣网络精排)
    """
    
    def __init__(self, config: Dict):
        self.config = config
    
    def process_user_features(self, user_data: pd.DataFrame) -> pd.DataFrame:
        """
        处理用户特征
        严格保留模型需要的列：user_id, active_degree (类别型), 以及 DIN 需要的数值型特征:
        is_live_streamer, is_video_author, follow_user_num, fans_user_num, register_days
        """
        keep_cols = [
            'user_id', 'active_degree', 
            'is_live_streamer', 'is_video_author', 
            'follow_user_num', 'fans_user_num', 'register_days'
        ]
        # 只保留存在的列
        keep_cols = [col for col in keep_cols if col in user_data.columns]
        
        features = user_data[keep_cols].copy()
        
        # 缺失值填充
        if 'active_degree' in features.columns:
            features['active_degree'] = features['active_degree'].fillna(0).astype(int)
        
        for col in ['is_live_streamer', 'is_video_author', 'follow_user_num', 'fans_user_num', 'register_days']:
            if col in features.columns:
                features[col] = features[col].fillna(0.0).astype(float)
                
        return features
    
    def process_item_features(self, item_data: pd.DataFrame) -> pd.DataFrame:
        """
        处理物品特征
        严格保留模型需要的列：item_id(video_id), video_type_id, tag
        """
        # 兼容 video_id 到 item_id 的转换
        features = item_data.copy()
        if 'video_id' in features.columns and 'item_id' not in features.columns:
            features = features.rename(columns={'video_id': 'item_id'})
            
        keep_cols = ['item_id', 'video_type_id', 'tag']
        keep_cols = [col for col in keep_cols if col in features.columns]
        features = features[keep_cols]
        
        # 缺失值填充
        for col in ['item_id', 'video_type_id', 'tag']:
            if col in features.columns:
                features[col] = features[col].fillna(0).astype(int)
                
        return features
        
    def build_two_tower_dataset(self, interactions: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame) -> pd.DataFrame:
        """
        构建双塔模型训练数据
        输出列：user_id, active_degree, item_id, video_type_id, tag, 以及标签等
        """
        df = interactions.copy()
        
        # 关联 User
        df = df.merge(self.process_user_features(user_features), on='user_id', how='left')
        
        # 关联 Item
        df = df.merge(self.process_item_features(item_features), on='item_id', how='left')
        
        df = df.fillna(0) # 兜底
        return df

    def build_din_dataset(self, 
                          interactions: pd.DataFrame, 
                          user_features: pd.DataFrame, 
                          max_seq_len: int = 50, 
                          label_col: str = 'is_click') -> pd.DataFrame:
        """
        构建 DIN 精排模型训练数据 (行为序列 + Context特征)
        除了基础特征，还要构造历史行为序列 hist_item_seq
        """
        df = interactions.copy()
        
        # 1. 包含上下文特征: 将 timestamp 转为 hour
        if 'hour' not in df.columns:
            if 'timestamp' in df.columns:
                df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            elif 'time_ms' in df.columns:
                df['hour'] = pd.to_datetime(df['time_ms'], unit='ms').dt.hour
            else:
                df['hour'] = 0
                
        if 'tab' not in df.columns:
            df['tab'] = 0

        # 2. 合并用户特征
        processed_users = self.process_user_features(user_features)
        df = df.merge(processed_users, on='user_id', how='left')
        
        # 3. 标签列
        df['label'] = df[label_col].fillna(0).astype(int)
        
        # 4. 严格防穿越 (Point-in-Time Join)：构建用户历史行为序列
        # 确保数据按时间排序，保障序列的时序正确性
        time_col = 'time_ms' if 'time_ms' in df.columns else 'timestamp'
        df = df.sort_values(['user_id', time_col]).reset_index(drop=True)
        
        user_ids = df['user_id'].values
        item_ids = df['item_id'].values
        is_clicks = df['is_click'].values
        
        hist_seqs = []
        user_history = {}
        
        # O(N) 遍历一遍，确保只有当前时刻之前发生的点击，才会被计入当前序列
        for uid, iid, click in zip(user_ids, item_ids, is_clicks):
            hist = user_history.get(uid, [])
            # 截取最近的 max_seq_len 作为该条样本的历史序列
            # 注意：列表切片创建的是新对象，避免引用污染
            hist_seqs.append(hist[-max_seq_len:])
            
            # 当前记录如果是点击，将其加入历史，供该用户之后的样本使用
            if click == 1:
                if uid not in user_history:
                    user_history[uid] = []
                user_history[uid].append(iid)
                
        df['hist_item_seq'] = hist_seqs
        df['seq_length'] = df['hist_item_seq'].apply(len)
        
        df = df.fillna(0) # 兜底
        return df


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
