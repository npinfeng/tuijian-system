"""
热门召回 + 新内容冷启动召回

包含两个独立的召回器:
1. HotRecall: 基于全局热度（点击量、完播率、时效性衰减）的热门内容召回
2. NewItemRecall: 基于发布时间的新内容冷启动扶持召回（近 N 小时新发布物品）

设计思路:
- 热门召回作为兜底保底策略，确保推荐结果不为空
- 新内容召回解决冷启动问题，保证新视频得到初始曝光流量
- 两者都支持类目过滤，避免单一类目占据推荐坑位
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class HotRecall:
    """
    热门召回
    综合点击率、完播率、点赞率和时效性衰减，计算物品热度分。
    """

    def __init__(self, decay_hours: float = 72.0, decay_rate: float = 0.8,
                 ctr_weight: float = 0.4, play_rate_weight: float = 0.3,
                 like_rate_weight: float = 0.2, share_rate_weight: float = 0.1):
        """
        Args:
            decay_hours: 时效衰减半衰期（小时）
            decay_rate: 衰减系数（每 decay_hours 衰减到该比例）
            ctr_weight: 点击率权重
            play_rate_weight: 完播率权重
            like_rate_weight: 点赞率权重
            share_rate_weight: 分享率权重
        """
        self.decay_hours = decay_hours
        self.decay_rate = decay_rate
        self.ctr_weight = ctr_weight
        self.play_rate_weight = play_rate_weight
        self.like_rate_weight = like_rate_weight
        self.share_rate_weight = share_rate_weight

        # 热门物品列表: [(item_id, hot_score), ...]，训练后填充
        self.hot_items: List[Tuple[int, float]] = []
        # 物品元信息 {item_id: {category, publish_time, ...}}
        self.item_info: Dict[int, Dict] = {}

    def fit(self, interactions: pd.DataFrame,
            items: Optional[pd.DataFrame] = None,
            window_days: int = 7):
        """
        训练热门模型（离线计算热度分）
        Args:
            interactions: 交互记录，需含 item_id, is_click, is_finish, is_like, is_share, timestamp
            items: 物品信息表，需含 item_id, category, publish_time
            window_days: 统计窗口（天）
        """
        print("计算热门物品热度分...")

        current_time = datetime.now()
        cutoff = current_time - timedelta(days=window_days)

        # 过滤时间窗口内的交互
        df = interactions.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df[df['timestamp'] >= cutoff]

        # 统计各指标
        agg_dict: Dict = {'user_id': 'count'}
        if 'is_click' in df.columns:
            agg_dict['is_click'] = 'sum'
        if 'is_finish' in df.columns:
            agg_dict['is_finish'] = 'sum'
        if 'is_like' in df.columns:
            agg_dict['is_like'] = 'sum'
        if 'is_share' in df.columns:
            agg_dict['is_share'] = 'sum'

        item_stats = df.groupby('item_id').agg(agg_dict).rename(
            columns={'user_id': 'impression'})

        # 计算各比率，加拉普拉斯平滑
        smoothing = 10
        item_stats['ctr'] = (item_stats.get('is_click', 0) + 1) / (item_stats['impression'] + smoothing)
        item_stats['play_rate'] = (item_stats.get('is_finish', 0) + 1) / (item_stats['impression'] + smoothing)
        item_stats['like_rate'] = (item_stats.get('is_like', 0) + 1) / (item_stats['impression'] + smoothing)
        item_stats['share_rate'] = (item_stats.get('is_share', 0) + 1) / (item_stats['impression'] + smoothing)

        # 归一化（min-max）
        for col in ['ctr', 'play_rate', 'like_rate', 'share_rate']:
            col_min = item_stats[col].min()
            col_max = item_stats[col].max()
            if col_max > col_min:
                item_stats[col] = (item_stats[col] - col_min) / (col_max - col_min)

        # 综合热度分（不含时效）
        item_stats['base_hot_score'] = (
            self.ctr_weight * item_stats['ctr'] +
            self.play_rate_weight * item_stats['play_rate'] +
            self.like_rate_weight * item_stats['like_rate'] +
            self.share_rate_weight * item_stats['share_rate']
        )

        # 构建物品元信息
        if items is not None:
            for _, row in items.iterrows():
                self.item_info[int(row['item_id'])] = row.to_dict()

        # 计算时效衰减并得到最终热度
        hot_list = []
        for item_id, row in item_stats.iterrows():
            base_score = float(row['base_hot_score'])

            # 时效衰减
            freshness = 1.0
            item_meta = self.item_info.get(int(item_id), {})
            publish_time = item_meta.get('publish_time')
            if publish_time is not None:
                if isinstance(publish_time, str):
                    publish_time = pd.to_datetime(publish_time)
                hours_elapsed = (current_time - publish_time).total_seconds() / 3600
                freshness = self.decay_rate ** (hours_elapsed / self.decay_hours)
                freshness = float(np.clip(freshness, 0.1, 1.0))

            hot_score = base_score * freshness
            hot_list.append((int(item_id), hot_score))

        # 排序
        self.hot_items = sorted(hot_list, key=lambda x: x[1], reverse=True)
        print(f"热门物品计算完成: {len(self.hot_items)} 个物品")

    def recommend(self, user_id: Optional[int] = None,
                  n: int = 50,
                  exclude_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        热门召回（不依赖用户特征，对冷启动用户也适用）
        Args:
            user_id: 用户 ID（此处不使用，保持接口统一）
            n: 返回数量
            exclude_items: 需要过滤的物品集合（如用户已看过的）
        Returns:
            [(item_id, score), ...]
        """
        if not self.hot_items:
            return [(i, 1.0 / (i + 1)) for i in range(1, n + 1)]

        exclude = exclude_items or set()
        results = [
            (item_id, score)
            for item_id, score in self.hot_items
            if item_id not in exclude
        ]
        return results[:n]

    def save(self, filepath: str):
        data = {
            'decay_hours': self.decay_hours,
            'decay_rate': self.decay_rate,
            'ctr_weight': self.ctr_weight,
            'play_rate_weight': self.play_rate_weight,
            'like_rate_weight': self.like_rate_weight,
            'share_rate_weight': self.share_rate_weight,
            'hot_items': self.hot_items,
            'item_info': self.item_info,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"HotRecall 已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'HotRecall':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(
            decay_hours=data['decay_hours'],
            decay_rate=data['decay_rate'],
            ctr_weight=data['ctr_weight'],
            play_rate_weight=data['play_rate_weight'],
            like_rate_weight=data['like_rate_weight'],
            share_rate_weight=data['share_rate_weight'],
        )
        model.hot_items = data['hot_items']
        model.item_info = data['item_info']
        print(f"HotRecall 已从 {filepath} 加载")
        return model


class NewItemRecall:
    """
    新内容冷启动召回
    针对近期发布的新物品，给予初始曝光流量，解决新内容冷启动问题。

    策略:
    1. 过滤出 N 小时内发布的新物品
    2. 按类目均匀采样（避免某一类目占据所有坑位）
    3. 对曝光量极少的新物品额外加权（流量扶持）
    """

    def __init__(self, new_threshold_hours: int = 24, category_balance: bool = True):
        """
        Args:
            new_threshold_hours: 新物品时间阈值（小时）
            category_balance: 是否按类目均衡采样
        """
        self.new_threshold_hours = new_threshold_hours
        self.category_balance = category_balance

        # 新物品列表（定期更新）
        self.new_items: List[Dict] = []
        # 物品曝光计数（用于流量扶持）
        self.item_impression_count: Dict[int, int] = defaultdict(int)

    def fit(self, items: pd.DataFrame,
            interactions: Optional[pd.DataFrame] = None):
        """
        扫描新发布物品
        Args:
            items: 物品信息表，需含 item_id, publish_time, category
            interactions: 交互记录（用于统计各物品曝光次数）
        """
        print("扫描新发布物品...")
        current_time = datetime.now()
        threshold = current_time - timedelta(hours=self.new_threshold_hours)

        # 过滤新物品
        self.new_items = []
        for _, row in items.iterrows():
            publish_time = row.get('publish_time')
            if publish_time is None:
                continue
            if isinstance(publish_time, str):
                publish_time = pd.to_datetime(publish_time)
            # 确保转换为无时区的 datetime 对象
            if hasattr(publish_time, 'to_pydatetime'):
                publish_time = publish_time.to_pydatetime()
            # 去掉时区信息（如有）
            if hasattr(publish_time, 'tzinfo') and publish_time.tzinfo is not None:
                publish_time = publish_time.replace(tzinfo=None)

            if publish_time >= threshold:
                self.new_items.append({
                    'item_id': int(row['item_id']),
                    'publish_time': publish_time,
                    'category': row.get('category', 'unknown'),
                    'author_id': row.get('author_id', None),
                })

        # 统计曝光次数（用于流量扶持排序）
        if interactions is not None and 'item_id' in interactions.columns:
            counts = interactions['item_id'].value_counts().to_dict()
            self.item_impression_count = defaultdict(int, {int(k): v for k, v in counts.items()})

        print(f"发现 {len(self.new_items)} 个新物品（近 {self.new_threshold_hours} 小时内发布）")

    def recommend(self, user_id: Optional[int] = None,
                  n: int = 40,
                  exclude_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        新内容召回
        Returns:
            [(item_id, score), ...] 曝光量越低分数越高（流量扶持）
        """
        if not self.new_items:
            return []

        exclude = exclude_items or set()

        # 过滤
        candidates = [
            item for item in self.new_items
            if item['item_id'] not in exclude
        ]

        if not candidates:
            return []

        if self.category_balance:
            # 按类目分组，均匀采样
            category_groups: Dict[str, List[Dict]] = defaultdict(list)
            for item in candidates:
                category_groups[str(item['category'])].append(item)

            results = []
            while len(results) < n and category_groups:
                for cat, group in list(category_groups.items()):
                    if not group:
                        del category_groups[cat]
                        continue
                    # 取曝光最少的物品
                    item = min(group, key=lambda x: self.item_impression_count.get(x['item_id'], 0))
                    group.remove(item)
                    # 曝光量越少，新内容扶持分越高
                    impression_count = self.item_impression_count.get(item['item_id'], 0)
                    score = 1.0 / (impression_count + 1)
                    results.append((item['item_id'], score))
                    if len(results) >= n:
                        break
        else:
            # 简单按曝光量排序（曝光少优先）
            sorted_candidates = sorted(
                candidates,
                key=lambda x: self.item_impression_count.get(x['item_id'], 0)
            )
            results = [
                (item['item_id'], 1.0 / (self.item_impression_count.get(item['item_id'], 0) + 1))
                for item in sorted_candidates[:n]
            ]

        return results[:n]

    def update_impression(self, item_id: int, count: int = 1):
        """更新物品曝光计数（在线更新）"""
        self.item_impression_count[item_id] += count

    def save(self, filepath: str):
        data = {
            'new_threshold_hours': self.new_threshold_hours,
            'category_balance': self.category_balance,
            'new_items': self.new_items,
            'item_impression_count': dict(self.item_impression_count),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"NewItemRecall 已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'NewItemRecall':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(
            new_threshold_hours=data['new_threshold_hours'],
            category_balance=data['category_balance'],
        )
        model.new_items = data['new_items']
        model.item_impression_count = defaultdict(int, data['item_impression_count'])
        print(f"NewItemRecall 已从 {filepath} 加载")
        return model
