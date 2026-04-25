"""
关注召回 (Follow Recall)

基于用户的社交关系（关注关系）进行召回：
- 召回用户关注的作者发布的最新内容
- 优先推荐关注的高互动作者的内容

在短视频推荐系统中，关注召回能精准命中用户的主动偏好，
是个性化推荐的重要补充，同时对创作者生态有直接价值。
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set


class FollowRecall:
    """
    关注召回
    召回用户关注的作者发布的最新内容
    """

    def __init__(self, max_items_per_author: int = 5,
                 time_window_days: int = 30,
                 decay_hours: float = 48.0):
        """
        Args:
            max_items_per_author: 每个关注作者最多召回的物品数
            time_window_days: 只召回该时间窗口内发布的内容
            decay_hours: 时效衰减半衰期（小时）
        """
        self.max_items_per_author = max_items_per_author
        self.time_window_days = time_window_days
        self.decay_hours = decay_hours

        # 用户关注关系: {user_id: {author_id, ...}}
        self.user_follows: Dict[int, Set[int]] = defaultdict(set)
        # 用户与每个关注作者的互动频次（用于排序）: {user_id: {author_id: count}}
        self.user_author_interact: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        # 作者最新物品: {author_id: [(item_id, publish_time, score), ...]} 按时间倒序
        self.author_items: Dict[int, List[Dict]] = defaultdict(list)

    def fit(self,
            interactions: pd.DataFrame,
            items: pd.DataFrame,
            follow_records: Optional[pd.DataFrame] = None):
        """
        构建关注召回所需的数据结构
        Args:
            interactions: 交互记录，需含 user_id, item_id, author_id（可选）
            items: 物品信息表，需含 item_id, author_id, publish_time
            follow_records: 关注记录表，需含 user_id, author_id
                           如无真实关注数据，从历史交互中推断用户偏好的作者
        """
        print("构建关注召回数据...")
        current_time = datetime.now()
        cutoff = current_time - timedelta(days=self.time_window_days)

        # 1. 构建作者-物品索引（只保留时间窗口内的内容）
        item_author_map: Dict[int, int] = {}
        for _, row in items.iterrows():
            item_id = int(row['item_id'])
            author_id = int(row.get('author_id', 0))
            item_author_map[item_id] = author_id

            publish_time = row.get('publish_time')
            if publish_time is None:
                continue
            if isinstance(publish_time, str):
                publish_time = pd.to_datetime(publish_time).to_pydatetime()
            elif hasattr(publish_time, 'to_pydatetime'):
                publish_time = publish_time.to_pydatetime()
            if hasattr(publish_time, 'tzinfo') and publish_time.tzinfo is not None:
                publish_time = publish_time.replace(tzinfo=None)

            if publish_time < cutoff:
                continue

            hours_elapsed = max(0, (current_time - publish_time).total_seconds() / 3600)
            freshness = 0.8 ** (hours_elapsed / self.decay_hours)
            freshness = float(np.clip(freshness, 0.05, 1.0))

            self.author_items[author_id].append({
                'item_id': item_id,
                'publish_time': publish_time,
                'freshness_score': freshness,
            })

        # 按时间倒序排列
        for author_id in self.author_items:
            self.author_items[author_id].sort(
                key=lambda x: x['publish_time'], reverse=True)

        # 2. 构建关注关系
        if follow_records is not None and len(follow_records) > 0:
            # 使用真实关注数据
            for _, row in follow_records.iterrows():
                uid = int(row['user_id'])
                aid = int(row['author_id'])
                self.user_follows[uid].add(aid)
            print(f"从关注记录构建: {len(self.user_follows)} 个用户有关注关系")
        else:
            # 从历史交互推断：用户交互过 >= 3 次的作者视为"隐式关注"
            print("无显式关注记录，从历史交互推断隐式关注...")
            user_author_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

            for _, row in interactions.iterrows():
                uid = int(row['user_id'])
                iid = int(row['item_id'])
                author_id = item_author_map.get(iid)
                if author_id is None:
                    # 尝试从 interactions 直接获取 author_id
                    author_id = int(row.get('author_id', 0))
                if author_id > 0:
                    user_author_counts[uid][author_id] += 1

            IMPLICIT_FOLLOW_THRESHOLD = 3
            for uid, author_counts in user_author_counts.items():
                for aid, cnt in author_counts.items():
                    if cnt >= IMPLICIT_FOLLOW_THRESHOLD:
                        self.user_follows[uid].add(aid)
                        self.user_author_interact[uid][aid] = cnt

            print(f"推断隐式关注: {len(self.user_follows)} 个用户")

        # 3. 统计用户与关注作者的互动频次（用于排序）
        for _, row in interactions.iterrows():
            uid = int(row['user_id'])
            iid = int(row['item_id'])
            is_click = int(row.get('is_click', 1))
            if not is_click:
                continue
            author_id = item_author_map.get(iid, 0)
            if author_id > 0 and author_id in self.user_follows.get(uid, set()):
                self.user_author_interact[uid][author_id] += 1

        print(f"关注召回构建完成: "
              f"{len(self.author_items)} 个活跃作者, "
              f"{sum(len(v) for v in self.author_items.values())} 条候选物品")

    def recommend(self, user_id: int, n: int = 30,
                  exclude_items: Optional[set] = None) -> List[Tuple[int, float]]:
        """
        关注召回
        Args:
            user_id: 用户 ID
            n: 召回数量
            exclude_items: 排除的物品集合
        Returns:
            [(item_id, score), ...]
        """
        followed_authors = self.user_follows.get(user_id, set())
        if not followed_authors:
            return []

        exclude = exclude_items or set()

        # 对关注作者按互动频次排序（优先高互动作者）
        author_interact = self.user_author_interact.get(user_id, {})
        sorted_authors = sorted(
            followed_authors,
            key=lambda aid: author_interact.get(aid, 0),
            reverse=True
        )

        results: List[Tuple[int, float]] = []
        for author_id in sorted_authors:
            author_item_list = self.author_items.get(author_id, [])
            count = 0
            for item_info in author_item_list:
                if count >= self.max_items_per_author:
                    break
                item_id = item_info['item_id']
                if item_id in exclude:
                    continue
                # 得分 = 时效新鲜度 * 作者互动权重
                interact_weight = 1 + np.log1p(author_interact.get(author_id, 0))
                score = item_info['freshness_score'] * interact_weight
                results.append((item_id, float(score)))
                count += 1

            if len(results) >= n:
                break

        # 按得分排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n]

    def add_follow(self, user_id: int, author_id: int):
        """实时新增关注关系"""
        self.user_follows[user_id].add(author_id)

    def remove_follow(self, user_id: int, author_id: int):
        """实时取消关注"""
        self.user_follows[user_id].discard(author_id)

    def save(self, filepath: str):
        data = {
            'max_items_per_author': self.max_items_per_author,
            'time_window_days': self.time_window_days,
            'decay_hours': self.decay_hours,
            'user_follows': {k: list(v) for k, v in self.user_follows.items()},
            'user_author_interact': {k: dict(v) for k, v in self.user_author_interact.items()},
            'author_items': dict(self.author_items),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"FollowRecall 已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'FollowRecall':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        model = cls(
            max_items_per_author=data['max_items_per_author'],
            time_window_days=data['time_window_days'],
            decay_hours=data['decay_hours'],
        )
        model.user_follows = defaultdict(set,
            {k: set(v) for k, v in data['user_follows'].items()})
        model.user_author_interact = defaultdict(
            lambda: defaultdict(int),
            {k: defaultdict(int, v) for k, v in data['user_author_interact'].items()})
        model.author_items = defaultdict(list, data['author_items'])
        print(f"FollowRecall 已从 {filepath} 加载")
        return model
