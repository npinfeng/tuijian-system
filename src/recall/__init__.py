"""
召回模块
支持 8 路多路召回:
  1. ItemCF  - 基于物品的协同过滤
  2. UserCF  - 基于用户的协同过滤
  3. TwoTower - 双塔模型（深度学习）
  4. MIND    - 多兴趣召回（胶囊网络）
  5. DeepWalk - 图召回（随机游走）
  6. HotRecall - 热门内容召回
  7. FollowRecall - 关注作者召回
  8. NewItemRecall - 新内容冷启动召回
"""

from src.recall.collaborative_filtering import ItemCF, UserCF
from src.recall.hot_recall import HotRecall, NewItemRecall
from src.recall.follow_recall import FollowRecall
from src.recall.deepwalk_recall import DeepWalkGraph, DeepWalkModel, DeepWalkRecall
from src.recall.mind_recall import MINDModel, MINDTrainer, MINDRecall

__all__ = [
    'ItemCF', 'UserCF',
    'HotRecall', 'NewItemRecall',
    'FollowRecall',
    'DeepWalkGraph', 'DeepWalkModel', 'DeepWalkRecall',
    'MINDModel', 'MINDTrainer', 'MINDRecall',
]
