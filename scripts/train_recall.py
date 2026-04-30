"""
多路召回模型训练脚本 (KuaiRand-1K 适配版)
包含 8 路召回模型的训练与保存：
1. ItemCF
2. UserCF
3. HotRecall (热门)
4. NewItemRecall (新内容)
5. FollowRecall (关注)
6. DeepWalk (图召回)
7. MIND (多兴趣)
8. TwoTower (双塔)
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import torch
from src.utils.common import load_config
from src.data.kuairand_loader import (
    load_kuairand_splits,
    load_user_features,
    load_video_features,
    NUM_USERS,
    NUM_VIDEOS
)

from src.recall import (
    ItemCF, UserCF, HotRecall, NewItemRecall, FollowRecall,
    DeepWalkModel, DeepWalkRecall
)
from src.recall.two_tower_model import TwoTowerModel, TwoTowerTrainer


def main():
    # 1. 加载配置和数据
    print("=" * 60)
    print("开始训练多路召回模型 (KuaiRand-1K)")
    print("=" * 60)
    
    config = load_config('config/config.yaml')
    data_dir = config['data'].get('kuairand_path', 'e:/data/KuaiRand-1K/data')
    
    user_features = load_user_features()
    # 全量加载视频特征用于热门和新内容扫描
    video_features = load_video_features().reset_index()
    
    train_log, val_log, test_log = load_kuairand_splits(
        use_early_log_as_train=True,
        val_ratio=0.1,
        max_train_rows=500000, # 限制部分数据用于加速训练
        verbose=True
    )
    
    # 只使用点击数据作为正样本
    pos_train_log = train_log[train_log['is_click'] == 1].copy()
    print(f"正样本交互数: {len(pos_train_log)}")
    
    # 创建模型保存目录
    Path('models').mkdir(exist_ok=True)
    
    # ── 1 & 2. 协同过滤 (ItemCF & UserCF) ────────────────
    print("\n[1/7] 训练 ItemCF...")
    itemcf = ItemCF(top_k_similar=100)
    itemcf.fit(pos_train_log)
    itemcf.save('models/itemcf_kuairand.pkl')
    
    print("\n[2/7] 训练 UserCF...")
    usercf = UserCF(top_k_similar=100)
    usercf.fit(pos_train_log)
    usercf.save('models/usercf_kuairand.pkl')
    
    # ── 3 & 4. 热门与新内容 (Hot & New) ──────────────────
    print("\n[3/7] 训练 HotRecall...")
    hot = HotRecall()
    hot.fit(train_log, video_features)
    hot.save('models/hot_kuairand.pkl')
    
    print("\n[4/7] 训练 NewItemRecall...")
    new_item = NewItemRecall(new_threshold_hours=72)
    new_item.fit(video_features, train_log)
    new_item.save('models/new_kuairand.pkl')
    
    # ── 5. 关注召回 (Follow) ───────────────────────────
    print("\n[5/7] 训练 FollowRecall (隐式关注)...")
    follow = FollowRecall()
    # 由于 KuaiRand 无显式关注表，使用历史交互推断
    follow.fit(train_log, video_features)
    follow.save('models/follow_kuairand.pkl')
    
    # ── 6. 图召回 (DeepWalk) ───────────────────────────
    print("\n[6/7] 训练 DeepWalk...")
    dw_model = DeepWalkModel(dimensions=64, walk_length=20, num_walks=5)
    dw_model.fit(pos_train_log)
    dw_model.save('models/deepwalk_kuairand')
    # DeepWalkRecall 不需要 fit，它是封装器
    
    # ── 7. 双塔召回 (TwoTower) ──────────────────────────
    print("\n[7/7] 训练 TwoTower...")
    # 简化的特征列
    user_cols = [{'name': 'user_id', 'type': 'categorical', 'vocab_size': NUM_USERS}]
    item_cols = [{'name': 'item_id', 'type': 'categorical', 'vocab_size': NUM_VIDEOS}]
    
    tt_model = TwoTowerModel(user_feature_columns=user_cols, item_feature_columns=item_cols, embedding_dim=32)
    tt_trainer = TwoTowerTrainer(tt_model, config={'learning_rate': 0.001})
    
    # 准备 DataLoader (简化版)
    class TTDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.u = df['user_id'].values
            self.i = df['item_id'].values
        def __len__(self): return len(self.u)
        def __getitem__(self, idx):
            return {'user_id': torch.tensor(self.u[idx], dtype=torch.long),
                    'item_id': torch.tensor(self.i[idx], dtype=torch.long)}
    
    tt_loader = torch.utils.data.DataLoader(TTDataset(pos_train_log.head(100000)), batch_size=1024, shuffle=True)
    tt_trainer.train(tt_loader, tt_loader, epochs=1, save_path='models/two_tower_kuairand')
    
    print("\n" + "=" * 60)
    print("所有路召回模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
