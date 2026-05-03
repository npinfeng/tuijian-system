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
    NUM_VIDEOS,
    NUM_ACTIVE_DEGREES,
    NUM_VIDEO_TYPES
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
    # 从 config 读取双塔超参
    tt_cfg = config.get('recall', {}).get('two_tower', {})
    tt_epochs      = tt_cfg.get('epochs', 5)
    tt_batch_size  = tt_cfg.get('batch_size', 1024)
    tt_lr          = tt_cfg.get('learning_rate', 0.001)
    tt_emb_dim     = tt_cfg.get('user_emb_dim', 64)
    tt_train_rows  = tt_cfg.get('max_train_rows', 200000)

    # 增强特征配置
    user_cols = [
        {'name': 'user_id',          'type': 'categorical', 'vocab_size': NUM_USERS},
        {'name': 'active_degree',    'type': 'categorical', 'vocab_size': NUM_ACTIVE_DEGREES},
        {'name': 'is_live_streamer', 'type': 'numerical'},
        {'name': 'is_video_author',  'type': 'numerical'},
    ]
    item_cols = [
        {'name': 'item_id',       'type': 'categorical', 'vocab_size': NUM_VIDEOS},
        {'name': 'video_type_id', 'type': 'categorical', 'vocab_size': NUM_VIDEO_TYPES},
        {'name': 'tag',           'type': 'categorical', 'vocab_size': 5000}, # 假设标签量级
        {'name': 'duration_s',    'type': 'numerical'},
    ]

    tt_model = TwoTowerModel(
        user_feature_columns=user_cols,
        item_feature_columns=item_cols,
        embedding_dim=tt_emb_dim,
        user_hidden_units=tt_cfg.get('hidden_units', [256, 128]),
        item_hidden_units=tt_cfg.get('hidden_units', [256, 128]),
        dropout_rate=tt_cfg.get('dropout', 0.2)
    )
    tt_trainer = TwoTowerTrainer(tt_model, config={
        'learning_rate': tt_lr,
        'weight_decay': tt_cfg.get('weight_decay', 1e-5),
    })

    # 准备增强版 DataLoader
    class TTDataset(torch.utils.data.Dataset):
        def __init__(self, df, user_features, video_features):
            # 合并特征
            self.data = df[['user_id', 'item_id']].copy()
            # 这里的 item_id 对应视频特征里的 video_id (index)
            self.u_feat = user_features
            self.v_feat = video_features
            
        def __len__(self): return len(self.data)
        
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            uid = row['user_id']
            iid = row['item_id']
            
            res = {}
            # 用户端特征
            u_info = self.u_feat.loc[uid]
            res['user_id'] = torch.tensor(uid, dtype=torch.long)
            res['active_degree'] = torch.tensor(u_info['active_degree'], dtype=torch.long)
            res['is_live_streamer'] = torch.tensor(u_info['is_live_streamer'], dtype=torch.float32)
            res['is_video_author'] = torch.tensor(u_info['is_video_author'], dtype=torch.float32)
            
            # 物品端特征
            try:
                v_info = self.v_feat.loc[iid]
                res['item_id'] = torch.tensor(iid, dtype=torch.long)
                res['video_type_id'] = torch.tensor(v_info['video_type_id'], dtype=torch.long)
                res['tag'] = torch.tensor(v_info['tag'], dtype=torch.long)
                res['duration_s'] = torch.tensor(v_info['duration_s'], dtype=torch.float32)
            except KeyError:
                # 缺失补 0
                res['item_id'] = torch.tensor(iid, dtype=torch.long)
                res['video_type_id'] = torch.tensor(2, dtype=torch.long)
                res['tag'] = torch.tensor(0, dtype=torch.long)
                res['duration_s'] = torch.tensor(0.0, dtype=torch.float32)
                
            return res

    # 准备特征数据供 Dataset 使用 (确保 video_features 以 video_id 为索引)
    if 'video_id' in video_features.columns:
        video_features_idx = video_features.set_index('video_id')
    else:
        video_features_idx = video_features

    # 正样本：训练集点击 & 验证集点击
    pos_val_log = val_log[val_log['is_click'] == 1].copy()
    # 随机采样训练集，避免只训练最老的数据
    train_data = pos_train_log.sample(n=min(len(pos_train_log), tt_train_rows), random_state=42)
    
    tt_train_loader = torch.utils.data.DataLoader(
        TTDataset(train_data, user_features, video_features_idx),
        batch_size=tt_batch_size, shuffle=True
    )
    tt_val_loader = torch.utils.data.DataLoader(
        TTDataset(pos_val_log, user_features, video_features_idx),
        batch_size=tt_batch_size, shuffle=True  # 验证集也必须打乱，防止局部极度相似导致 InfoNCE 计算偏差
    )
    
    # 从 config.training 读取 patience
    train_patience = config.get('training', {}).get('early_stopping_patience', 5)
    
    print(f"双塔训练集: {min(len(pos_train_log), tt_train_rows):,} 条  "
          f"验证集: {len(pos_val_log):,} 条  epochs={tt_epochs} patience={train_patience}")
    
    tt_trainer.train(tt_train_loader, tt_val_loader, epochs=tt_epochs,
                     save_path='models/two_tower_kuairand',
                     patience=train_patience)
    
    print("\n" + "=" * 60)
    print("所有路召回模型训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
