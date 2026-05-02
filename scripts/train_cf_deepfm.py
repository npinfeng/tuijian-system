"""
实验训练脚本：召回(ItemCF/UserCF) + 精排(DeepFM)
针对 KuaiRand-1K 数据集
"""

import sys
import os
from pathlib import Path
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config
from src.data.kuairand_loader import (
    load_kuairand_splits,
    load_user_features,
    load_video_features,
    get_kuairand_feature_columns,
    build_din_features # 虽然名字带DIN，但其实是通用的排序特征构建
)
from src.recall import ItemCF, UserCF
from src.ranking.deepfm import DeepFM, DeepFMTrainer

def main():
    # 1. 加载配置和数据
    config = load_config('config/config_cf_deepfm.yaml')
    print("=" * 60)
    print("开始实验：召回(CF) + 精排(DeepFM)")
    print("=" * 60)
    
    user_features = load_user_features()
    train_log, val_log, _ = load_kuairand_splits(
        use_early_log_as_train=True,
        val_ratio=0.1,
        max_train_rows=config['ranking']['deepfm'].get('max_train_rows', 500000),
        verbose=True
    )
    
    # ── 阶段 1: 训练召回模型 (ItemCF & UserCF) ───────────
    print("\n[Step 1/2] 训练召回模型...")
    pos_train_log = train_log[train_log['is_click'] == 1].copy()
    
    print("训练 ItemCF...")
    itemcf = ItemCF(top_k_similar=100)
    itemcf.fit(pos_train_log)
    itemcf.save('models/itemcf_experiment.pkl')
    
    print("训练 UserCF...")
    usercf = UserCF(top_k_similar=100)
    usercf.fit(pos_train_log)
    usercf.save('models/usercf_experiment.pkl')
    
    # ── 阶段 2: 训练精排模型 (DeepFM) ───────────────────
    print("\n[Step 2/2] 训练精排模型 (DeepFM)...")
    
    # 准备特征
    print("构建特征...")
    train_df = build_din_features(train_log, user_features)
    val_df = build_din_features(val_log, user_features)
    
    (u_cols, i_cols, c_cols, _) = get_kuairand_feature_columns()
    dfm_config = config['ranking']['deepfm']
    
    model = DeepFM(
        user_feature_columns=u_cols,
        item_feature_columns=i_cols,
        context_feature_columns=c_cols,
        embedding_dim=dfm_config.get('embedding_dim', 16),
        dnn_hidden_units=dfm_config.get('dnn_hidden_units', [256, 128, 64])
    )
    
    trainer = DeepFMTrainer(model, config={'learning_rate': dfm_config.get('learning_rate', 0.001)})
    
    # 构造 DataLoader
    class DeepFMDataset(Dataset):
        def __init__(self, df):
            self.labels = df['label'].values
            self.data = df
            self.feat_names = [c['name'] for c in u_cols + i_cols + c_cols]
            
        def __len__(self): return len(self.labels)
        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            features = {name: torch.tensor(row[name], dtype=torch.long if name in ['user_id', 'item_id', 'active_degree', 'hour', 'tab'] else torch.float32) 
                       for name in self.feat_names}
            return features, torch.tensor(self.labels[idx], dtype=torch.float32)

    train_loader = DataLoader(DeepFMDataset(train_df), batch_size=dfm_config.get('batch_size', 1024), shuffle=True)
    val_loader = DataLoader(DeepFMDataset(val_df), batch_size=dfm_config.get('batch_size', 1024), shuffle=False)
    
    trainer.train(
        train_loader, 
        val_loader, 
        epochs=dfm_config.get('epochs', 10),
        save_path='models/deepfm_experiment'
    )
    
    print("\n实验模型训练全部完成！")

if __name__ == "__main__":
    main()
