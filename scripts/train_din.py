"""
DIN 模型训练脚本 —— 使用 KuaiRand-1K 真实数据集

数据集划分：
  - 训练集 : log_standard_4_08_to_4_21 + log_standard_4_22_to_5_08 前80%
  - 验证集 : log_standard_4_22_to_5_08 后20%（时间序划分）
  - 测试集 : log_random_4_22_to_5_08（无偏随机曝光，评估最客观）
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

from src.ranking.din_model import DIN, DINTrainer
from src.utils.common import load_config
from src.data.kuairand_loader import (
    load_kuairand_splits,
    load_user_features,
    build_din_features,
    get_kuairand_feature_columns,
)


# ══════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════

class KuaiRandDINDataset(Dataset):
    """
    KuaiRand-1K 数据集的 PyTorch Dataset 封装
    特征字段与 DIN 模型输入保持一致
    """

    def __init__(self, df: pd.DataFrame, max_seq_len: int = 50):
        self.max_seq_len = max_seq_len

        # ── 类别特征 ─────────────────────────────
        self.user_id       = df['user_id'].values.astype(np.int64)
        self.active_degree = df['active_degree'].values.astype(np.int64)
        self.item_id       = df['item_id'].values.astype(np.int64)
        self.hour          = df['hour'].values.astype(np.int64)
        self.tab           = df['tab'].values.astype(np.int64)

        # ── 数值特征 ─────────────────────────────
        self.is_live_streamer = df['is_live_streamer'].values.astype(np.float32)
        self.is_video_author  = df['is_video_author'].values.astype(np.float32)
        self.follow_user_num  = df['follow_user_num'].values.astype(np.float32)
        self.fans_user_num    = df['fans_user_num'].values.astype(np.float32)
        self.register_days    = df['register_days'].values.astype(np.float32)

        # ── 标签 ─────────────────────────────────
        self.labels      = df['label'].values.astype(np.float32)
        self.seq_length  = df['seq_length'].values.astype(np.int64)

        # ── 行为序列（padding/truncation）────────
        hist_item_seq = []
        for seq in df['hist_item_seq']:
            seq = list(seq)
            if len(seq) < max_seq_len:
                seq = seq + [0] * (max_seq_len - len(seq))   # zero-padding
            else:
                seq = seq[-max_seq_len:]                       # 保留最近的
            hist_item_seq.append(seq)
        self.hist_item_seq = np.array(hist_item_seq, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {
            # 类别特征
            'user_id':          torch.tensor(self.user_id[idx],       dtype=torch.long),
            'active_degree':    torch.tensor(self.active_degree[idx],  dtype=torch.long),
            'item_id':          torch.tensor(self.item_id[idx],        dtype=torch.long),
            'hour':             torch.tensor(self.hour[idx],            dtype=torch.long),
            'tab':              torch.tensor(self.tab[idx],             dtype=torch.long),
            # 数值特征
            'is_live_streamer': torch.tensor(self.is_live_streamer[idx], dtype=torch.float32),
            'is_video_author':  torch.tensor(self.is_video_author[idx],  dtype=torch.float32),
            'follow_user_num':  torch.tensor(self.follow_user_num[idx],  dtype=torch.float32),
            'fans_user_num':    torch.tensor(self.fans_user_num[idx],    dtype=torch.float32),
            'register_days':    torch.tensor(self.register_days[idx],    dtype=torch.float32),
            # 序列特征
            'hist_item_seq':    torch.tensor(self.hist_item_seq[idx],    dtype=torch.long),
            'seq_length':       torch.tensor(self.seq_length[idx],       dtype=torch.long),
        }
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


# ══════════════════════════════════════════════
# 主训练流程
# ══════════════════════════════════════════════

def main():
    """主训练流程"""

    # ── 1. 加载配置 ───────────────────────────
    config     = load_config('config/config.yaml')
    din_config = config['ranking']['din']

    max_seq_len = din_config.get('max_seq_len', 50)
    batch_size  = din_config.get('batch_size', 1024)
    epochs      = din_config.get('epochs', 5)

    # ── 2. 加载 KuaiRand-1K 数据 ─────────────
    print("=" * 55)
    print("加载 KuaiRand-1K 数据集")
    print("=" * 55)

    # 加载用户特征
    user_feat = load_user_features()

    # 加载交互日志并划分
    # max_train_rows 控制内存使用；设为 None 使用全量（约 700万条）
    train_log, val_log, test_log = load_kuairand_splits(
        use_early_log_as_train=True,
        val_ratio=0.2,
        max_train_rows=din_config.get('max_train_rows', 500_000),
        verbose=True,
    )

    # ── 3. 特征工程 ───────────────────────────
    print("构建训练集特征...")
    train_df = build_din_features(train_log, user_feat, max_seq_len=max_seq_len)

    print("构建验证集特征...")
    val_df = build_din_features(val_log, user_feat, max_seq_len=max_seq_len)

    print("构建测试集特征...")
    test_df = build_din_features(test_log, user_feat, max_seq_len=max_seq_len)

    print(f"\n训练集: {len(train_df):,} 条 | 验证集: {len(val_df):,} 条 | 测试集: {len(test_df):,} 条")

    # ── 4. 构建 DataLoader ────────────────────
    train_dataset = KuaiRandDINDataset(train_df, max_seq_len=max_seq_len)
    val_dataset   = KuaiRandDINDataset(val_df,   max_seq_len=max_seq_len)
    test_dataset  = KuaiRandDINDataset(test_df,  max_seq_len=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=0)

    # ── 5. 特征列配置 & 构建模型 ──────────────
    (user_feat_cols, item_feat_cols,
     context_feat_cols, behavior_feat_cols) = get_kuairand_feature_columns()

    print("\n构建 DIN 模型（KuaiRand-1K 特征配置）...")
    model = DIN(
        user_feature_columns=user_feat_cols,
        item_feature_columns=item_feat_cols,
        context_feature_columns=context_feat_cols,
        behavior_feature_columns=behavior_feat_cols,
        embedding_dim=din_config.get('embedding_dim', 32),
        attention_hidden_units=din_config.get('attention_hidden_units', [80, 40]),
        dnn_hidden_units=din_config.get('dnn_hidden_units', [256, 128, 64]),
        dropout_rate=din_config.get('dropout', 0.2),
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数量: {n_params:,}")

    # ── 6. 训练 ───────────────────────────────
    print("\n开始训练...")
    trainer = DINTrainer(model, din_config)

    Path('models').mkdir(exist_ok=True)

    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        save_path='models/din_kuairand',
    )

    # ── 7. 测试集评估（无偏 log_random）────────
    print("\n" + "=" * 55)
    print("在测试集（log_random，无偏）上评估...")
    print("=" * 55)

    model.eval()
    device    = trainer.device
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for x, y in test_loader:
            x      = {k: v.to(device) for k, v in x.items()}
            y_pred = model(x).squeeze(-1)
            all_labels.append(y.numpy())
            all_preds.append(y_pred.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)

    if len(np.unique(all_labels)) > 1:
        test_auc = roc_auc_score(all_labels, all_preds)
        print(f"测试集 AUC (无偏评估): {test_auc:.4f}")
    else:
        print("测试集标签单一，无法计算 AUC")

    # 额外统计
    pred_binary = (all_preds > 0.5).astype(int)
    accuracy    = (pred_binary == all_labels).mean()
    print(f"测试集 Accuracy      : {accuracy:.4f}")
    print(f"测试集样本数         : {len(all_labels):,}")
    print(f"测试集正样本比例     : {all_labels.mean():.4f}")

    print("\n训练完成！模型已保存至 models/din_kuairand_best.pt")


if __name__ == '__main__':
    main()
