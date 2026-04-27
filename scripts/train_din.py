"""
模型训练脚本 - DIN模型 (PyTorch 版)
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

from src.ranking.din_model import DIN, DINTrainer
from src.utils.common import load_config, split_by_time


class DINDataset(Dataset):
    """DIN模型的PyTorch Dataset"""

    def __init__(self, df: pd.DataFrame, max_seq_len: int = 50):
        self.max_seq_len = max_seq_len

        # 标量特征
        self.user_id = df['user_id'].values
        self.age = df['age'].values.astype(np.float32)
        self.gender = df['gender'].values
        self.item_id = df['item_id'].values
        self.category = df['category'].values
        self.hour = df['hour'].values
        self.seq_length = df['seq_length'].values
        self.labels = df['label'].values.astype(np.float32)

        # 处理序列特征（padding）
        hist_item_seq = []
        for seq in df['hist_item_seq']:
            if len(seq) < max_seq_len:
                seq = seq + [0] * (max_seq_len - len(seq))  # padding
            else:
                seq = seq[-max_seq_len:]  # 截断
            hist_item_seq.append(seq)
        self.hist_item_seq = np.array(hist_item_seq)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = {
            'user_id': torch.tensor(self.user_id[idx], dtype=torch.long),
            'age': torch.tensor(self.age[idx], dtype=torch.float32),
            'gender': torch.tensor(self.gender[idx], dtype=torch.long),
            'item_id': torch.tensor(self.item_id[idx], dtype=torch.long),
            'category': torch.tensor(self.category[idx], dtype=torch.long),
            'hour': torch.tensor(self.hour[idx], dtype=torch.long),
            'seq_length': torch.tensor(self.seq_length[idx], dtype=torch.long),
            'hist_item_seq': torch.tensor(self.hist_item_seq[idx], dtype=torch.long),
        }
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label


def prepare_dataset():
    """
    准备训练数据集
    生成模拟数据用于演示
    """
    print("准备训练数据...")

    # 生成模拟数据
    n_samples = 100000
    n_users = 10000
    n_items = 50000

    data = {
        'user_id': np.random.randint(0, n_users, n_samples),
        'item_id': np.random.randint(0, n_items, n_samples),
        'age': np.random.randint(18, 60, n_samples),
        'gender': np.random.randint(0, 2, n_samples),
        'category': np.random.randint(0, 20, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'label': np.random.randint(0, 2, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min')
    }

    df = pd.DataFrame(data)

    # 构建用户行为序列
    print("构建用户行为序列...")
    user_sequences = {}
    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].sort_values('timestamp')
        seq = user_data['item_id'].tolist()[-50:]  # 最近50个
        user_sequences[user_id] = seq

    df['hist_item_seq'] = df['user_id'].map(lambda x: user_sequences.get(x, []))
    df['seq_length'] = df['hist_item_seq'].apply(len)

    return df


def build_feature_columns():
    """
    构建特征列配置
    """
    user_feature_columns = [
        {'name': 'user_id', 'type': 'categorical', 'vocab_size': 10000},
        {'name': 'age', 'type': 'numerical'},
        {'name': 'gender', 'type': 'categorical', 'vocab_size': 2},
    ]

    item_feature_columns = [
        {'name': 'item_id', 'type': 'categorical', 'vocab_size': 50000},
        {'name': 'category', 'type': 'categorical', 'vocab_size': 20},
    ]

    context_feature_columns = [
        {'name': 'hour', 'type': 'categorical', 'vocab_size': 24},
    ]

    behavior_feature_columns = [
        {'name': 'hist_item_seq', 'type': 'sequence'},
    ]

    return user_feature_columns, item_feature_columns, context_feature_columns, behavior_feature_columns


def main():
    """主训练流程"""

    # 1. 加载配置
    config = load_config('config/config.yaml')
    din_config = config['ranking']['din']

    # 2. 准备数据
    df = prepare_dataset()
    print(f"数据集大小: {len(df)}")

    # 按时间划分数据集
    train_df, val_df, test_df = split_by_time(df, train_days=70, val_days=15, test_days=15)

    print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

    # 3. 构建特征列
    user_feat_cols, item_feat_cols, context_feat_cols, behavior_feat_cols = build_feature_columns()

    # 4. 创建数据集和DataLoader
    batch_size = din_config['batch_size']
    train_dataset = DINDataset(train_df)
    val_dataset = DINDataset(val_df)
    test_dataset = DINDataset(test_df)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 5. 构建模型
    print("构建DIN模型...")
    model = DIN(
        user_feature_columns=user_feat_cols,
        item_feature_columns=item_feat_cols,
        context_feature_columns=context_feat_cols,
        behavior_feature_columns=behavior_feat_cols,
        embedding_dim=din_config['embedding_dim'],
        attention_hidden_units=din_config['attention_hidden_units'],
        dnn_hidden_units=din_config['dnn_hidden_units'],
        dropout_rate=din_config['dropout']
    )

    # 6. 训练模型
    print("开始训练...")
    trainer = DINTrainer(model, din_config)

    # 创建保存目录
    Path('models').mkdir(exist_ok=True)

    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=din_config['epochs'],
        save_path='models/din_model'
    )

    # 7. 测试集评估
    print("\n在测试集上评估...")
    model.eval()
    device = trainer.device
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for x, y in test_dataloader:
            x = {k: v.to(device) for k, v in x.items()}
            y_pred = model(x).squeeze(-1)
            all_labels.append(y.numpy())
            all_preds.append(y_pred.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)

    if len(np.unique(all_labels)) > 1:
        test_auc = roc_auc_score(all_labels, all_preds)
        print(f"测试集 AUC: {test_auc:.4f}")
    else:
        print("测试集标签单一，无法计算AUC")

    print("\n训练完成！")


if __name__ == '__main__':
    main()
