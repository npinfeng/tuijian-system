"""
模型训练脚本 - DIN模型
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path

from src.ranking.din_model import DIN, DINTrainer
from src.utils.common import load_config, split_by_time


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


def create_tf_dataset(df: pd.DataFrame, batch_size: int = 512) -> tf.data.Dataset:
    """
    创建TensorFlow Dataset
    """
    # 准备特征字典
    feature_dict = {
        'user_id': df['user_id'].values,
        'age': df['age'].values.astype(np.float32),
        'gender': df['gender'].values,
        'item_id': df['item_id'].values,
        'category': df['category'].values,
        'hour': df['hour'].values,
        'seq_length': df['seq_length'].values,
    }
    
    # 处理序列特征（padding）
    max_seq_len = 50
    hist_item_seq = []
    for seq in df['hist_item_seq']:
        if len(seq) < max_seq_len:
            seq = seq + [0] * (max_seq_len - len(seq))  # padding
        else:
            seq = seq[-max_seq_len:]  # 截断
        hist_item_seq.append(seq)
    
    feature_dict['hist_item_seq'] = np.array(hist_item_seq)
    
    # 标签
    labels = df['label'].values.astype(np.float32)
    
    # 创建dataset
    dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


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
    
    # 4. 创建数据集
    batch_size = din_config['batch_size']
    train_dataset = create_tf_dataset(train_df, batch_size)
    val_dataset = create_tf_dataset(val_df, batch_size)
    test_dataset = create_tf_dataset(test_df, batch_size)
    
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
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=din_config['epochs'],
        save_path='models/din_model'
    )
    
    # 7. 测试集评估
    print("\n在测试集上评估...")
    test_loss_metric = tf.keras.metrics.Mean()
    test_auc_metric = tf.keras.metrics.AUC()
    
    for x, y in test_dataset:
        y_pred = model(x, training=False)
        test_auc_metric.update_state(y, y_pred)
    
    print(f"测试集 AUC: {test_auc_metric.result():.4f}")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
