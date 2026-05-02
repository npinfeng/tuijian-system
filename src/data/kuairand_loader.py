"""
KuaiRand-1K 数据集加载与预处理模块

数据划分策略：
  - 训练集 (train) : log_standard_4_08_to_4_21 （时间上更早的标准日志）
  - 验证集 (val)   : log_standard_4_22_to_5_08 的前 80%（按时间）
  - 测试集 (test)  : log_random_4_22_to_5_08   （随机曝光日志，无位置偏差，评估更客观）

KuaiRand 数据集字段说明：
  - user_id      : 用户 ID (0~999)
  - video_id     : 视频 ID (0~4371899)
  - time_ms      : 交互时间戳（毫秒）
  - is_click     : 是否点击（标签）
  - is_like      : 是否点赞
  - is_follow    : 是否关注
  - is_comment   : 是否评论
  - is_forward   : 是否转发
  - long_view    : 是否长播放
  - play_time_ms : 播放时长（毫秒）
  - duration_ms  : 视频总时长（毫秒）
  - tab          : 曝光所在 tab（0=推荐，1=关注等）
"""

import os
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ──────────────────────────────────────────────
# 数据集常量（从完整数据集统计得出）
# ──────────────────────────────────────────────
def get_default_data_dir():
    # 尝试加载配置文件
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return Path(config['data'].get('kuairand_path', r"e:\data\KuaiRand-1K\data"))
    except:
        pass
    return Path(r"e:\data\KuaiRand-1K\data")

KUAIRAND_DATA_DIR = get_default_data_dir()

NUM_USERS  = 1000        # user_id: 0~999
NUM_VIDEOS = 4371900     # video_id: 0~4371899  (max_id + 1，用于 Embedding vocab_size)
NUM_TABS   = 15          # tab 值域 0~14
NUM_HOURS  = 24          # 小时 0~23

# 用户活跃度标签映射
ACTIVE_DEGREE_MAP = {
    'full_active':        0,
    'high_active':        1,
    'middle_active':      2,
    'low_active':         3,
    'single_low_active':  4,
    '2_14_day_new':       5,
    '30day_retention':    6,
}
NUM_ACTIVE_DEGREES = len(ACTIVE_DEGREE_MAP)

# 视频类型映射
VIDEO_TYPE_MAP = {'NORMAL': 0, 'AD': 1, 'UNKNOWN': 2}
NUM_VIDEO_TYPES = len(VIDEO_TYPE_MAP)


# ══════════════════════════════════════════════
# 1. 原始数据加载
# ══════════════════════════════════════════════

def load_user_features(data_dir: Path = KUAIRAND_DATA_DIR) -> pd.DataFrame:
    """
    加载用户特征表
    返回字段：user_id, active_degree（整数编码）, is_live_streamer,
              is_video_author, follow_user_num, fans_user_num,
              register_days
    """
    df = pd.read_csv(data_dir / "user_features_1k.csv")

    # 活跃度编码
    df['active_degree'] = df['user_active_degree'].map(ACTIVE_DEGREE_MAP).fillna(0).astype(int)

    # 保留核心特征
    keep_cols = ['user_id', 'active_degree', 'is_live_streamer',
                 'is_video_author', 'follow_user_num', 'fans_user_num', 'register_days']
    df = df[keep_cols].copy()

    # 异常值处理（is_live_streamer 有负值，clip 到 0）
    df['is_live_streamer'] = df['is_live_streamer'].clip(0, 1)
    df['is_video_author']  = df['is_video_author'].clip(0, 1)

    # 数值特征归一化
    for col in ['follow_user_num', 'fans_user_num', 'register_days']:
        col_max = df[col].max()
        if col_max > 0:
            df[col] = df[col] / col_max

    return df.set_index('user_id')


def load_video_features(data_dir: Path = KUAIRAND_DATA_DIR,
                        sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    加载视频特征表（basic）
    返回字段：video_id, video_type（整数编码）, duration_s（秒）, tag
    由于视频数量很大（437万），可以通过 sample_n 限制加载数量
    """
    usecols = ['video_id', 'video_type', 'video_duration', 'tag']
    df = pd.read_csv(
        data_dir / "video_features_basic_1k.csv",
        usecols=usecols,
        nrows=sample_n
    )

    df['video_type_id'] = df['video_type'].map(VIDEO_TYPE_MAP).fillna(2).astype(int)
    df['duration_s']    = (df['video_duration'].fillna(0) / 1000).clip(0, 600)  # ms → s，最多10分钟
    # 处理可能存在的多个标签（如 "20,43"），只取第一个
    if df['tag'].dtype == object:
        df['tag'] = df['tag'].astype(str).str.split(',').str[0].replace('nan', '0').astype(int)
    else:
        df['tag'] = df['tag'].fillna(0).astype(int)

    return df[['video_id', 'video_type_id', 'duration_s', 'tag']].set_index('video_id')


def _load_log(csv_path: Path, chunksize: Optional[int] = None) -> pd.DataFrame:
    """
    内部函数：加载一个交互日志文件，统一字段处理
    """
    usecols = ['user_id', 'video_id', 'time_ms', 'is_click',
               'is_like', 'is_follow', 'is_comment', 'is_forward',
               'long_view', 'play_time_ms', 'duration_ms', 'tab']

    if chunksize:
        chunks = []
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(csv_path, usecols=usecols)

    # 时间戳转换
    df['timestamp'] = pd.to_datetime(df['time_ms'], unit='ms')
    df['hour']      = df['timestamp'].dt.hour

    # 播放完成率（安全计算，避免除零）
    df['play_ratio'] = np.where(
        df['duration_ms'] > 0,
        (df['play_time_ms'] / df['duration_ms']).clip(0, 1),
        0.0
    )

    # 重命名对齐项目内部字段名
    df = df.rename(columns={'video_id': 'item_id'})

    return df.sort_values('timestamp').reset_index(drop=True)


# ══════════════════════════════════════════════
# 2. 主要数据集加载函数（外部调用入口）
# ══════════════════════════════════════════════

def load_kuairand_splits(
    data_dir: Path = KUAIRAND_DATA_DIR,
    use_early_log_as_train: bool = True,
    val_ratio: float = 0.2,
    max_train_rows: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载 KuaiRand-1K 数据集并划分为训练/验证/测试集。

    划分策略：
      - train : log_standard_4_08_to_4_21  （时间早，用于训练）
                + log_standard_4_22_to_5_08 的前 (1-val_ratio)
      - val   : log_standard_4_22_to_5_08 的后 val_ratio（时间晚）
      - test  : log_random_4_22_to_5_08   （无偏随机曝光）

    Args:
        data_dir              : KuaiRand-1K 数据目录路径
        use_early_log_as_train: 是否把 4_08_to_4_21 的日志也加入训练集
        val_ratio             : 从 log_standard_4_22 中切出验证集的比例
        max_train_rows        : 限制训练集最大行数（None = 不限制）
        verbose               : 是否打印统计信息

    Returns:
        (train_df, val_df, test_df)  — 均包含统一处理后的交互字段
    """
    data_dir = Path(data_dir)

    # ── 加载日志 ──────────────────────────────────
    if verbose:
        print("正在加载 log_standard_4_22_to_5_08 ...")
    log_std_new = _load_log(data_dir / "log_standard_4_22_to_5_08_1k.csv")

    if verbose:
        print("正在加载 log_random_4_22_to_5_08 (测试集) ...")
    test_df = _load_log(data_dir / "log_random_4_22_to_5_08_1k.csv")

    # ── 从 log_standard_4_22 中按时间切出 val ────
    split_idx = int(len(log_std_new) * (1.0 - val_ratio))
    val_df    = log_std_new.iloc[split_idx:].reset_index(drop=True)
    train_new = log_std_new.iloc[:split_idx].reset_index(drop=True)

    # ── 可选：加入更早的训练日志 ─────────────────
    if use_early_log_as_train:
        if verbose:
            print("正在加载 log_standard_4_08_to_4_21 (更早训练数据) ...")
        log_std_old = _load_log(data_dir / "log_standard_4_08_to_4_21_1k.csv")
        train_df = pd.concat([log_std_old, train_new], ignore_index=True)
        train_df = train_df.sort_values('timestamp').reset_index(drop=True)
    else:
        train_df = train_new

    # ── 限制训练集大小 ───────────────────────────
    if max_train_rows and len(train_df) > max_train_rows:
        # 保留时间最近的 max_train_rows 条（模拟真实场景）
        train_df = train_df.tail(max_train_rows).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*50}")
        print(f"数据集划分完成：")
        print(f"  训练集 (train) : {len(train_df):>10,} 条  | CTR={train_df['is_click'].mean():.3f}")
        print(f"  验证集 (val)   : {len(val_df):>10,} 条  | CTR={val_df['is_click'].mean():.3f}")
        print(f"  测试集 (test)  : {len(test_df):>10,} 条  | CTR={test_df['is_click'].mean():.3f}")
        print(f"{'='*50}\n")

    return train_df, val_df, test_df


# ══════════════════════════════════════════════
# 3. DIN 模型所需的特征工程
# ══════════════════════════════════════════════

def build_din_features(
    interactions: pd.DataFrame,
    user_features: pd.DataFrame,
    max_seq_len: int = 50,
    label_col: str = 'is_click',
) -> pd.DataFrame:
    """
    在交互日志基础上，合并用户特征并构建 DIN 所需的行为序列。

    Args:
        interactions : 交互日志 DataFrame（来自 load_kuairand_splits）
        user_features: load_user_features() 返回的用户特征（以 user_id 为 index）
        max_seq_len  : 用户行为序列的最大长度
        label_col    : 标签字段名

    Returns:
        包含以下字段的 DataFrame：
          user_id, item_id, active_degree, is_live_streamer, is_video_author,
          follow_user_num, fans_user_num, register_days,
          hour, tab, play_ratio, label,
          hist_item_seq (List[int]), seq_length (int)
    """
    df = interactions.copy()

    # 1. 合并用户特征
    df = df.join(user_features, on='user_id', how='left')

    # 填充未匹配用户的特征（新用户 cold start）
    df['active_degree']    = df['active_degree'].fillna(0).astype(int)
    df['is_live_streamer'] = df['is_live_streamer'].fillna(0).astype(int)
    df['is_video_author']  = df['is_video_author'].fillna(0).astype(int)
    df['follow_user_num']  = df['follow_user_num'].fillna(0).astype(float)
    df['fans_user_num']    = df['fans_user_num'].fillna(0).astype(float)
    df['register_days']    = df['register_days'].fillna(0).astype(float)

    # 2. 标签列
    df['label'] = df[label_col].astype(int)

    # 3. 构建用户历史行为序列（只使用点击行为）
    print("  构建用户历史行为序列...")
    clicked = df[df['is_click'] == 1].sort_values('timestamp')
    user_hist: Dict[int, List[int]] = {}
    for uid, grp in clicked.groupby('user_id'):
        user_hist[uid] = grp['item_id'].tolist()

    # 为每条样本挂载历史序列（截取该样本时间点之前的）
    # 简化处理：直接取用户全量历史的最近 max_seq_len 条
    def get_seq(user_id):
        seq = user_hist.get(user_id, [])[-max_seq_len:]
        return seq

    df['hist_item_seq'] = df['user_id'].map(get_seq)
    df['seq_length']    = df['hist_item_seq'].apply(len)

    # 4. 选择输出字段
    out_cols = [
        'user_id', 'item_id',
        'active_degree', 'is_live_streamer', 'is_video_author',
        'follow_user_num', 'fans_user_num', 'register_days',
        'hour', 'tab', 'play_ratio',
        'label', 'hist_item_seq', 'seq_length',
    ]
    df = df[out_cols].reset_index(drop=True)

    return df


# ══════════════════════════════════════════════
# 4. 特征列配置（供 DIN 模型使用）
# ══════════════════════════════════════════════

def get_kuairand_feature_columns() -> Tuple[List, List, List, List]:
    """
    返回适配 KuaiRand-1K 的 DIN 特征列配置。

    Returns:
        (user_feature_columns, item_feature_columns,
         context_feature_columns, behavior_feature_columns)
    """
    user_feature_columns = [
        {'name': 'user_id',          'type': 'categorical', 'vocab_size': NUM_USERS},
        {'name': 'active_degree',    'type': 'categorical', 'vocab_size': NUM_ACTIVE_DEGREES},
        {'name': 'is_live_streamer', 'type': 'numerical'},
        {'name': 'is_video_author',  'type': 'numerical'},
        {'name': 'follow_user_num',  'type': 'numerical'},
        {'name': 'fans_user_num',    'type': 'numerical'},
        {'name': 'register_days',    'type': 'numerical'},
    ]

    item_feature_columns = [
        {'name': 'item_id', 'type': 'categorical', 'vocab_size': NUM_VIDEOS},
    ]

    context_feature_columns = [
        {'name': 'hour', 'type': 'categorical', 'vocab_size': NUM_HOURS},
        {'name': 'tab',  'type': 'categorical', 'vocab_size': NUM_TABS},
    ]

    behavior_feature_columns = [
        {'name': 'hist_item_seq', 'type': 'sequence'},
    ]

    return (user_feature_columns, item_feature_columns,
            context_feature_columns, behavior_feature_columns)


# ══════════════════════════════════════════════
# 5. 快速验证（直接运行此文件时执行）
# ══════════════════════════════════════════════

if __name__ == '__main__':
    print("加载用户特征...")
    user_feat = load_user_features()
    print(f"  用户数: {len(user_feat)}")
    print(user_feat.head(3))

    print("\n加载数据集划分...")
    train, val, test = load_kuairand_splits(
        use_early_log_as_train=False,  # 快速验证先不加早期日志
        max_train_rows=100_000,
        verbose=True,
    )

    print("构建 DIN 特征（训练集前5000条示例）...")
    sample = build_din_features(train.head(5000), user_feat)
    print(sample[['user_id', 'item_id', 'label', 'seq_length']].head())
    print(f"\n示例序列: {sample['hist_item_seq'].iloc[10]}")
