"""
多路召回与精排模型真实评估脚本
1. 评估各路召回模型的 Recall@K, HitRate@K
2. 评估 DIN 精排模型的 AUC, GAUC (在 log_random 无偏数据上)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config, gauc, print_metrics
from src.data.kuairand_loader import (
    load_kuairand_splits, 
    load_user_features, 
    load_video_features,
    build_din_features,
    get_kuairand_feature_columns
)
from src.recall import ItemCF, UserCF, HotRecall, NewItemRecall, FollowRecall, DeepWalkRecall, DeepWalkModel
from src.recall.two_tower_model import TwoTowerModel
from src.ranking.din_model import DIN

# ── 召回模型评估函数 ────────────────────────────────
def evaluate_recall_model(model, test_clicks: Dict[int, set], user_features_df: pd.DataFrame = None, top_k: int = 50):
    """
    评估单路召回模型指标
    """
    recalls = []
    hit_counts = 0
    total_users = 0
    
    # 只对有点击行为的用户进行评估
    for user_id, ground_truth in tqdm(test_clicks.items(), desc=f"评估 {model.__class__.__name__}"):
        # 获取推荐列表
        try:
            # 兼容不同模型的接口
            if hasattr(model, 'recommend'):
                # 特殊处理需要 user_features 的模型 (如 TwoTower)
                if 'user_features' in model.recommend.__code__.co_varnames:
                    u_feat = user_features_df.loc[user_id] if user_features_df is not None and user_id in user_features_df.index else None
                    recs = model.recommend(user_id, n=top_k, user_features=u_feat)
                else:
                    recs = model.recommend(user_id, n=top_k)
            else:
                continue
            
            if not recs:
                recalls.append(0)
                total_users += 1
                continue

            rec_ids = set([item[0] for item in recs])
            
            # 计算 Recall
            hit = len(rec_ids & ground_truth)
            recalls.append(hit / len(ground_truth))
            if hit > 0:
                hit_counts += 1
            total_users += 1
        except Exception as e:
            # print(f"Error evaluating user {user_id}: {e}")
            continue
            
    return {
        f'Recall@{top_k}': np.mean(recalls) if recalls else 0,
        f'HitRate@{top_k}': hit_counts / total_users if total_users > 0 else 0
    }

# ── 主评估流程 ─────────────────────────────────────
def main():
    print("="*60)
    print("开始多阶段推荐系统全面评估")
    print("="*60)
    
    config = load_config('config/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    user_features = load_user_features()
    # 视频特征需要将 video_id 统一为 item_id
    video_features = load_video_features().reset_index()
    video_features = video_features.rename(columns={'video_id': 'item_id'})
    
    _, _, test_log = load_kuairand_splits(
        use_early_log_as_train=False,
        val_ratio=0.2,
        verbose=True
    )
    
    # 构建测试集标准答案 {user_id: {clicked_item_ids}}
    test_pos = test_log[test_log['is_click'] == 1]
    test_clicks = {}
    for uid, group in test_pos.groupby('user_id'):
        test_clicks[int(uid)] = set(group['item_id'].astype(int).tolist())
    
    print(f"评估有效用户数: {len(test_clicks)}")

    # 2. 评估召回模型
    recall_results = {}
    
    # 定义召回模型加载配置
    recall_configs = [
        ('ItemCF', 'models/itemcf_kuairand.pkl', ItemCF),
        ('UserCF', 'models/usercf_kuairand.pkl', UserCF),
        ('HotRecall', 'models/hot_kuairand.pkl', HotRecall),
        ('NewItemRecall', 'models/new_kuairand.pkl', NewItemRecall),
        ('FollowRecall', 'models/follow_kuairand.pkl', FollowRecall),
    ]
    
    for name, path, cls in recall_configs:
        if os.path.exists(path):
            model = cls.load(path)
            # 对于 CF 模型，评估时关闭历史过滤，以获得更真实的 Recall
            if hasattr(model, 'filter_history'):
                model.filter_history = False
            recall_results[name] = evaluate_recall_model(model, test_clicks)

    # --- DeepWalk ---
    if os.path.exists('models/deepwalk_kuairand'):
        try:
            dw_m = DeepWalkModel.load('models/deepwalk_kuairand')
            model = DeepWalkRecall(dw_m)
            recall_results['DeepWalk'] = evaluate_recall_model(model, test_clicks)
        except: pass
    else:
        print("  - 跳过 DeepWalk: 模型文件 models/deepwalk_kuairand 不存在")

    # --- MIND ---
    if os.path.exists('models/mind_kuairand.model.pt'):
        try:
            from src.recall.mind_recall import MINDModel, MINDRecall
            mind_m = MINDModel(num_items=NUM_VIDEOS, embedding_dim=64)
            model = MINDRecall.load('models/mind_kuairand', mind_m)
            recall_results['MIND'] = evaluate_recall_model(model, test_clicks)
        except Exception as e:
            print(f"  - MIND 评估失败: {e}")
    else:
        print("  - 跳过 MIND: 模型文件 models/mind_kuairand.model.pt 不存在")

    # --- TwoTower ---
    tt_path = 'models/two_tower_kuairand_best.pt'
    if os.path.exists(tt_path):
        try:
            from src.recall.two_tower_model import TwoTowerModel, TwoTowerRecall
            (u_cols, i_cols, _, _) = get_kuairand_feature_columns()
            tt_model = TwoTowerModel(
                user_feature_columns=u_cols,
                item_feature_columns=i_cols,
                embedding_dim=config['recall']['two_tower'].get('user_emb_dim', 64)
            )
            tt_model.load_state_dict(torch.load(tt_path, map_location=device))
            model = TwoTowerRecall(tt_model, video_features)
            recall_results['TwoTower'] = evaluate_recall_model(model, test_clicks, user_features_df=user_features)
        except Exception as e:
            print(f"  - TwoTower 评估失败: {e}")
    else:
        print(f"  - 跳过 TwoTower: 模型文件 {tt_path} 不存在")

    print("\n[召回评估结果]:")
    print("-" * 60)
    print(f"{'Model':15s} | {'Recall@50':10s} | {'HitRate@50':10s}")
    print("-" * 60)
    for name, metrics in recall_results.items():
        print(f"{name:15s} | {metrics['Recall@50']:.4f}     | {metrics['HitRate@50']:.4f}")
    print("-" * 60)

    # 3. 评估精排模型 (DIN)
    print("\n" + "="*60)
    print("开始评估精排模型 (DIN)")
    print("="*60)
    
    din_model_path = 'models/din_kuairand_best.pt'
    if os.path.exists(din_model_path):
        # 准备特征
        (user_cols, item_cols, ctx_cols, beh_cols) = get_kuairand_feature_columns()
        din_config = config['ranking']['din']
        
        model = DIN(
            user_feature_columns=user_cols,
            item_feature_columns=item_cols,
            context_feature_columns=ctx_cols,
            behavior_feature_columns=beh_cols,
            embedding_dim=din_config.get('embedding_dim', 32)
        )
        model.load_state_dict(torch.load(din_model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # 构建测试特征 (log_random)
        print("构建测试集特征...")
        test_df = build_din_features(test_log, user_features, max_seq_len=50)
        
        # 批量预测
        all_preds = []
        all_labels = test_df['label'].values
        
        print("执行模型预测...")
        batch_size = 1024
        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), batch_size)):
                batch = test_df.iloc[i:i+batch_size]
                
                # 构造输入 tensor
                inputs = {
                    'user_id': torch.tensor(batch['user_id'].values, dtype=torch.long).to(device),
                    'item_id': torch.tensor(batch['item_id'].values, dtype=torch.long).to(device),
                    'active_degree': torch.tensor(batch['active_degree'].values, dtype=torch.long).to(device),
                    'hour': torch.tensor(batch['hour'].values, dtype=torch.long).to(device),
                    'tab': torch.tensor(batch['tab'].values, dtype=torch.long).to(device),
                    'is_live_streamer': torch.tensor(batch['is_live_streamer'].values, dtype=torch.float32).to(device),
                    'is_video_author': torch.tensor(batch['is_video_author'].values, dtype=torch.float32).to(device),
                    'follow_user_num': torch.tensor(batch['follow_user_num'].values, dtype=torch.float32).to(device),
                    'fans_user_num': torch.tensor(batch['fans_user_num'].values, dtype=torch.float32).to(device),
                    'register_days': torch.tensor(batch['register_days'].values, dtype=torch.float32).to(device),
                    'seq_length': torch.tensor(batch['seq_length'].values, dtype=torch.long).to(device),
                }
                
                # 处理行为序列
                seqs = [list(s) for s in batch['hist_item_seq'].values]
                # Padding to 50
                padded_seqs = [s + [0]*(50-len(s)) if len(s)<50 else s[-50:] for s in seqs]
                inputs['hist_item_seq'] = torch.tensor(padded_seqs, dtype=torch.long).to(device)
                
                preds = model(inputs).squeeze(-1).cpu().numpy()
                all_preds.extend(preds)
        
        all_preds = np.array(all_preds)
        
        # 计算指标
        from sklearn.metrics import roc_auc_score
        auc_score = roc_auc_score(all_labels, all_preds)
        gauc_score = gauc(test_df['user_id'].values, all_labels, all_preds)
        
        print("\n[精排评估结果]:")
        print(f"AUC  : {auc_score:.4f} (无偏评估)")
        print(f"GAUC : {gauc_score:.4f}")
    else:
        print(f"未找到 DIN 模型文件: {din_model_path}")

    print("\n评估完成！")

if __name__ == '__main__':
    main()
