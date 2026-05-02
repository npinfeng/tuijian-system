"""
实验评估脚本：召回(ItemCF/UserCF) + 精排(DeepFM)
"""

import sys
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config, gauc
from src.data.kuairand_loader import (
    load_kuairand_splits, 
    load_user_features, 
    load_video_features,
    build_din_features,
    get_kuairand_feature_columns
)
from src.recall import ItemCF, UserCF
from src.ranking.deepfm import DeepFM

def evaluate_recall(model, test_clicks, top_k=50):
    recalls = []
    hit_counts = 0
    total_users = 0
    for user_id, ground_truth in tqdm(test_clicks.items(), desc=f"评估 {model.__class__.__name__}"):
        try:
            recs = model.recommend(user_id, n=top_k)
            if not recs:
                recalls.append(0)
                total_users += 1
                continue
            rec_ids = set([item[0] for item in recs])
            hit = len(rec_ids & ground_truth)
            recalls.append(hit / len(ground_truth))
            if hit > 0: hit_counts += 1
            total_users += 1
        except: continue
    return np.mean(recalls), hit_counts / total_users

def main():
    config = load_config('config/config_cf_deepfm.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载数据
    user_features = load_user_features()
    _, _, test_log = load_kuairand_splits(val_ratio=0.2, verbose=False)
    
    test_pos = test_log[test_log['is_click'] == 1]
    test_clicks = {}
    for uid, group in test_pos.groupby('user_id'):
        test_clicks[int(uid)] = set(group['item_id'].astype(int).tolist())
    
    # 2. 评估召回
    print("\n" + "="*30 + " 召回阶段评估 " + "="*30)
    for name, path in [('ItemCF', 'models/itemcf_experiment.pkl'), ('UserCF', 'models/usercf_experiment.pkl')]:
        if os.path.exists(path):
            model = ItemCF.load(path) if 'itemcf' in path else UserCF.load(path)
            model.filter_history = False
            rec, hit = evaluate_recall(model, test_clicks)
            print(f"{name:10s} | Recall@50: {rec:.4f} | HitRate@50: {hit:.4f}")

    # 3. 评估精排
    print("\n" + "="*30 + " 精排阶段评估 (DeepFM) " + "="*30)
    model_path = 'models/deepfm_experiment_best.pt'
    if os.path.exists(model_path):
        (u_cols, i_cols, c_cols, _) = get_kuairand_feature_columns()
        model = DeepFM(user_feature_columns=u_cols, item_feature_columns=i_cols, context_feature_columns=c_cols)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
        
        test_df = build_din_features(test_log, user_features)
        all_preds = []
        all_labels = test_df['label'].values
        feat_names = [c['name'] for c in u_cols + i_cols + c_cols]
        
        with torch.no_grad():
            for i in tqdm(range(0, len(test_df), 1024), desc="DeepFM 预测"):
                batch = test_df.iloc[i:i+1024]
                inputs = {name: torch.tensor(batch[name].values, dtype=torch.long if name in ['user_id', 'item_id', 'active_degree', 'hour', 'tab'] else torch.float32).to(device) 
                         for name in feat_names}
                preds = model(inputs).cpu().numpy()
                all_preds.extend(preds)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(all_labels, all_preds)
        gauc_val = gauc(test_df['user_id'].values, all_labels, np.array(all_preds))
        print(f"DeepFM | AUC: {auc:.4f} | GAUC: {gauc_val:.4f}")
    
    print("\n评估完成！")

if __name__ == "__main__":
    main()
