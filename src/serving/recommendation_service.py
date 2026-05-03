"""
推荐服务主程序 (KuaiRand-1K 版)
整合多路召回、DIN精排、MMR重排等模块
"""

import sys
import os
import torch
import random
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到Python路径
file_path = Path(__file__).resolve()
project_root = file_path.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.common import load_config
from src.recall import ItemCF, UserCF, HotRecall, NewItemRecall, FollowRecall, DeepWalkRecall, DeepWalkModel
from src.recall.two_tower_model import TwoTowerModel, TwoTowerRecall
from src.ranking.din_model import DIN
from src.rerank.diversity_rerank import DiversityReranker
from src.data.kuairand_loader import (
    load_user_features, 
    load_video_features, 
    get_kuairand_feature_columns,
    load_kuairand_splits
)

class RecommendationService:
    """推荐系统在线服务模拟器"""
    
    def __init__(self):
        self.config = load_config('config/config.yaml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 核心组件
        self.recall_models = {}
        self.ranking_model = None
        self.reranker = None
        
        # 特征缓存
        self.user_features = None
        self.item_features = None
        self.user_history = {} # {uid: [item_id, ...]}
        
        print("="*60)
        print("初始化推荐服务 (KuaiRand-1K)...")
        self.load_data_and_models()
        print("="*60)

    def load_data_and_models(self):
        """加载数据和模型文件"""
        # 1. 加载特征和历史日志 (用于构造 DIN 输入)
        print("正在加载特征数据...")
        self.user_features = load_user_features()
        # 统一 video_id -> item_id
        items = load_video_features().reset_index()
        self.item_features = items.rename(columns={'video_id': 'item_id'})
        
        # 加载历史行为 (这里简单加载训练集中的点击作为历史)
        train_log, _, _ = load_kuairand_splits(use_early_log_as_train=True, verbose=False)
        pos_log = train_log[train_log['is_click'] == 1]
        for uid, group in pos_log.groupby('user_id'):
            self.user_history[int(uid)] = group['item_id'].tolist()[-50:]
        
        # 2. 加载召回模型
        print("加载召回模型...")
        model_map = {
            'item_cf': ('models/itemcf_kuairand.pkl', ItemCF),
            'user_cf': ('models/usercf_kuairand.pkl', UserCF),
            'hot': ('models/hot_kuairand.pkl', HotRecall),
            'new': ('models/new_kuairand.pkl', NewItemRecall),
            'follow': ('models/follow_kuairand.pkl', FollowRecall),
        }
        
        for name, (path, cls) in model_map.items():
            if os.path.exists(path):
                try:
                    self.recall_models[name] = cls.load(path)
                    print(f"  ✓ {name} 加载成功")
                except Exception as e:
                    print(f"  ✗ {name} 加载失败: {e}")
        
        # 特殊处理 DeepWalk
        if os.path.exists('models/deepwalk_kuairand'):
            try:
                dw_m = DeepWalkModel.load('models/deepwalk_kuairand')
                self.recall_models['deepwalk'] = DeepWalkRecall(dw_m)
                print("  ✓ deepwalk 加载成功")
            except: pass

        # 4. 加载双塔召回模型
        print("加载双塔召回模型...")
        tt_path = 'models/two_tower_kuairand_best.pt'
        if os.path.exists(tt_path):
            try:
                from src.data.kuairand_loader import NUM_USERS, NUM_VIDEOS, NUM_ACTIVE_DEGREES, NUM_VIDEO_TYPES
                # 特征列需与训练时保持严格一致
                user_cols = [
                    {'name': 'user_id',          'type': 'categorical', 'vocab_size': NUM_USERS},
                    {'name': 'active_degree',    'type': 'categorical', 'vocab_size': NUM_ACTIVE_DEGREES},
                    {'name': 'is_live_streamer', 'type': 'numerical'},
                    {'name': 'is_video_author',  'type': 'numerical'},
                ]
                item_cols = [
                    {'name': 'item_id',       'type': 'categorical', 'vocab_size': NUM_VIDEOS},
                    {'name': 'video_type_id', 'type': 'categorical', 'vocab_size': NUM_VIDEO_TYPES},
                    {'name': 'tag',           'type': 'categorical', 'vocab_size': 5000},
                    {'name': 'duration_s',    'type': 'numerical'},
                ]
                tt_cfg = self.config.get('recall', {}).get('two_tower', {})
                tt_model = TwoTowerModel(
                    user_feature_columns=user_cols,
                    item_feature_columns=item_cols,
                    embedding_dim=tt_cfg.get('user_emb_dim', 64),
                    user_hidden_units=tt_cfg.get('hidden_units', [256, 128]),
                    item_hidden_units=tt_cfg.get('hidden_units', [256, 128]),
                )
                tt_model.load_state_dict(torch.load(tt_path, map_location=self.device))
                # 注意：self.item_features 已经 rename 过，包含 item_id, tag, duration_s, video_type_id
                self.recall_models['two_tower'] = TwoTowerRecall(tt_model, self.item_features)
                print("  ✓ two_tower 加载成功")
            except Exception as e:
                print(f"  ✗ two_tower 加载失败: {e}")

        # 3. 加载 DIN 精排模型
        print("加载精排模型...")
        din_path = 'models/din_kuairand_best.pt'
        if os.path.exists(din_path):
            try:
                (u_cols, i_cols, c_cols, b_cols) = get_kuairand_feature_columns()
                din_cfg = self.config['ranking']['din']
                self.ranking_model = DIN(
                    user_feature_columns=u_cols,
                    item_feature_columns=i_cols,
                    context_feature_columns=c_cols,
                    behavior_feature_columns=b_cols,
                    embedding_dim=din_cfg.get('embedding_dim', 32)
                )
                self.ranking_model.load_state_dict(torch.load(din_path, map_location=self.device))
                self.ranking_model.to(self.device).eval()
                print("  ✓ DIN 精排模型加载成功")
            except Exception as e:
                print(f"  ✗ DIN 加载失败: {e}")

        # 4. 初始化重排器
        self.reranker = DiversityReranker(self.config.get('reranking', {}))
        print("  ✓ 重排模块初始化完成")

    def multi_recall(self, user_id: int) -> List[int]:
        """执行多路召回"""
        all_candidates = set()
        recall_channels = self.config.get('recall', {}).get('channels', [])
        
        for ch in recall_channels:
            name = ch['name']
            top_k = ch.get('top_k', 50)
            if name in self.recall_models:
                try:
                    if name == 'two_tower':
                        u_feat = self.user_features.loc[user_id] if user_id in self.user_features.index else None
                        recs = self.recall_models[name].recommend(user_id, n=top_k, user_features=u_feat)
                    else:
                        recs = self.recall_models[name].recommend(user_id, n=top_k)
                    all_candidates.update([item[0] for item in recs])
                except Exception as e:
                    print(f"召回异常 [{name}]: {e}")
                    continue
        
        # 兜底：如果没召回够，加点热门
        if len(all_candidates) < 10 and 'hot' in self.recall_models:
            hot_recs = self.recall_models['hot'].recommend(user_id, n=50)
            all_candidates.update([item[0] for item in hot_recs])
            
        return list(all_candidates)

    def ranking(self, user_id: int, item_ids: List[int]) -> List[Tuple[int, float]]:
        """执行 DIN 精排打分"""
        if not self.ranking_model or not item_ids:
            return [(iid, 0.5) for iid in item_ids]
            
        n = len(item_ids)
        u_feat = self.user_features.loc[user_id] if user_id in self.user_features.index else None
        hist_seq = self.user_history.get(user_id, [])
        
        # 构造 Tensor 输入
        inputs = {
            'user_id': torch.full((n,), user_id, dtype=torch.long).to(self.device),
            'item_id': torch.tensor(item_ids, dtype=torch.long).to(self.device),
            'active_degree': torch.full((n,), int(u_feat['active_degree']) if u_feat is not None else 0, dtype=torch.long).to(self.device),
            'hour': torch.full((n,), datetime.now().hour, dtype=torch.long).to(self.device),
            'tab': torch.full((n,), 0, dtype=torch.long).to(self.device),
            'is_live_streamer': torch.full((n,), float(u_feat['is_live_streamer']) if u_feat is not None else 0, dtype=torch.float32).to(self.device),
            'is_video_author': torch.full((n,), float(u_feat['is_video_author']) if u_feat is not None else 0, dtype=torch.float32).to(self.device),
            'follow_user_num': torch.full((n,), float(u_feat['follow_user_num']) if u_feat is not None else 0, dtype=torch.float32).to(self.device),
            'fans_user_num': torch.full((n,), float(u_feat['fans_user_num']) if u_feat is not None else 0, dtype=torch.float32).to(self.device),
            'register_days': torch.full((n,), float(u_feat['register_days']) if u_feat is not None else 0, dtype=torch.float32).to(self.device),
            'seq_length': torch.full((n,), len(hist_seq), dtype=torch.long).to(self.device),
        }
        
        # 序列 Padding
        padded_seq = hist_seq + [0] * (50 - len(hist_seq)) if len(hist_seq) < 50 else hist_seq[-50:]
        inputs['hist_item_seq'] = torch.tensor([padded_seq] * n, dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            scores = self.ranking_model(inputs).squeeze(-1).cpu().numpy()
            
        return sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)

    def get_recommendations(self, user_id: int, n: int = 10) -> List[Dict]:
        """全流程接口"""
        start = datetime.now()
        
        # 1. 召回
        candidates = self.multi_recall(user_id)
        
        # 2. 精排
        ranked_list = self.ranking(user_id, candidates)
        
        # 3. 重排 (MMR)
        # 准备重排所需的特征
        items_for_rerank = [x[0] for x in ranked_list]
        scores_for_rerank = [float(x[1]) for x in ranked_list]
        
        # 我们这里只取前 50 个做重排以保证效率
        top_ranked = ranked_list[:50]
        final_list = self.reranker.rerank(
            items=[x[0] for x in top_ranked],
            scores=[float(x[1]) for x in top_ranked],
            item_features=self.item_features,
            top_k=n
        )
        
        # 4. 组装返回结果
        results = []
        for iid, score in final_list:
            results.append({
                'item_id': int(iid),
                'score': round(float(score), 4),
                'info': self.get_item_info(iid)
            })
            
        cost = (datetime.now() - start).total_seconds() * 1000
        print(f"User {user_id} 推荐完成 | 耗时: {cost:.2f}ms | 召回数: {len(candidates)}")
        
        return results

    def get_item_info(self, item_id: int) -> Dict:
        """获取视频详细信息"""
        if self.item_features is not None and item_id in self.item_features['item_id'].values:
            row = self.item_features[self.item_features['item_id'] == item_id].iloc[0]
            return {
                'tag': int(row['tag']),
                'duration': round(float(row['duration_s']), 1),
                'type': int(row['video_type_id'])
            }
        return {}

if __name__ == "__main__":
    # 模拟线上调用
    service = RecommendationService()
    
    # 为用户 123 推荐
    uid = 123
    print(f"\n正在为用户 {uid} 计算推荐...")
    recs = service.get_recommendations(uid, n=5)
    
    for i, item in enumerate(recs):
        print(f"{i+1}. 视频ID: {item['item_id']:7d} | 得分: {item['score']:.4f} | 标签: {item['info'].get('tag')} | 时长: {item['info'].get('duration')}s")
