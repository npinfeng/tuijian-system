"""
推荐服务主程序
整合召回、排序、重排等模块，纯Python本地调用的形式
"""

from typing import List, Dict, Optional
import yaml
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.recall.collaborative_filtering import ItemCF, UserCF
    from src.recall.hot_recall import HotRecall, NewItemRecall
    from src.recall.follow_recall import FollowRecall
    from src.recall.deepwalk_recall import DeepWalkModel, DeepWalkRecall
    from src.recall.two_tower_model import TwoTowerModel
    from src.ranking.din_model import DIN
    from src.data.kuairand_loader import load_user_features, load_video_features, get_kuairand_feature_columns, NUM_VIDEOS, NUM_USERS
except ImportError as e:
    print(f"Warning: Failed to import some modules: {e}")

from src.rerank.diversity_rerank import DiversityReranker
from src.utils.common import load_config
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import random


class RecommendationService:
    """推荐服务类，整合 7 路召回、排序、重排等模块"""
    
    def __init__(self):
        self.config = load_config('config/config.yaml')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 召回模型字典
        self.recall_models = {}
        
        # 排序模型
        self.ranking_model = None
        
        # 数据缓存
        self.user_features = None
        self.item_features = None
        self._hot_items_cache = []
        
        # 重排器
        self.reranker = DiversityReranker(self.config.get('reranking', {}))
        
        # 加载数据和模型
        self.load_data()
        self.load_models()
    
    def load_data(self):
        """加载 KuaiRand-1K 特征数据"""
        print("正在加载 KuaiRand-1K 特征数据...")
        try:
            self.user_features = load_user_features()
            self.item_features = load_video_features(sample_n=200000)
            self._hot_items_cache = self.item_features.index.tolist()[:1000]
            print(f"特征数据加载成功: {len(self.user_features)} 用户, {len(self.item_features)} 视频")
        except Exception as e:
            print(f"特征数据加载失败: {e}")

    def load_models(self):
        """加载所有 7 路召回模型和 1 路排序模型"""
        print("开始加载模型...")
        
        # 1. ItemCF
        try:
            self.recall_models['item_cf'] = ItemCF.load('models/itemcf_kuairand.pkl')
            print("  [1/7] ItemCF 加载成功")
        except Exception: pass

        # 2. UserCF
        try:
            self.recall_models['user_cf'] = UserCF.load('models/usercf_kuairand.pkl')
            print("  [2/7] UserCF 加载成功")
        except Exception: pass

        # 3. Hot Recall
        try:
            self.recall_models['hot'] = HotRecall.load('models/hot_kuairand.pkl')
            print("  [3/7] HotRecall 加载成功")
        except Exception: pass

        # 4. New Item Recall
        try:
            self.recall_models['new'] = NewItemRecall.load('models/new_kuairand.pkl')
            print("  [4/7] NewItemRecall 加载成功")
        except Exception: pass

        # 5. Follow Recall
        try:
            self.recall_models['follow'] = FollowRecall.load('models/follow_kuairand.pkl')
            print("  [5/7] FollowRecall 加载成功")
        except Exception: pass

        # 6. DeepWalk
        try:
            dw_m = DeepWalkModel.load('models/deepwalk_kuairand')
            self.recall_models['deepwalk'] = DeepWalkRecall(dw_m)
            print("  [6/7] DeepWalkRecall 加载成功")
        except Exception: pass

        # 7. TwoTower
        try:
            # 双塔通常做向量检索，这里简化演示
            print("  [7/7] TwoTower 加载跳过 (需向量数据库配合)")
        except Exception: pass

        # 9. DIN Ranking
        try:
            din_config = self.config['ranking']['din']
            feat_cols = get_kuairand_feature_columns()
            self.ranking_model = DIN(
                user_feature_columns=feat_cols[0], item_feature_columns=feat_cols[1],
                context_feature_columns=feat_cols[2], behavior_feature_columns=feat_cols[3],
                embedding_dim=din_config.get('embedding_dim', 32),
                attention_hidden_units=din_config.get('attention_hidden_units', [80, 40]),
                dnn_hidden_units=din_config.get('dnn_hidden_units', [256, 128, 64])
            )
            model_path = 'models/din_kuairand_best.pt'
            if Path(model_path).exists():
                self.ranking_model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.ranking_model.to(self.device).eval()
                print("DIN精排模型加载成功")
        except Exception as e:
            print(f"DIN模型加载失败: {e}")
            
        print("模型加载完成！")

    def recommend(self, user_id: int, scene: str = "feed", num: int = 10, context: Optional[Dict] = None) -> Dict:
        """推荐主接口"""
        start_time = datetime.now()
        trace_id = f"{user_id}_{int(start_time.timestamp() * 1000)}"
        
        try:
            # 1. 多路召回
            recall_results = self.multi_recall(user_id=user_id)
            
            if not recall_results:
                recall_results = self.get_hot_items(num * 2)
            
            # 2. 精排
            ranking_results = self.ranking(user_id=user_id, candidates=recall_results, context=context)
            
            # 3. 重排 (Top-K)
            final_results = ranking_results[:num]
            
            # 4. 构造返回结果
            items = []
            for item_id, score in final_results:
                items.append({
                    'item_id': int(item_id),
                    'score': float(score),
                    'reason': 'multi_channel_recall'
                })
            
            cost_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return {
                "user_id": user_id,
                "items": items,
                "trace_id": trace_id,
                "cost_time": cost_time,
                "recall_count": len(recall_results)
            }
        except Exception as e:
            print(f"推荐异常: {e}")
            return {"error": str(e)}

    def multi_recall(self, user_id: int) -> List[tuple]:
        """多路召回合并"""
        all_candidates = []
        recall_config = {c['name']: c for c in self.config.get('recall', {}).get('channels', [])}
        
        for name, model in self.recall_models.items():
            try:
                conf = recall_config.get(name, {'weight': 1.0, 'top_k': 50})
                res = model.recommend(user_id=user_id, n=conf['top_k'])
                # 加权融合
                weighted_res = [(item_id, score * conf['weight']) for item_id, score in res]
                all_candidates.extend(weighted_res)
            except Exception as e:
                print(f"{name} 召回失败: {e}")
        
        # 合并去重
        item_scores = {}
        for item_id, score in all_candidates:
            item_scores[item_id] = max(item_scores.get(item_id, 0), score)
        
        return sorted(item_scores.items(), key=lambda x: x[1], reverse=True)

    def ranking(self, user_id: int, candidates: List[tuple], context: Optional[Dict] = None) -> List[tuple]:
        """DIN 精排"""
        if not self.ranking_model or not candidates or self.user_features is None:
            return sorted(candidates, key=lambda x: x[1], reverse=True)
        
        try:
            item_ids = [c[0] for c in candidates]
            n = len(item_ids)
            u_feat = self.user_features.loc[user_id] if user_id in self.user_features.index else None
            
            inputs = {
                'user_id': torch.full((n,), user_id, dtype=torch.long),
                'item_id': torch.tensor(item_ids, dtype=torch.long),
                'active_degree': torch.full((n,), int(u_feat['active_degree']) if u_feat is not None else 0, dtype=torch.long),
                'is_live_streamer': torch.full((n,), float(u_feat['is_live_streamer']) if u_feat is not None else 0.0, dtype=torch.float32),
                'is_video_author': torch.full((n,), float(u_feat['is_video_author']) if u_feat is not None else 0.0, dtype=torch.float32),
                'follow_user_num': torch.full((n,), float(u_feat['follow_user_num']) if u_feat is not None else 0.0, dtype=torch.float32),
                'fans_user_num': torch.full((n,), float(u_feat['fans_user_num']) if u_feat is not None else 0.0, dtype=torch.float32),
                'register_days': torch.full((n,), float(u_feat['register_days']) if u_feat is not None else 0.0, dtype=torch.float32),
                'hour': torch.full((n,), datetime.now().hour, dtype=torch.long),
                'tab': torch.full((n,), 0, dtype=torch.long),
                'hist_item_seq': torch.zeros((n, 50), dtype=torch.long),
                'seq_length': torch.zeros((n,), dtype=torch.long),
            }
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                scores = self.ranking_model(inputs).squeeze(-1).cpu().numpy()
            return sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)
        except Exception as e:
            print(f"精排降级: {e}")
            return sorted(candidates, key=lambda x: x[1], reverse=True)

    def get_hot_items(self, num: int = 50) -> List[tuple]:
        """热门兜底"""
        if not self._hot_items_cache: return []
        items = random.sample(self._hot_items_cache, min(num, len(self._hot_items_cache)))
        return [(item_id, 1.0) for item_id in items]

    def stats(self):
        """系统统计信息"""
        return {
            "dataset": "KuaiRand-1K",
            "models": {
                "recall_channels": list(self.recall_models.keys()),
                "ranking_din": self.ranking_model is not None,
            }
        }


if __name__ == "__main__":
    print("="*50)
    print("推荐系统 KuaiRand-1K 测试")
    print("="*50)
    
    service = RecommendationService()
    test_user_id = 0 # KuaiRand-1K 有 0~999 用户
    
    print(f"\n获取用户 {test_user_id} 的推荐结果:")
    res = service.recommend(user_id=test_user_id, num=5)
    print(res)
