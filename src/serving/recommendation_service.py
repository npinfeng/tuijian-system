"""
推荐服务主程序
整合召回、排序、重排等模块，纯Python本地调用的形式
"""

from typing import List, Dict, Optional
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import random
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.recall.collaborative_filtering import ItemCF, UserCF
    from src.recall.two_tower_model import TwoTowerModel
    from src.ranking.din_model import DIN
except ImportError as e:
    print(f"Warning: Failed to import some models (possibly missing tensorflow): {e}")
    # Provide dummy classes for type hinting if needed, though they aren't strictly required
    ItemCF = None
    UserCF = None
    TwoTowerModel = None
    DIN = None

from src.recall.collaborative_filtering import ItemCF, UserCF
from src.rerank.diversity_rerank import DiversityReranker

from src.features.feature_engineering import OfflineFeatureStore
from src.utils.common import load_config


class RecommendationService:
    """推荐服务类，整合召回、排序、重排等模块"""
    
    def __init__(self):
        self.config = load_config('config/config.yaml')
        
        # 召回模型
        self.itemcf_model = None
        self.usercf_model = None
        self.two_tower_model = None
        
        # 排序模型
        self.ranking_model = None
        
        # 重排器
        self.reranker = DiversityReranker(self.config.get('reranking', {}))
        
        # 离线批量特征存储查询接口
        self.feature_store = OfflineFeatureStore(self.config.get('features', {}))
        
        # 加载模型
        self.load_models()
        self._hot_items_cache = list(range(1, 101))
    
    def load_models(self):
        """加载所有模型"""
        print("开始加载模型...")
        try:
            self.itemcf_model = ItemCF.load('models/itemcf_model.pkl')
            print("ItemCF模型加载成功")
        except Exception as e:
            print(f"ItemCF模型未找到或加载失败 (文件可能不存在): {e}")
            
        print("UserCF和深度学习模型加载跳过（demo）")
        print("模型加载完成！")

    def recommend(self, user_id: int, scene: str = "feed", num: int = 10, context: Optional[Dict] = None) -> Dict:
        """
        推荐主接口
        """
        start_time = datetime.now()
        trace_id = f"{user_id}_{int(start_time.timestamp() * 1000)}"
        
        try:
            # 1. 多路召回
            recall_results = self.multi_recall(
                user_id=user_id,
                num=self.config.get('recall', {}).get('total_recall_num', 500)
            )
            
            if not recall_results:
                # 降级：返回热门推荐
                recall_results = self.get_hot_items(num)
            
            # 3. 精排
            ranking_results = self.ranking(
                user_id=user_id,
                candidates=recall_results,
                context=context
            )
            
            # 4. 重排
            final_results = self.rerank(
                user_id=user_id,
                candidates=ranking_results,
                top_k=num
            )
            
            # 5. 构造返回结果
            items = []
            for item_id, score in final_results:
                items.append({
                    'item_id': item_id,
                    'score': float(score),
                    'reason': 'personalized'
                })
            
            cost_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            return {
                "user_id": user_id,
                "items": items,
                "trace_id": trace_id,
                "cost_time": cost_time
            }
        
        except Exception as e:
            print(f"推荐失败: {e}")
            return {"error": str(e)}

    def multi_recall(self, user_id: int, num: int = 500) -> List[tuple]:
        """多路召回"""
        all_candidates = []
        
        # 1. ItemCF召回
        if self.itemcf_model:
            try:
                itemcf_results = self.itemcf_model.recommend(user_id=user_id, n=100)
                weight = 0.15
                itemcf_results = [(item_id, score * weight) for item_id, score in itemcf_results]
                all_candidates.extend(itemcf_results)
            except Exception as e:
                print(f"ItemCF召回警告: {e}")
        
        # 2. UserCF召回 (如果存在)
        if self.usercf_model:
            try:
                usercf_results = self.usercf_model.recommend(user_id=user_id, n=80)
                weight = 0.10
                usercf_results = [(item_id, score * weight) for item_id, score in usercf_results]
                all_candidates.extend(usercf_results)
            except Exception:
                pass
        
        # 热门召回（保底）
        hot_items = self.get_hot_items(50)
        all_candidates.extend(hot_items)
        
        # 合并去重，按分数排序
        item_scores = {}
        for item_id, score in all_candidates:
            item_scores[item_id] = item_scores.get(item_id, 0) + score
        
        # 排序
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:num]

    def ranking(self, user_id: int, candidates: List[tuple], context: Optional[Dict] = None) -> List[tuple]:
        """精排"""
        if not candidates:
            return []
        return candidates

    def rerank(self, user_id: int, candidates: List[tuple], top_k: int = 10) -> List[tuple]:
        """重排"""
        if not candidates:
            return []
        
        # 模拟物品特征数据
        items = []
        scores = []
        item_features = []
        now = datetime.now()
        
        for item_id, score in candidates:
            items.append(item_id)
            scores.append(score)
            item_features.append({
                'item_id': item_id,
                'category': f'category_{item_id % 5}',
                'author_id': f'author_{item_id % 20}',
                'publish_time': now
            })
            
        reranked = self.reranker.rerank(
            items=items,
            scores=scores,
            item_features=item_features,
            top_k=top_k
        )
        return reranked

    def get_hot_items(self, num: int = 50) -> List[tuple]:
        """获取热门物品（降级策略）"""
        items = random.sample(self._hot_items_cache, min(num, len(self._hot_items_cache)))
        return [(item_id, 1.0 / (i + 1)) for i, item_id in enumerate(items)]

    def feedback(self, user_id: int, item_id: int, action: str):
        """用户反馈接口，收集用户行为数据，用于实时特征更新"""
        try:
            if action == 'click':
                self.feature_store.update_item_ctr(item_id, 1)
            elif action == 'impression':
                self.feature_store.update_item_ctr(item_id, 0)
            return {"status": "success", "message": "反馈已记录"}
        except Exception as e:
            print(f"反馈记录失败: {e}")
            return {"status": "error", "message": str(e)}

    def stats(self):
        """系统统计信息"""
        return {
            "total_users": "5000000+",
            "total_items": "1000000+",
            "models": {
                "itemcf": self.itemcf_model is not None,
                "usercf": self.usercf_model is not None,
                "two_tower": self.two_tower_model is not None,
                "ranking": self.ranking_model is not None,
            }
        }


if __name__ == "__main__":
    # 使用示例
    print("="*50)
    print("推荐系统本地测试运行")
    print("="*50)
    
    # 初始化服务
    service = RecommendationService()
    
    user_id = 1001
    
    # 模拟获取推荐
    print(f"\n获取用户 {user_id} 的推荐结果:")
    res = service.recommend(user_id=user_id, num=5)
    print(yaml.dump(res, allow_unicode=True))
    
    # 模拟用户反馈
    if "items" in res and res["items"]:
        first_item = res['items'][0]['item_id']
        print(f"\n记录用户 {user_id} 点击 物品 {first_item} 的行为:")
        service.feedback(user_id=user_id, item_id=first_item, action='click')
    
    # 查看统计
    print("\n系统状态:")
    print(yaml.dump(service.stats(), allow_unicode=True))
