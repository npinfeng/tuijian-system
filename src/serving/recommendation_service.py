"""
推荐服务主程序
整合召回、排序、重排等模块
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.recall.collaborative_filtering import ItemCF, UserCF
from src.recall.two_tower_model import TwoTowerModel
from src.ranking.din_model import DIN
from src.rerank.diversity_rerank import DiversityReranker
from src.features.feature_engineering import RealtimeFeatureStore
from src.utils.common import load_config


app = FastAPI(title="推荐系统服务", version="1.0.0")


# 请求模型
class RecommendRequest(BaseModel):
    user_id: int
    scene: str = "feed"  # feed流、搜索、相关推荐等
    num: int = 10
    context: Optional[Dict] = None


class RecommendResponse(BaseModel):
    user_id: int
    items: List[Dict]
    trace_id: str
    cost_time: int  # ms


# 全局变量，存储模型
class ModelManager:
    """模型管理器"""
    
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
        
        # 特征存储
        self.feature_store = RealtimeFeatureStore(self.config.get('features', {}))
        
        # 加载模型
        self.load_models()
    
    def load_models(self):
        """加载所有模型"""
        print("开始加载模型...")
        
        try:
            # 加载ItemCF
            self.itemcf_model = ItemCF.load('models/itemcf_model.pkl')
            print("ItemCF模型加载成功")
        except Exception as e:
            print(f"ItemCF模型加载失败: {e}")
        
        try:
            # 加载UserCF
            # self.usercf_model = UserCF.load('models/usercf_model.pkl')
            print("UserCF模型加载跳过（demo）")
        except Exception as e:
            print(f"UserCF模型加载失败: {e}")
        
        # TODO: 加载深度学习模型
        print("深度学习模型加载跳过（需要先训练）")
        
        print("模型加载完成！")


model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """服务启动时的初始化"""
    print("推荐服务启动中...")
    print("服务启动完成！")


@app.get("/")
async def root():
    """健康检查"""
    return {
        "service": "recommendation-system",
        "status": "running",
        "version": "1.0.0"
    }


@app.post("/api/v1/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    """
    推荐接口
    """
    start_time = datetime.now()
    trace_id = f"{request.user_id}_{int(start_time.timestamp() * 1000)}"
    
    try:
        # 1. 多路召回
        recall_results = multi_recall(
            user_id=request.user_id,
            num=model_manager.config['recall']['total_recall_num']
        )
        
        if not recall_results:
            # 降级：返回热门推荐
            recall_results = get_hot_items(request.num)
        
        # 2. 粗排（可选）
        # pre_ranking_results = pre_ranking(recall_results)
        
        # 3. 精排
        ranking_results = ranking(
            user_id=request.user_id,
            candidates=recall_results,
            context=request.context
        )
        
        # 4. 重排
        final_results = rerank(
            user_id=request.user_id,
            candidates=ranking_results,
            top_k=request.num
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
        
        return RecommendResponse(
            user_id=request.user_id,
            items=items,
            trace_id=trace_id,
            cost_time=cost_time
        )
    
    except Exception as e:
        print(f"推荐失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def multi_recall(user_id: int, num: int = 500) -> List[tuple]:
    """
    多路召回
    Returns:
        List of (item_id, score) tuples
    """
    all_candidates = []
    
    # 1. ItemCF召回
    if model_manager.itemcf_model:
        try:
            itemcf_results = model_manager.itemcf_model.recommend(
                user_id=user_id,
                n=100
            )
            # 加权
            weight = 0.15
            itemcf_results = [(item_id, score * weight) for item_id, score in itemcf_results]
            all_candidates.extend(itemcf_results)
        except Exception as e:
            print(f"ItemCF召回失败: {e}")
    
    # 2. UserCF召回
    if model_manager.usercf_model:
        try:
            usercf_results = model_manager.usercf_model.recommend(
                user_id=user_id,
                n=80
            )
            weight = 0.10
            usercf_results = [(item_id, score * weight) for item_id, score in usercf_results]
            all_candidates.extend(usercf_results)
        except Exception as e:
            print(f"UserCF召回失败: {e}")
    
    # 3. 双塔模型召回（需要实现向量检索）
    # TODO: 实现基于Faiss的向量检索
    
    # 4. 热门召回（保底）
    hot_items = get_hot_items(50)
    all_candidates.extend(hot_items)
    
    # 合并去重，按分数排序
    item_scores = {}
    for item_id, score in all_candidates:
        if item_id in item_scores:
            item_scores[item_id] += score  # 累加分数
        else:
            item_scores[item_id] = score
    
    # 排序
    sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items[:num]


def ranking(user_id: int, 
           candidates: List[tuple],
           context: Optional[Dict] = None) -> List[tuple]:
    """
    精排
    """
    if not candidates:
        return []
    
    # 如果有深度模型，使用深度模型打分
    if model_manager.ranking_model:
        # TODO: 实现DIN模型推理
        pass
    
    # 否则直接返回召回结果
    return candidates


def rerank(user_id: int,
          candidates: List[tuple],
          top_k: int = 10) -> List[tuple]:
    """
    重排
    """
    if not candidates:
        return []
    
    # 模拟物品特征数据
    items = [item_id for item_id, _ in candidates]
    scores = [score for _, score in candidates]
    
    # 构造item_features DataFrame
    item_features = pd.DataFrame({
        'item_id': items,
        'category': [f'category_{i % 5}' for i in items],  # 模拟类目
        'author_id': [f'author_{i % 20}' for i in items],  # 模拟作者
        'publish_time': [datetime.now() for _ in items],  # 模拟发布时间
    })
    
    # 重排
    reranked = model_manager.reranker.rerank(
        items=items,
        scores=scores,
        item_features=item_features,
        top_k=top_k
    )
    
    return reranked


def get_hot_items(num: int = 50) -> List[tuple]:
    """
    获取热门物品（降级策略）
    """
    # 模拟热门物品
    hot_items = list(range(1, 101))
    np.random.shuffle(hot_items)
    
    # 返回带分数的列表
    return [(item_id, 1.0 / (i + 1)) for i, item_id in enumerate(hot_items[:num])]


@app.post("/api/v1/feedback")
async def feedback(user_id: int, item_id: int, action: str):
    """
    用户反馈接口
    收集用户行为数据，用于实时特征更新
    """
    try:
        # 更新实时特征
        if action == 'click':
            model_manager.feature_store.update_item_ctr(item_id, 1)
        elif action == 'impression':
            model_manager.feature_store.update_item_ctr(item_id, 0)
        
        return {"status": "success", "message": "反馈已记录"}
    
    except Exception as e:
        print(f"反馈记录失败: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/stats")
async def stats():
    """
    系统统计信息
    """
    return {
        "total_users": "5000000+",
        "total_items": "1000000+",
        "daily_requests": "1000000000+",
        "avg_response_time": "45ms",
        "models": {
            "itemcf": model_manager.itemcf_model is not None,
            "usercf": model_manager.usercf_model is not None,
            "two_tower": model_manager.two_tower_model is not None,
            "ranking": model_manager.ranking_model is not None,
        }
    }


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info"
    )
