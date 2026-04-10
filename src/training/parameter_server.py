"""
简单的离线 Parameter Server (PS) 实现
用于在单机模拟大规模推荐系统离线训练中 Embedding 的分布式存储和更新。
核心逻辑：大宽表和海量 Embedding 存放于 PS 节点，离线 Worker 节点训练时执行 Pull (前向传播) 与 Push (反向传播)。
"""

import numpy as np
from multiprocessing import Manager
from typing import List, Dict

class ParameterServer:
    """模拟 Parameter Server 架构"""
    
    def __init__(self, embedding_dim: int = 16):
        # 使用多进程安全的 dict 来模拟独立的 PS 内存存储
        self.manager = Manager()
        self.user_embeddings = self.manager.dict()
        self.item_embeddings = self.manager.dict()
        self.embedding_dim = embedding_dim
        
    def _init_embedding(self) -> np.ndarray:
        """随机初始化 Embedding"""
        return np.random.normal(0, 0.01, self.embedding_dim)

    def pull_user_embedding(self, user_ids: List[int]) -> Dict[int, np.ndarray]:
        """Worker 拉取：PULL 操作，获取用户 Embedding 进行前向传播"""
        result = {}
        for uid in user_ids:
            if uid not in self.user_embeddings:
                self.user_embeddings[uid] = self._init_embedding()
            result[uid] = self.user_embeddings[uid]
        return result

    def push_user_gradient(self, user_grads: Dict[int, np.ndarray], learning_rate: float = 0.01):
        """Worker 推送：PUSH 操作，利用梯度更新用户 Embedding"""
        for uid, grad in user_grads.items():
            if uid in self.user_embeddings:
                current_emb = self.user_embeddings[uid]
                # SGD 更新
                self.user_embeddings[uid] = current_emb - learning_rate * grad

    def pull_item_embedding(self, item_ids: List[int]) -> Dict[int, np.ndarray]:
        """Worker 拉取：PULL 操作，获取物品 Embedding 进行前向传播"""
        result = {}
        for iid in item_ids:
            if iid not in self.item_embeddings:
                self.item_embeddings[iid] = self._init_embedding()
            result[iid] = self.item_embeddings[iid]
        return result

    def push_item_gradient(self, item_grads: Dict[int, np.ndarray], learning_rate: float = 0.01):
        """Worker 推送：PUSH 操作，利用梯度更新物品 Embedding"""
        for iid, grad in item_grads.items():
            if iid in self.item_embeddings:
                current_emb = self.item_embeddings[iid]
                self.item_embeddings[iid] = current_emb - learning_rate * grad

    def save_offline_embeddings(self, filepath: str):
        """离线训练结束后落盘，供各种离线打分任务使用"""
        import pickle
        data_to_save = {
            'user': dict(self.user_embeddings),
            'item': dict(self.item_embeddings)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"离线嵌入向量成功落盘保存到 {filepath}")

if __name__ == "__main__":
    # 使用示例：模拟一次完整的 Pull / Push 离线训练网络调用
    print("初始化 Parameter Server...")
    ps = ParameterServer(embedding_dim=8)
    
    # 离线数据 Batch 中出现了 user 1 和 item 100
    batch_users = [1, 2]
    batch_items = [100, 101]
    
    # 1. Pull / 前向获取
    print(f"PULL: 正在拉取 users {batch_users} 和 items {batch_items} 的 Embedding...")
    user_embs = ps.pull_user_embedding(batch_users)
    item_embs = ps.pull_item_embedding(batch_items)
    print("User 1 Embedding:", user_embs[1])
    
    # 2. 模拟计算后得出梯度 (假设都是长度为8的简单向量)
    mock_grads = {1: np.ones(8) * 0.1, 2: np.ones(8) * 0.1}
    
    # 3. Push / 梯度反传与更新
    print("PUSH: 正在向 PS 节点推送更新梯度...")
    ps.push_user_gradient(mock_grads, learning_rate=0.1)
    
    # 验证更新
    updated_user_embs = ps.pull_user_embedding([1])
    print("User 1 Updated Embedding:", updated_user_embs[1])
    print("Parameter Server 原型测试通过！")
