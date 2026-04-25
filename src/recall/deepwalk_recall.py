"""
DeepWalk 图召回实现
基于用户-物品交互图，使用随机游走 + Word2Vec 学习节点 Embedding，
再通过向量相似度召回候选物品。

核心思想：
1. 将用户-物品交互建模为二部图 (Bipartite Graph)
2. 在图上进行随机游走，生成节点序列（类比句子中的词序列）
3. 使用 Word2Vec/Skip-Gram 训练节点 Embedding
4. 查询时用用户节点向量最近邻检索物品节点
"""

import numpy as np
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

import pandas as pd


class DeepWalkGraph:
    """
    用户-物品二部图
    节点: user_{user_id} 和 item_{item_id}
    边: 用户交互过的物品
    """

    def __init__(self):
        # 邻接表表示，节点用字符串 "user_1" / "item_100" 区分
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.user_nodes: List[str] = []
        self.item_nodes: List[str] = []

    def build(self, interactions: pd.DataFrame, min_interactions: int = 2):
        """
        根据交互记录构建图
        Args:
            interactions: 包含 user_id, item_id 列的 DataFrame
            min_interactions: 用户/物品最少交互次数（过滤低频节点）
        """
        print("开始构建交互图...")

        # 统计交互频次，过滤低频
        user_counts = interactions['user_id'].value_counts()
        item_counts = interactions['item_id'].value_counts()

        valid_users = set(user_counts[user_counts >= min_interactions].index)
        valid_items = set(item_counts[item_counts >= min_interactions].index)

        edge_count = 0
        for _, row in interactions.iterrows():
            uid = row['user_id']
            iid = row['item_id']
            if uid not in valid_users or iid not in valid_items:
                continue

            u_node = f"user_{uid}"
            i_node = f"item_{iid}"

            # 双向边（随机游走可以在用户和物品之间交替）
            self.graph[u_node].append(i_node)
            self.graph[i_node].append(u_node)
            edge_count += 1

        self.user_nodes = [n for n in self.graph if n.startswith("user_")]
        self.item_nodes = [n for n in self.graph if n.startswith("item_")]

        print(f"图构建完成: {len(self.user_nodes)} 个用户节点, "
              f"{len(self.item_nodes)} 个物品节点, "
              f"{edge_count} 条边")

    def random_walk(self, start_node: str, walk_length: int) -> List[str]:
        """从指定节点开始随机游走"""
        walk = [start_node]

        for _ in range(walk_length - 1):
            current = walk[-1]
            neighbors = self.graph.get(current, [])
            if not neighbors:
                break
            walk.append(random.choice(neighbors))

        return walk

    def generate_walks(self, num_walks: int = 10, walk_length: int = 80) -> List[List[str]]:
        """
        对所有节点生成随机游走序列
        Args:
            num_walks: 每个节点游走次数
            walk_length: 每次游走长度
        """
        all_nodes = list(self.graph.keys())
        all_walks = []

        print(f"生成随机游走序列 (节点数={len(all_nodes)}, "
              f"每节点游走{num_walks}次, 每次长度{walk_length})...")

        for walk_idx in range(num_walks):
            random.shuffle(all_nodes)
            for node in all_nodes:
                walk = self.random_walk(node, walk_length)
                all_walks.append(walk)

            if (walk_idx + 1) % 2 == 0:
                print(f"  已完成 {walk_idx + 1}/{num_walks} 轮游走")

        print(f"共生成 {len(all_walks)} 条游走序列")
        return all_walks


class DeepWalkModel:
    """
    DeepWalk 模型
    使用 Word2Vec 的 Skip-Gram 训练节点 Embedding
    """

    def __init__(self,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 window_size: int = 5,
                 workers: int = 4,
                 min_count: int = 1):
        """
        Args:
            dimensions: Embedding 维度
            walk_length: 随机游走长度
            num_walks: 每个节点游走次数
            window_size: Word2Vec 窗口大小
            workers: 训练并发线程数
            min_count: 最少出现次数（过滤低频节点）
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.workers = workers
        self.min_count = min_count

        self.graph: Optional[DeepWalkGraph] = None
        self.model: Optional[object] = None  # Word2Vec model

    def fit(self, interactions: pd.DataFrame, min_interactions: int = 2):
        """训练 DeepWalk 模型"""
        if not GENSIM_AVAILABLE:
            print("警告: gensim 未安装，使用随机 Embedding 作为降级策略")
            print("安装命令: pip install gensim")
            self._fit_fallback(interactions)
            return

        print("=" * 50)
        print("开始训练 DeepWalk 图召回模型...")

        # 1. 构建图
        self.graph = DeepWalkGraph()
        self.graph.build(interactions, min_interactions=min_interactions)

        # 2. 生成随机游走序列
        walks = self.graph.generate_walks(
            num_walks=self.num_walks,
            walk_length=self.walk_length
        )

        # 3. 训练 Word2Vec
        print("训练 Word2Vec Embedding...")
        self.model = Word2Vec(
            sentences=walks,
            vector_size=self.dimensions,
            window=self.window_size,
            min_count=self.min_count,
            workers=self.workers,
            sg=1,       # Skip-Gram
            epochs=5
        )

        print(f"DeepWalk 训练完成! 词表大小: {len(self.model.wv)}")

    def _fit_fallback(self, interactions: pd.DataFrame):
        """gensim 不可用时的降级：使用随机 Embedding"""
        self.graph = DeepWalkGraph()
        self.graph.build(interactions)
        # 随机初始化 embedding 字典
        all_nodes = list(self.graph.graph.keys())
        self._fallback_embeddings = {
            node: np.random.normal(0, 0.1, self.dimensions)
            for node in all_nodes
        }
        self.model = None

    def get_embedding(self, node: str) -> Optional[np.ndarray]:
        """获取节点 Embedding 向量"""
        if self.model is not None:
            try:
                return self.model.wv[node]
            except KeyError:
                return None
        elif hasattr(self, '_fallback_embeddings'):
            return self._fallback_embeddings.get(node)
        return None

    def save(self, filepath: str):
        """保存模型"""
        save_data = {
            'dimensions': self.dimensions,
            'walk_length': self.walk_length,
            'num_walks': self.num_walks,
            'window_size': self.window_size,
            'graph': self.graph,
        }
        if self.model is not None and GENSIM_AVAILABLE:
            self.model.save(filepath + ".w2v")
            save_data['has_w2v'] = True
        elif hasattr(self, '_fallback_embeddings'):
            save_data['fallback_embeddings'] = self._fallback_embeddings
            save_data['has_w2v'] = False

        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"DeepWalk 模型已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'DeepWalkModel':
        """加载模型"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)

        model = cls(
            dimensions=save_data['dimensions'],
            walk_length=save_data['walk_length'],
            num_walks=save_data['num_walks'],
            window_size=save_data['window_size'],
        )
        model.graph = save_data['graph']

        if save_data.get('has_w2v') and GENSIM_AVAILABLE:
            model.model = Word2Vec.load(filepath + ".w2v")
        elif 'fallback_embeddings' in save_data:
            model._fallback_embeddings = save_data['fallback_embeddings']

        print(f"DeepWalk 模型已从 {filepath} 加载")
        return model


class DeepWalkRecall:
    """
    DeepWalk 图召回器
    使用训练好的节点 Embedding 进行 ANN（近似最近邻）检索
    """

    def __init__(self, deepwalk_model: DeepWalkModel):
        self.dw_model = deepwalk_model
        # 预计算所有物品 Embedding，用于快速检索
        self._item_embeddings: Dict[int, np.ndarray] = {}
        self._item_ids: List[int] = []
        self._item_matrix: Optional[np.ndarray] = None
        self._build_item_index()

    def _build_item_index(self):
        """预构建物品向量索引"""
        if self.dw_model.graph is None:
            return

        print("构建物品向量索引...")
        valid_items = []
        valid_embs = []

        for node in self.dw_model.graph.item_nodes:
            emb = self.dw_model.get_embedding(node)
            if emb is not None:
                item_id = int(node.split("_")[1])
                valid_items.append(item_id)
                valid_embs.append(emb)

        if valid_embs:
            self._item_ids = valid_items
            # 归一化，方便内积 = 余弦相似度
            matrix = np.array(valid_embs, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._item_matrix = matrix / norms
            print(f"物品向量索引构建完成: {len(self._item_ids)} 个物品")

    def recommend(self, user_id: int, n: int = 80) -> List[Tuple[int, float]]:
        """
        为用户推荐物品
        Args:
            user_id: 用户ID
            n: 推荐数量
        Returns:
            [(item_id, score), ...]
        """
        if self._item_matrix is None or len(self._item_ids) == 0:
            return []

        # 获取用户向量
        user_node = f"user_{user_id}"
        user_emb = self.dw_model.get_embedding(user_node)

        if user_emb is None:
            return []

        # 归一化
        norm = np.linalg.norm(user_emb)
        if norm == 0:
            return []
        user_emb = user_emb / norm

        # 计算余弦相似度（内积）
        scores = self._item_matrix @ user_emb  # (n_items,)

        # 取 top-n
        top_n = min(n, len(self._item_ids))
        top_indices = np.argsort(scores)[-top_n:][::-1]

        results = [
            (self._item_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def batch_recommend(self,
                        user_ids: List[int],
                        n: int = 80) -> Dict[int, List[Tuple[int, float]]]:
        """批量推荐"""
        return {uid: self.recommend(uid, n) for uid in user_ids}
