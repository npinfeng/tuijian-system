"""
MIND (Multi-Interest Network with Dynamic routing) 多兴趣召回实现
论文: CIKM 2019 - Alibaba

核心思想:
  用户兴趣是多峰的，MIND 通过胶囊网络（动态路由）为每个用户提取 K 个兴趣向量，
  每个兴趣向量分别召回，最终合并去重。
"""

import numpy as np
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


class CapsuleLayer(nn.Module):
    """胶囊网络动态路由层：将用户历史行为 Embedding 压缩为 K 个兴趣胶囊向量"""

    def __init__(self, input_dim: int, num_capsules: int = 4,
                 capsule_dim: int = 64, num_routing: int = 3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
        self.W = nn.Linear(input_dim, num_capsules * capsule_dim, bias=False)

    def forward(self, x: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
            mask: (batch, seq_len), True 为有效位置
        Returns:
            (batch, num_capsules, capsule_dim)
        """
        batch_size, seq_len, _ = x.shape
        u = self.W(x).view(batch_size, seq_len, self.num_capsules, self.capsule_dim)
        b = torch.zeros(batch_size, seq_len, self.num_capsules, device=x.device)

        for _ in range(self.num_routing):
            if mask is not None:
                mask_exp = mask.unsqueeze(-1)
                b_masked = b.masked_fill(~mask_exp, float('-inf'))
                c = F.softmax(b_masked, dim=2) * mask_exp.float()
            else:
                c = F.softmax(b, dim=2)

            s = (c.unsqueeze(-1) * u).sum(dim=1)  # (batch, num_capsules, capsule_dim)
            v = self._squash(s)
            b = b + (u * v.unsqueeze(1)).sum(dim=-1)

        return v

    def _squash(self, s: torch.Tensor) -> torch.Tensor:
        norm_sq = (s ** 2).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(norm_sq + 1e-8)
        return (norm_sq / (1 + norm_sq)) * s / norm


class MINDModel(nn.Module):
    """MIND 多兴趣网络"""

    def __init__(self, num_items: int, embedding_dim: int = 64,
                 num_interests: int = 4, num_routing: int = 3):
        super(MINDModel, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_interests = num_interests

        self.item_embedding = nn.Embedding(num_items + 1, embedding_dim, padding_idx=0)
        self.capsule_layer = CapsuleLayer(
            input_dim=embedding_dim,
            num_capsules=num_interests,
            capsule_dim=embedding_dim,
            num_routing=num_routing
        )

    def forward(self, hist_items: torch.Tensor, hist_mask: torch.Tensor,
                target_items: Optional[torch.Tensor] = None):
        """
        Args:
            hist_items: (batch, seq_len) 历史物品 ID
            hist_mask: (batch, seq_len) 有效位置 mask
            target_items: (batch,) 目标物品 ID [训练时]
        Returns:
            interest_vectors: (batch, num_interests, embedding_dim)
            scores: (batch, num_interests) 或 None
        """
        hist_emb = self.item_embedding(hist_items)
        interest_vecs = self.capsule_layer(hist_emb, mask=hist_mask)
        interest_vecs = F.normalize(interest_vecs, p=2, dim=-1)

        if target_items is not None:
            target_emb = F.normalize(self.item_embedding(target_items), p=2, dim=-1)
            scores = (interest_vecs * target_emb.unsqueeze(1)).sum(dim=-1)
            return interest_vecs, scores

        return interest_vecs, None

    def get_user_interests(self, hist_items: torch.Tensor,
                           hist_mask: torch.Tensor) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            vecs, _ = self.forward(hist_items, hist_mask)
        return vecs.cpu().numpy()


class MINDTrainer:
    """MIND 模型训练器"""

    def __init__(self, model: MINDModel, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def _build_sequences(self, interactions: pd.DataFrame,
                         max_seq_len: int) -> Dict[int, List[int]]:
        if 'timestamp' in interactions.columns:
            interactions = interactions.sort_values('timestamp')
        user_seqs: Dict[int, List[int]] = defaultdict(list)
        for _, row in interactions.iterrows():
            user_seqs[int(row['user_id'])].append(int(row['item_id']))
        return {uid: seq[-max_seq_len:] for uid, seq in user_seqs.items()}

    def train_step(self, hist: torch.Tensor, mask: torch.Tensor,
                   pos: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        hist, mask, pos = hist.to(self.device), mask.to(self.device), pos.to(self.device)

        interest_vecs, pos_scores = self.model(hist, mask, pos)
        max_scores = pos_scores.max(dim=-1)[0]

        batch_size = hist.size(0)
        neg_items = torch.randint(1, self.model.num_items, (batch_size, 5), device=self.device)
        neg_embs = F.normalize(self.model.item_embedding(neg_items), p=2, dim=-1)
        neg_scores = torch.bmm(interest_vecs, neg_embs.transpose(1, 2)).max(dim=1)[0]

        logits = torch.cat([max_scores.unsqueeze(1), neg_scores], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, interactions: pd.DataFrame, epochs: int = 5,
              batch_size: int = 512, max_seq_len: int = 50):
        print("开始训练 MIND 多兴趣召回模型...")
        user_seqs = self._build_sequences(interactions, max_seq_len)
        user_list = [uid for uid, seq in user_seqs.items() if len(seq) >= 2]

        for epoch in range(epochs):
            np.random.shuffle(user_list)
            total_loss, n_batches = 0.0, 0

            for i in range(0, len(user_list), batch_size):
                batch_users = user_list[i: i + batch_size]
                hist_list, mask_list, pos_list = [], [], []

                for uid in batch_users:
                    seq = user_seqs[uid]
                    pos_item = seq[-1]
                    hist = seq[:-1]
                    pad_len = max_seq_len - len(hist)
                    hist_padded = [0] * pad_len + hist
                    mask_padded = [False] * pad_len + [True] * len(hist)
                    hist_list.append(hist_padded[-max_seq_len:])
                    mask_list.append(mask_padded[-max_seq_len:])
                    pos_list.append(pos_item)

                loss = self.train_step(
                    torch.tensor(hist_list, dtype=torch.long),
                    torch.tensor(mask_list, dtype=torch.bool),
                    torch.tensor(pos_list, dtype=torch.long)
                )
                total_loss += loss
                n_batches += 1

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/max(n_batches,1):.4f}")

        print("MIND 模型训练完成!")


class MINDRecall:
    """MIND 多兴趣召回器"""

    def __init__(self, mind_model: MINDModel, interactions: pd.DataFrame,
                 max_seq_len: int = 50):
        self.mind_model = mind_model
        self.max_seq_len = max_seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mind_model.to(self.device)

        # 构建用户历史序列
        self.user_seqs: Dict[int, List[int]] = defaultdict(list)
        if 'timestamp' in interactions.columns:
            interactions = interactions.sort_values('timestamp')
        for _, row in interactions.iterrows():
            self.user_seqs[int(row['user_id'])].append(int(row['item_id']))

        self._item_ids: List[int] = []
        self._item_matrix: Optional[np.ndarray] = None
        self._build_item_index()

    def _build_item_index(self):
        print("构建 MIND 物品向量索引...")
        all_item_ids = list(range(1, self.mind_model.num_items + 1))
        batch_size = 2048
        all_embs = []

        self.mind_model.eval()
        with torch.no_grad():
            for i in range(0, len(all_item_ids), batch_size):
                batch = torch.tensor(all_item_ids[i: i + batch_size],
                                     dtype=torch.long).to(self.device)
                emb = self.mind_model.item_embedding(batch)
                emb = F.normalize(emb, p=2, dim=-1)
                all_embs.append(emb.cpu().numpy())

        self._item_ids = all_item_ids
        self._item_matrix = np.vstack(all_embs).astype(np.float32)
        print(f"MIND 物品向量索引构建完成: {len(self._item_ids)} 个物品")

    def recommend(self, user_id: int, n: int = 120) -> List[Tuple[int, float]]:
        if self._item_matrix is None:
            return []

        seq = self.user_seqs.get(user_id, [])
        if not seq:
            return []

        hist = seq[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(hist)
        hist_padded = [0] * pad_len + hist
        mask = [False] * pad_len + [True] * len(hist)

        hist_tensor = torch.tensor([hist_padded], dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor([mask], dtype=torch.bool).to(self.device)

        interest_vecs = self.mind_model.get_user_interests(hist_tensor, mask_tensor)[0]

        n_per_interest = max(n // interest_vecs.shape[0], 10)
        user_history = set(seq)
        all_candidates: Dict[int, float] = {}

        for vec in interest_vecs:
            scores = self._item_matrix @ vec
            top_indices = np.argsort(scores)[-n_per_interest * 2:][::-1]
            for idx in top_indices:
                item_id = self._item_ids[idx]
                if item_id in user_history:
                    continue
                score = float(scores[idx])
                if item_id not in all_candidates or all_candidates[item_id] < score:
                    all_candidates[item_id] = score

        return sorted(all_candidates.items(), key=lambda x: x[1], reverse=True)[:n]

    def save(self, filepath: str):
        save_data = {
            'max_seq_len': self.max_seq_len,
            'user_seqs': dict(self.user_seqs),
            'item_ids': self._item_ids,
            'item_matrix': self._item_matrix,
        }
        with open(filepath + ".recall.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        torch.save(self.mind_model.state_dict(), filepath + ".model.pt")
        print(f"MIND 召回器已保存到 {filepath}")

    @classmethod
    def load(cls, filepath: str, mind_model: MINDModel) -> 'MINDRecall':
        with open(filepath + ".recall.pkl", 'rb') as f:
            save_data = pickle.load(f)
        state_dict = torch.load(filepath + ".model.pt", map_location='cpu')
        mind_model.load_state_dict(state_dict)

        instance = cls.__new__(cls)
        instance.mind_model = mind_model
        instance.max_seq_len = save_data['max_seq_len']
        instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        instance.mind_model.to(instance.device)
        instance.user_seqs = defaultdict(list, save_data['user_seqs'])
        instance._item_ids = save_data['item_ids']
        instance._item_matrix = save_data['item_matrix']
        print(f"MIND 召回器已从 {filepath} 加载")
        return instance
