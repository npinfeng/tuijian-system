"""
双塔模型实现 (Two Tower Model) - PyTorch 版
用于召回阶段，支持高效的向量检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import roc_auc_score


class TwoTowerModel(nn.Module):
    """
    双塔模型
    User Tower: 编码用户特征
    Item Tower: 编码物品特征
    通过余弦相似度计算user-item匹配分数
    """

    def __init__(self,
                 user_feature_columns: List[Dict],
                 item_feature_columns: List[Dict],
                 embedding_dim: int = 32,
                 user_hidden_units: List[int] = [256, 128, 64],
                 item_hidden_units: List[int] = [256, 128, 64],
                 output_dim: int = 64,
                 dropout_rate: float = 0.2):
        """
        Args:
            user_feature_columns: 用户特征配置
            item_feature_columns: 物品特征配置
            embedding_dim: categorical特征的embedding维度
            user_hidden_units: User Tower的隐藏层配置
            item_hidden_units: Item Tower的隐藏层配置
            output_dim: 最终向量的维度
            dropout_rate: dropout比例
        """
        super(TwoTowerModel, self).__init__()

        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim

        # User Tower的Embedding层
        self.user_embedding_layers = nn.ModuleDict()
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'categorical':
                self.user_embedding_layers[feat_col['name']] = nn.Embedding(
                    num_embeddings=feat_col['vocab_size'],
                    embedding_dim=embedding_dim,
                )

        # Item Tower的Embedding层
        self.item_embedding_layers = nn.ModuleDict()
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'categorical':
                self.item_embedding_layers[feat_col['name']] = nn.Embedding(
                    num_embeddings=feat_col['vocab_size'],
                    embedding_dim=embedding_dim,
                )

        # 特征处理层
        self.user_num_projs = nn.ModuleDict()
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'numerical':
                # 将 1 维数值特征投影到 16 维，增强存在感
                self.user_num_projs[feat_col['name']] = nn.Linear(1, 16)

        # 计算User Tower输入维度
        user_input_dim = 0
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'categorical':
                user_input_dim += embedding_dim
            else:
                user_input_dim += 16

        # User Tower的DNN层
        user_dnn_layers = []
        in_dim = user_input_dim
        for units in user_hidden_units:
            user_dnn_layers.append(nn.Linear(in_dim, units))
            user_dnn_layers.append(nn.LeakyReLU(0.2)) # 改用 LeakyReLU 防止神经元坏死
            # 移除 BatchNorm1d，在双塔对比学习中 BN 容易造成信息泄露或训练波动
            user_dnn_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.user_dnn = nn.Sequential(*user_dnn_layers)
        self.user_output_layer = nn.Linear(in_dim, output_dim)

        self.item_num_projs = nn.ModuleDict()
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'numerical':
                self.item_num_projs[feat_col['name']] = nn.Linear(1, 16)

        # 计算Item Tower输入维度
        item_input_dim = 0
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'categorical':
                item_input_dim += embedding_dim
            else:
                item_input_dim += 16

        # Item Tower的DNN层
        item_dnn_layers = []
        in_dim = item_input_dim
        for units in item_hidden_units:
            item_dnn_layers.append(nn.Linear(in_dim, units))
            item_dnn_layers.append(nn.LeakyReLU(0.2))
            item_dnn_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.item_dnn = nn.Sequential(*item_dnn_layers)
        self.item_output_layer = nn.Linear(in_dim, output_dim)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """更科学的初始化，防止梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def user_tower(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        User Tower
        Args:
            inputs: 用户特征字典
        Returns:
            用户向量表示, shape: (batch_size, output_dim)
        """
        # 获取所有用户特征的embedding
        user_embeddings = []
        for feat_col in self.user_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.user_embedding_layers[feat_col['name']](inputs[feat_col['name']])
                user_embeddings.append(emb)
            else:  # numerical
                val = inputs[feat_col['name']].unsqueeze(-1)
                user_embeddings.append(self.user_num_projs[feat_col['name']](val))

        # 拼接所有特征
        user_features = torch.cat(user_embeddings, dim=-1)

        # 通过DNN
        user_features = self.user_dnn(user_features)

        # 输出层 (移除 L2 归一化，改用纯内积，释放梯度)
        user_vector = self.user_output_layer(user_features)

        return user_vector

    def item_tower(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Item Tower
        Args:
            inputs: 物品特征字典
        Returns:
            物品向量表示, shape: (batch_size, output_dim)
        """
        # 获取所有物品特征的embedding
        item_embeddings = []
        for feat_col in self.item_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.item_embedding_layers[feat_col['name']](inputs[feat_col['name']])
                item_embeddings.append(emb)
            else:  # numerical
                val = inputs[feat_col['name']].unsqueeze(-1)
                item_embeddings.append(self.item_num_projs[feat_col['name']](val))

        # 拼接所有特征
        item_features = torch.cat(item_embeddings, dim=-1)

        # 通过DNN
        item_features = self.item_dnn(item_features)

        # 输出层 (移除 L2 归一化)
        item_vector = self.item_output_layer(item_features)

        return item_vector

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            inputs: 包含用户和物品特征的字典
        Returns:
            预测得分 (余弦相似度), shape: (batch_size,)
        """
        # 分离用户和物品特征
        user_inputs = {k: v for k, v in inputs.items()
                       if any(k == fc['name'] for fc in self.user_feature_columns)}
        item_inputs = {k: v for k, v in inputs.items()
                       if any(k == fc['name'] for fc in self.item_feature_columns)}

        # 计算用户和物品向量
        user_vector = self.user_tower(user_inputs)
        item_vector = self.item_tower(item_inputs)

        # 计算内积打分
        score = (user_vector * item_vector).sum(dim=1)

        return score


class TwoTowerTrainer:
    """双塔模型训练器"""

    def __init__(self, model: TwoTowerModel, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器：余弦退火，有助于后期收敛
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 10), eta_min=1e-5
        )

    def contrastive_loss(self,
                         user_vectors: torch.Tensor,
                         item_vectors: torch.Tensor,
                         item_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        对比学习损失 (InfoNCE Loss)
        item_ids: 用于处理 batch 内的 False Negatives (多个用户点击了同一个视频)
        """
        # 计算所有user-item对的相似度矩阵（纯内积，无需温度缩放）
        similarity_matrix = torch.mm(user_vectors, item_vectors.t())  # (batch, batch)

        batch_size = user_vectors.size(0)
        labels = torch.arange(batch_size, device=user_vectors.device)

        # 处理 False Negatives：如果 batch 内有相同的 item_id，将对应的负样本位置 Mask 掉
        if item_ids is not None:
            # 找到 item_id 相同的位置 (batch, batch)
            # mask[i, j] = True 表示 i-user 和 j-item 实际上是同一种匹配，不应作为负样本
            mask = (item_ids.unsqueeze(0) == item_ids.unsqueeze(1))
            # 将这些位置的相似度减去一个大值，使其在 softmax 中失效（忽略对角线，对角线是正样本）
            diag_mask = torch.eye(batch_size, device=user_vectors.device).bool()
            mask = mask & (~diag_mask)
            similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

        # 计算交叉熵损失
        loss_ui = F.cross_entropy(similarity_matrix, labels)
        loss_iu = F.cross_entropy(similarity_matrix.t(), labels)
        loss = (loss_ui + loss_iu) / 2.0

        return loss

    def train_step(self, inputs: Dict[str, torch.Tensor]) -> float:
        """训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()

        # 将数据移到设备上
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 分离用户和物品特征
        user_inputs = {k: v for k, v in inputs.items()
                       if any(k == fc['name'] for fc in self.model.user_feature_columns)}
        item_inputs = {k: v for k, v in inputs.items()
                       if any(k == fc['name'] for fc in self.model.item_feature_columns)}

        # 获取向量表示
        user_vectors = self.model.user_tower(user_inputs)
        item_vectors = self.model.item_tower(item_inputs)

        # 计算损失
        item_ids = inputs.get('item_id')
        loss = self.contrastive_loss(user_vectors, item_vectors, item_ids)

        loss.backward()
        # 放大裁剪阈值 1.0 -> 5.0，给模型更多“跳出”局部最优的动力
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss.item()

    def train(self,
              train_dataloader,
              val_dataloader,
              epochs: int = 10,
              save_path: str = "models/two_tower",
              patience: int = 3):
        """
        训练模型
        Args:
            patience: 容忍验证集 Loss 不下降的轮数
        """
        best_val_loss = float('inf')
        no_improve_epochs = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练阶段
            self.model.train()
            total_train_loss = 0.0
            n_train_batches = 0

            for batch_idx, inputs in enumerate(train_dataloader):
                loss = self.train_step(inputs)
                total_train_loss += loss
                n_train_batches += 1

                if batch_idx % 100 == 0:
                    avg_loss = total_train_loss / n_train_batches
                    print(f"Batch {batch_idx}, Loss: {avg_loss:.4f}")

            # 验证阶段
            self.model.eval()
            total_val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for inputs in val_dataloader:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    user_inputs = {k: v for k, v in inputs.items()
                                   if any(k == fc['name'] for fc in self.model.user_feature_columns)}
                    item_inputs = {k: v for k, v in inputs.items()
                                   if any(k == fc['name'] for fc in self.model.item_feature_columns)}

                    user_vectors = self.model.user_tower(user_inputs)
                    item_vectors = self.model.item_tower(item_inputs)
                    item_ids = inputs.get('item_id')
                    loss = self.contrastive_loss(user_vectors, item_vectors, item_ids)
                    
                    total_val_loss += loss.item()
                    n_val_batches += 1

            train_loss = total_train_loss / max(n_train_batches, 1)
            val_loss = total_val_loss / max(n_val_batches, 1)

            print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # 保存最佳模型并检查早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_epochs = 0
                torch.save(self.model.state_dict(), f"{save_path}_best.pt")
                print(f"保存最佳模型，Val Loss: {val_loss:.4f}")
            else:
                no_improve_epochs += 1
                print(f"验证集性能未提升 ({no_improve_epochs}/{patience})")

            if no_improve_epochs >= patience:
                print(f"触发早停！在第 {epoch + 1} 轮停止训练。")
                break
                
            # 更新学习率
            self.scheduler.step()

        print(f"\n训练完成！最佳Val Loss: {best_val_loss:.4f}")


class TwoTowerRecall:
    """
    双塔模型召回器
    使用训练好的双塔模型进行向量检索
    """

    def __init__(self, model: TwoTowerModel, item_features: pd.DataFrame):
        """
        Args:
            model: 已加载权重的 TwoTowerModel
            item_features: 包含所有 item 特征的 DataFrame
        """
        self.model = model
        self.item_features = item_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self._item_ids = []
        self._item_matrix = None
        self._build_item_index()

    def _build_item_index(self):
        """预计算并保存所有物品的向量"""
        print("开始构建双塔物品向量索引...")
        self._item_ids = self.item_features['item_id'].tolist()
        
        # 准备物品特征输入
        item_inputs = {}
        for feat_col in self.model.item_feature_columns:
            name = feat_col['name']
            if name in self.item_features.columns:
                if feat_col['type'] == 'categorical':
                    item_inputs[name] = torch.tensor(self.item_features[name].values, dtype=torch.long).to(self.device)
                else:
                    item_inputs[name] = torch.tensor(self.item_features[name].values, dtype=torch.float32).to(self.device)
        
        # 批量计算物品向量
        batch_size = 4096
        item_vectors = []
        
        with torch.no_grad():
            for i in range(0, len(self._item_ids), batch_size):
                batch_inputs = {k: v[i:i+batch_size] for k, v in item_inputs.items()}
                vectors = self.model.item_tower(batch_inputs)
                item_vectors.append(vectors.cpu().numpy())
        
        self._item_matrix = np.vstack(item_vectors)
        print(f"物品向量索引构建完成: {len(self._item_ids)} 个物品")

    def recommend(self, user_id: int, n: int = 50, user_features: Optional[pd.Series] = None) -> List[Tuple[int, float]]:
        """为特定用户推荐"""
        if user_features is None:
            return []

        # 构造用户特征输入
        user_inputs = {}
        for feat_col in self.model.user_feature_columns:
            name = feat_col['name']
            if name == 'user_id':
                user_inputs[name] = torch.tensor([user_id], dtype=torch.long).to(self.device)
            elif name in user_features:
                if feat_col['type'] == 'categorical':
                    user_inputs[name] = torch.tensor([user_features[name]], dtype=torch.long).to(self.device)
                else:
                    user_inputs[name] = torch.tensor([user_features[name]], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            user_vector = self.model.user_tower(user_inputs).cpu().numpy() # (1, dim)
        
        # 计算余弦相似度（已归一化，直接内积）
        scores = (self._item_matrix @ user_vector.T).flatten()
        
        # 排序并返回 top-n
        top_indices = np.argsort(scores)[-n:][::-1]
        return [(self._item_ids[idx], float(scores[idx])) for idx in top_indices]
