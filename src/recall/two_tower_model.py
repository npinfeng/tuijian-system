"""
双塔模型实现 (Two Tower Model) - PyTorch 版
用于召回阶段，支持高效的向量检索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
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

        # 计算User Tower输入维度
        user_input_dim = 0
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'categorical':
                user_input_dim += embedding_dim
            else:
                user_input_dim += 1

        # User Tower的DNN层
        user_dnn_layers = []
        in_dim = user_input_dim
        for units in user_hidden_units:
            user_dnn_layers.append(nn.Linear(in_dim, units))
            user_dnn_layers.append(nn.ReLU())
            user_dnn_layers.append(nn.BatchNorm1d(units))
            user_dnn_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.user_dnn = nn.Sequential(*user_dnn_layers)
        self.user_output_layer = nn.Linear(in_dim, output_dim)

        # 计算Item Tower输入维度
        item_input_dim = 0
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'categorical':
                item_input_dim += embedding_dim
            else:
                item_input_dim += 1

        # Item Tower的DNN层
        item_dnn_layers = []
        in_dim = item_input_dim
        for units in item_hidden_units:
            item_dnn_layers.append(nn.Linear(in_dim, units))
            item_dnn_layers.append(nn.ReLU())
            item_dnn_layers.append(nn.BatchNorm1d(units))
            item_dnn_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.item_dnn = nn.Sequential(*item_dnn_layers)
        self.item_output_layer = nn.Linear(in_dim, output_dim)

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
                user_embeddings.append(inputs[feat_col['name']].unsqueeze(-1))

        # 拼接所有特征
        user_features = torch.cat(user_embeddings, dim=-1)

        # 通过DNN
        user_features = self.user_dnn(user_features)

        # 输出层
        user_vector = self.user_output_layer(user_features)

        # L2归一化
        user_vector = F.normalize(user_vector, p=2, dim=1)

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
                item_embeddings.append(inputs[feat_col['name']].unsqueeze(-1))

        # 拼接所有特征
        item_features = torch.cat(item_embeddings, dim=-1)

        # 通过DNN
        item_features = self.item_dnn(item_features)

        # 输出层
        item_vector = self.item_output_layer(item_features)

        # L2归一化
        item_vector = F.normalize(item_vector, p=2, dim=1)

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

        # 计算余弦相似度 (已归一化，直接内积)
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
            model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )

        # 使用对比学习损失
        self.temperature = config.get('temperature', 0.05)

    def contrastive_loss(self,
                         user_vectors: torch.Tensor,
                         item_vectors: torch.Tensor) -> torch.Tensor:
        """
        对比学习损失 (InfoNCE Loss)
        正样本：batch内的user-item配对
        负样本：batch内其他所有item
        """
        # 计算所有user-item对的相似度矩阵
        # user_vectors: (batch_size, emb_dim)
        # item_vectors: (batch_size, emb_dim)
        similarity_matrix = torch.mm(user_vectors, item_vectors.t())  # (batch, batch)
        similarity_matrix = similarity_matrix / self.temperature

        # 对角线是正样本
        batch_size = user_vectors.size(0)
        labels = torch.arange(batch_size, device=user_vectors.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(similarity_matrix, labels)

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
        loss = self.contrastive_loss(user_vectors, item_vectors)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self,
              train_dataloader,
              val_dataloader,
              epochs: int = 10,
              save_path: str = "models/two_tower"):
        """训练模型"""
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练
            total_train_loss = 0.0
            n_train_batches = 0

            for batch_idx, inputs in enumerate(train_dataloader):
                loss = self.train_step(inputs)
                total_train_loss += loss
                n_train_batches += 1

                if batch_idx % 100 == 0:
                    avg_loss = total_train_loss / n_train_batches
                    print(f"Batch {batch_idx}, Loss: {avg_loss:.4f}")

            # 验证
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
                    loss = self.contrastive_loss(user_vectors, item_vectors)

                    total_val_loss += loss.item()
                    n_val_batches += 1

            # 打印结果
            train_loss = total_train_loss / max(n_train_batches, 1)
            val_loss = total_val_loss / max(n_val_batches, 1)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f"{save_path}_best.pt")
                print(f"保存最佳模型，Val Loss: {val_loss:.4f}")

        print(f"\n训练完成！最佳Val Loss: {best_val_loss:.4f}")
