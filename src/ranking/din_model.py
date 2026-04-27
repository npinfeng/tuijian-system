"""
DIN (Deep Interest Network) 模型实现 (PyTorch 版)
基于阿里巴巴的DIN论文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score


class DIN(nn.Module):
    """
    Deep Interest Network
    核心思想：使用注意力机制动态计算用户兴趣表示
    """

    def __init__(self,
                 user_feature_columns: List,
                 item_feature_columns: List,
                 context_feature_columns: List,
                 behavior_feature_columns: List,
                 embedding_dim: int = 32,
                 attention_hidden_units: List[int] = [80, 40],
                 dnn_hidden_units: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2):
        """
        Args:
            user_feature_columns: 用户特征列
            item_feature_columns: 物品特征列
            context_feature_columns: 上下文特征列
            behavior_feature_columns: 用户行为序列特征列
            embedding_dim: embedding维度
            attention_hidden_units: 注意力网络隐藏层单元数
            dnn_hidden_units: DNN隐藏层单元数
            dropout_rate: dropout比例
        """
        super(DIN, self).__init__()

        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.context_feature_columns = context_feature_columns
        self.behavior_feature_columns = behavior_feature_columns

        self.embedding_dim = embedding_dim
        self.attention_hidden_units = attention_hidden_units
        self.dnn_hidden_units = dnn_hidden_units
        self.dropout_rate = dropout_rate

        # 构建Embedding层
        self.embedding_layers = nn.ModuleDict()
        all_feature_columns = (user_feature_columns + item_feature_columns +
                               context_feature_columns + behavior_feature_columns)

        for feat_col in all_feature_columns:
            if feat_col['type'] == 'categorical':
                self.embedding_layers[feat_col['name']] = nn.Embedding(
                    num_embeddings=feat_col['vocab_size'],
                    embedding_dim=embedding_dim,
                )

        # 注意力网络
        attention_layers = []
        attention_input_dim = embedding_dim * 4  # [query, key, query-key, query*key]
        for units in attention_hidden_units:
            attention_layers.append(nn.Linear(attention_input_dim, units))
            attention_layers.append(nn.ReLU())
            attention_input_dim = units
        self.attention_mlp = nn.Sequential(*attention_layers)
        self.attention_output = nn.Linear(attention_input_dim, 1)

        # 计算DNN输入维度
        # 用户特征维度
        user_input_dim = 0
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'categorical':
                user_input_dim += embedding_dim
            else:
                user_input_dim += 1

        # 候选物品特征维度
        item_input_dim = 0
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'categorical':
                item_input_dim += embedding_dim
            else:
                item_input_dim += 1

        # 上下文特征维度
        context_input_dim = 0
        for feat_col in context_feature_columns:
            if feat_col['type'] == 'categorical':
                context_input_dim += embedding_dim
            else:
                context_input_dim += 1

        # 用户兴趣向量维度 = embedding_dim
        total_input_dim = user_input_dim + embedding_dim + item_input_dim + context_input_dim

        # DNN网络
        dnn_layers = []
        for units in dnn_hidden_units:
            dnn_layers.append(nn.Linear(total_input_dim, units))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            total_input_dim = units
        self.dnn = nn.Sequential(*dnn_layers)

        # 输出层
        self.output_layer = nn.Linear(total_input_dim, 1)

    def attention(self,
                  query: torch.Tensor,
                  keys: torch.Tensor,
                  keys_length: torch.Tensor) -> torch.Tensor:
        """
        注意力机制
        Args:
            query: 候选物品 embedding, shape: (batch_size, embedding_dim)
            keys: 用户历史行为序列 embeddings, shape: (batch_size, seq_len, embedding_dim)
            keys_length: 每个用户的真实序列长度, shape: (batch_size,)
        Returns:
            加权后的用户兴趣表示, shape: (batch_size, embedding_dim)
        """
        seq_len = keys.size(1)

        # 将query扩展到序列长度
        queries = query.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, emb_dim)

        # 计算query和key的交互特征
        # [query, key, query-key, query*key]
        attention_input = torch.cat([
            queries,
            keys,
            queries - keys,
            queries * keys
        ], dim=-1)  # (batch, seq_len, 4*emb_dim)

        # 通过注意力网络
        attention_output = self.attention_mlp(attention_input)

        # 计算注意力得分
        attention_scores = self.attention_output(attention_output).squeeze(-1)  # (batch, seq_len)

        # 创建mask，屏蔽padding位置
        mask = torch.arange(seq_len, device=keys.device).unsqueeze(0) < keys_length.unsqueeze(1)  # (batch, seq_len)
        attention_scores = attention_scores.masked_fill(~mask, -1e9)

        # Softmax归一化
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)  # (batch, 1, seq_len)

        # 加权求和
        user_interest = torch.bmm(attention_weights, keys).squeeze(1)  # (batch, emb_dim)

        return user_interest

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        Args:
            inputs: 输入特征字典
        Returns:
            预测概率
        """
        # 1. 获取各类特征的embedding
        user_embeddings = []
        for feat_col in self.user_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.embedding_layers[feat_col['name']](inputs[feat_col['name']])
                user_embeddings.append(emb)
            else:
                user_embeddings.append(inputs[feat_col['name']].unsqueeze(-1))

        # 2. 候选物品embedding
        item_embeddings = []
        for feat_col in self.item_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.embedding_layers[feat_col['name']](inputs[feat_col['name']])
                item_embeddings.append(emb)
            else:
                item_embeddings.append(inputs[feat_col['name']].unsqueeze(-1))

        candidate_item_emb = torch.cat(item_embeddings, dim=-1)

        # 3. 用户行为序列embedding
        # 获取候选物品的item_id embedding（用于注意力计算）
        candidate_item_id_emb = self.embedding_layers['item_id'](inputs['item_id'])

        # 获取历史行为序列的item_id embedding
        behavior_seq_emb = self.embedding_layers['item_id'](inputs['hist_item_seq'])
        seq_length = inputs['seq_length']

        # 4. 计算注意力，得到动态用户兴趣（使用相同维度的item_id embedding）
        user_interest = self.attention(
            query=candidate_item_id_emb,
            keys=behavior_seq_emb,
            keys_length=seq_length
        )

        # 5. 上下文特征
        context_embeddings = []
        for feat_col in self.context_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.embedding_layers[feat_col['name']](inputs[feat_col['name']])
                context_embeddings.append(emb)
            else:
                context_embeddings.append(inputs[feat_col['name']].unsqueeze(-1))

        # 6. 拼接所有特征
        all_features = torch.cat(
            user_embeddings + [user_interest, candidate_item_emb] + context_embeddings,
            dim=-1
        )

        # 7. 通过DNN
        dnn_output = self.dnn(all_features)

        # 8. 输出预测
        output = torch.sigmoid(self.output_layer(dnn_output))

        return output


class DINTrainer:
    """DIN模型训练器"""

    def __init__(self, model: DIN, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001)
        )

        # 损失函数
        self.loss_fn = nn.BCELoss()

    def train_step(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> float:
        """训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()

        # 将数据移到设备上
        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device)

        y_pred = self.model(x).squeeze(-1)
        loss = self.loss_fn(y_pred, y)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        """验证步骤"""
        self.model.eval()
        with torch.no_grad():
            x = {k: v.to(self.device) for k, v in x.items()}
            y = y.to(self.device)

            y_pred = self.model(x).squeeze(-1)
            loss = self.loss_fn(y_pred, y)

        return loss.item(), y.cpu().numpy(), y_pred.cpu().numpy()

    def train(self,
              train_dataloader,
              val_dataloader,
              epochs: int = 10,
              save_path: str = "models/din_model"):
        """
        训练模型
        """
        best_val_auc = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # 训练阶段
            total_train_loss = 0.0
            n_train_batches = 0

            for batch_idx, (x, y) in enumerate(train_dataloader):
                loss = self.train_step(x, y)
                total_train_loss += loss
                n_train_batches += 1

                if batch_idx % 100 == 0:
                    avg_loss = total_train_loss / n_train_batches
                    print(f"Batch {batch_idx}, Loss: {avg_loss:.4f}")

            # 验证阶段
            all_labels = []
            all_preds = []
            total_val_loss = 0.0
            n_val_batches = 0

            for x, y in val_dataloader:
                loss, labels, preds = self.val_step(x, y)
                total_val_loss += loss
                n_val_batches += 1
                all_labels.append(labels)
                all_preds.append(preds)

            all_labels = np.concatenate(all_labels)
            all_preds = np.concatenate(all_preds)

            train_loss = total_train_loss / max(n_train_batches, 1)
            val_loss = total_val_loss / max(n_val_batches, 1)
            val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.0

            # 打印epoch结果
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

            # 保存最佳模型
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), f"{save_path}_best.pt")
                print(f"保存最佳模型，Val AUC: {val_auc:.4f}")

        print(f"\n训练完成！最佳Val AUC: {best_val_auc:.4f}")
