"""
DeepFM 模型实现 (PyTorch 版)
用于精排阶段，结合了 FM 的低阶特征交叉和 DNN 的高阶特征交叉
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score


class FMLayer(nn.Module):
    """
    Factorization Machine (FM) 层
    包含一阶线性部分和二阶特征交叉部分
    """
    def __init__(self, feature_columns: List, embedding_dim: int):
        super(FMLayer, self).__init__()
        self.feature_columns = feature_columns
        
        # 一阶线性部分权重 (w)
        self.linear_layers = nn.ModuleDict()
        for feat_col in feature_columns:
            if feat_col['type'] == 'categorical':
                self.linear_layers[feat_col['name']] = nn.Embedding(feat_col['vocab_size'], 1)
            else:
                self.linear_layers[feat_col['name']] = nn.Linear(1, 1, bias=False)
        
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs: Dict[str, torch.Tensor], embeddings: List[torch.Tensor]) -> torch.Tensor:
        # 1. 一阶部分 (Linear)
        linear_logit = self.bias
        for feat_col in self.feature_columns:
            name = feat_col['name']
            if feat_col['type'] == 'categorical':
                linear_logit = linear_logit + self.linear_layers[name](inputs[name])
            else:
                linear_logit = linear_logit + self.linear_layers[name](inputs[name].unsqueeze(-1))
        
        # 2. 二阶部分 (FM Cross)
        # 拼接所有 categorical embedding: (batch, num_fields, emb_dim)
        # 注意：这里假设只有 categorical 特征参与二阶交叉
        if not embeddings:
            return linear_logit.squeeze(-1)
            
        stacked_embeddings = torch.stack(embeddings, dim=1)
        
        # ΣΣ <vi, vj> xi xj = 0.5 * ( (Σvi)^2 - Σ(vi^2) )
        sum_of_embedding = torch.sum(stacked_embeddings, dim=1)  # (batch, emb_dim)
        square_of_sum = torch.pow(sum_of_embedding, 2)
        
        sum_of_square = torch.sum(torch.pow(stacked_embeddings, 2), dim=1)
        
        cross_logit = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        
        return (linear_logit + cross_logit).squeeze(-1)


class DeepFM(nn.Module):
    """
    DeepFM 模型
    """
    def __init__(self,
                 user_feature_columns: List,
                 item_feature_columns: List,
                 context_feature_columns: List,
                 embedding_dim: int = 32,
                 dnn_hidden_units: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.2):
        super(DeepFM, self).__init__()
        
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.context_feature_columns = context_feature_columns
        
        self.all_feature_columns = user_feature_columns + item_feature_columns + context_feature_columns
        self.embedding_dim = embedding_dim
        
        # 1. Embedding 层
        self.embedding_layers = nn.ModuleDict()
        for feat_col in self.all_feature_columns:
            if feat_col['type'] == 'categorical':
                self.embedding_layers[feat_col['name']] = nn.Embedding(
                    feat_col['vocab_size'], embedding_dim
                )

        # 2. FM 部分
        self.fm = FMLayer(self.all_feature_columns, embedding_dim)
        
        # 3. Deep 部分 (DNN)
        dnn_input_dim = 0
        for feat_col in self.all_feature_columns:
            if feat_col['type'] == 'categorical':
                dnn_input_dim += embedding_dim
            else:
                dnn_input_dim += 1
                
        dnn_layers = []
        in_dim = dnn_input_dim
        for units in dnn_hidden_units:
            dnn_layers.append(nn.Linear(in_dim, units))
            dnn_layers.append(nn.ReLU())
            dnn_layers.append(nn.Dropout(dropout_rate))
            in_dim = units
        self.dnn = nn.Sequential(*dnn_layers)
        self.dnn_output = nn.Linear(in_dim, 1)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 获取所有 Categorical Embeddings
        categorical_embeddings = []
        all_embeddings_list = []
        
        for feat_col in self.all_feature_columns:
            name = feat_col['name']
            if feat_col['type'] == 'categorical':
                emb = self.embedding_layers[name](inputs[name])
                categorical_embeddings.append(emb)
                all_embeddings_list.append(emb)
            else:
                all_embeddings_list.append(inputs[name].unsqueeze(-1))
        
        # FM Logit
        fm_logit = self.fm(inputs, categorical_embeddings)
        
        # Deep Logit
        dnn_input = torch.cat(all_embeddings_list, dim=-1)
        dnn_logit = self.dnn_output(self.dnn(dnn_input)).squeeze(-1)
        
        # 合并
        output = torch.sigmoid(fm_logit + dnn_logit)
        return output


class DeepFMTrainer:
    """DeepFM 训练器"""
    def __init__(self, model: DeepFM, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.get('learning_rate', 0.001))
        self.loss_fn = nn.BCELoss()

    def train_step(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        x = {k: v.to(self.device) for k, v in x.items()}
        y = y.to(self.device).float()
        y_pred = self.model(x)
        loss = self.loss_fn(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, x: Dict[str, torch.Tensor], y: torch.Tensor) -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            x = {k: v.to(self.device) for k, v in x.items()}
            y = y.to(self.device).float()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
        return loss.item(), y.cpu().numpy(), y_pred.cpu().numpy()

    def train(self, train_dataloader, val_dataloader, epochs: int = 10, save_path: str = "models/deepfm_model"):
        best_val_auc = 0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            total_train_loss = 0.0
            n_train_batches = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):
                loss = self.train_step(x, y)
                total_train_loss += loss
                n_train_batches += 1
                if batch_idx % 100 == 0:
                    print(f"Batch {batch_idx}, Loss: {total_train_loss/n_train_batches:.4f}")

            all_labels, all_preds = [], []
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
            val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
            print(f"Val Loss: {total_val_loss/n_val_batches:.4f}, Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(self.model.state_dict(), f"{save_path}_best.pt")
                print(f"保存最佳模型，Val AUC: {val_auc:.4f}")
        print(f"\n训练完成！最佳Val AUC: {best_val_auc:.4f}")
