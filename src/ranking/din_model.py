"""
DIN (Deep Interest Network) 模型实现
基于阿里巴巴的DIN论文
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List, Tuple


class DIN(keras.Model):
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
        self.embedding_layers = {}
        all_feature_columns = (user_feature_columns + item_feature_columns + 
                              context_feature_columns + behavior_feature_columns)
        
        for feat_col in all_feature_columns:
            if feat_col['type'] == 'categorical':
                self.embedding_layers[feat_col['name']] = layers.Embedding(
                    input_dim=feat_col['vocab_size'],
                    output_dim=embedding_dim,
                    name=f"emb_{feat_col['name']}"
                )
        
        # 注意力网络
        self.attention_layers = []
        for units in attention_hidden_units:
            self.attention_layers.append(layers.Dense(units, activation='relu'))
        self.attention_output = layers.Dense(1, activation=None)
        
        # DNN网络
        self.dnn_layers = []
        for units in dnn_hidden_units:
            self.dnn_layers.append(layers.Dense(units, activation='relu'))
            self.dnn_layers.append(layers.Dropout(dropout_rate))
        
        # 输出层
        self.output_layer = layers.Dense(1, activation='sigmoid', name='output')
    
    def attention(self, 
                  query: tf.Tensor,
                  keys: tf.Tensor,
                  keys_length: tf.Tensor) -> tf.Tensor:
        """
        注意力机制
        Args:
            query: 候选物品 embedding, shape: (batch_size, embedding_dim)
            keys: 用户历史行为序列 embeddings, shape: (batch_size, seq_len, embedding_dim)
            keys_length: 每个用户的真实序列长度, shape: (batch_size,)
        Returns:
            加权后的用户兴趣表示, shape: (batch_size, embedding_dim)
        """
        batch_size = tf.shape(keys)[0]
        max_seq_len = tf.shape(keys)[1]
        
        # 将query扩展到序列长度
        queries = tf.tile(tf.expand_dims(query, 1), [1, max_seq_len, 1])  # (batch, seq_len, emb_dim)
        
        # 计算query和key的交互特征
        # [query, key, query-key, query*key]
        attention_input = tf.concat([
            queries,
            keys,
            queries - keys,
            queries * keys
        ], axis=-1)  # (batch, seq_len, 4*emb_dim)
        
        # 通过注意力网络
        attention_output = attention_input
        for layer in self.attention_layers:
            attention_output = layer(attention_output)
        
        # 计算注意力得分
        attention_scores = self.attention_output(attention_output)  # (batch, seq_len, 1)
        attention_scores = tf.squeeze(attention_scores, axis=-1)  # (batch, seq_len)
        
        # 创建mask，屏蔽padding位置
        mask = tf.sequence_mask(keys_length, max_seq_len)  # (batch, seq_len)
        paddings = tf.ones_like(attention_scores) * (-2**32 + 1)
        attention_scores = tf.where(mask, attention_scores, paddings)
        
        # Softmax归一化
        attention_weights = tf.nn.softmax(attention_scores, axis=1)  # (batch, seq_len)
        attention_weights = tf.expand_dims(attention_weights, axis=1)  # (batch, 1, seq_len)
        
        # 加权求和
        user_interest = tf.matmul(attention_weights, keys)  # (batch, 1, emb_dim)
        user_interest = tf.squeeze(user_interest, axis=1)  # (batch, emb_dim)
        
        return user_interest
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        前向传播
        Args:
            inputs: 输入特征字典
            training: 是否训练模式
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
                user_embeddings.append(tf.expand_dims(inputs[feat_col['name']], axis=-1))
          # 2. 候选物品embedding
        item_embeddings = []
        for feat_col in self.item_feature_columns:
            if feat_col['type'] == 'categorical':
                emb = self.embedding_layers[feat_col['name']](inputs[feat_col['name']])
                item_embeddings.append(emb)
            else:
                item_embeddings.append(tf.expand_dims(inputs[feat_col['name']], axis=-1))
        
        candidate_item_emb = tf.concat(item_embeddings, axis=-1)
        
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
                context_embeddings.append(tf.expand_dims(inputs[feat_col['name']], axis=-1))
        
        # 6. 拼接所有特征
        all_features = tf.concat(
            user_embeddings + [user_interest, candidate_item_emb] + context_embeddings,
            axis=-1
        )
        
        # 7. 通过DNN
        dnn_output = all_features
        for layer in self.dnn_layers:
            dnn_output = layer(dnn_output, training=training)
        
        # 8. 输出预测
        output = self.output_layer(dnn_output)
        
        return output


class DINTrainer:
    """DIN模型训练器"""
    
    def __init__(self, model: DIN, config: Dict):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        )
        
        # 损失函数
        self.loss_fn = keras.losses.BinaryCrossentropy()
        
        # 评估指标
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.train_auc_metric = keras.metrics.AUC(name='train_auc')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
        self.val_auc_metric = keras.metrics.AUC(name='val_auc')
    
    @tf.function
    def train_step(self, x: Dict[str, tf.Tensor], y: tf.Tensor) -> tf.Tensor:
        """训练步骤"""
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.loss_fn(y, y_pred)
        
        # 计算梯度并更新
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # 更新指标
        self.train_loss_metric.update_state(loss)
        self.train_auc_metric.update_state(y, y_pred)
        
        return loss
    
    @tf.function
    def val_step(self, x: Dict[str, tf.Tensor], y: tf.Tensor) -> tf.Tensor:
        """验证步骤"""
        y_pred = self.model(x, training=False)
        loss = self.loss_fn(y, y_pred)
        
        # 更新指标
        self.val_loss_metric.update_state(loss)
        self.val_auc_metric.update_state(y, y_pred)
        
        return loss
    
    def train(self, 
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int = 10,
              save_path: str = "models/din_model"):
        """
        训练模型
        """
        best_val_auc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 重置指标
            self.train_loss_metric.reset_state()
            self.train_auc_metric.reset_state()
            self.val_loss_metric.reset_state()
            self.val_auc_metric.reset_state()
            
            # 训练
            for batch, (x, y) in enumerate(train_dataset):
                self.train_step(x, y)
                
                if batch % 100 == 0:
                    print(f"Batch {batch}, Loss: {self.train_loss_metric.result():.4f}, "
                          f"AUC: {self.train_auc_metric.result():.4f}")
            
            # 验证
            for x, y in val_dataset:
                self.val_step(x, y)
            
            # 打印epoch结果
            print(f"\nTrain Loss: {self.train_loss_metric.result():.4f}, "
                  f"Train AUC: {self.train_auc_metric.result():.4f}")
            print(f"Val Loss: {self.val_loss_metric.result():.4f}, "
                  f"Val AUC: {self.val_auc_metric.result():.4f}")
              # 保存最佳模型
            val_auc = self.val_auc_metric.result()
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                self.model.save_weights(f"{save_path}_best.weights.h5")
                print(f"保存最佳模型，Val AUC: {val_auc:.4f}")
        
        print(f"\n训练完成！最佳Val AUC: {best_val_auc:.4f}")
