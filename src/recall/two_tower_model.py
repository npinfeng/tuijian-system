"""
双塔模型实现 (Two Tower Model)
用于召回阶段，支持高效的向量检索
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, List


class TwoTowerModel(keras.Model):
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
        self.user_embedding_layers = {}
        for feat_col in user_feature_columns:
            if feat_col['type'] == 'categorical':
                self.user_embedding_layers[feat_col['name']] = layers.Embedding(
                    input_dim=feat_col['vocab_size'],
                    output_dim=embedding_dim,
                    name=f"user_emb_{feat_col['name']}"
                )
        
        # Item Tower的Embedding层
        self.item_embedding_layers = {}
        for feat_col in item_feature_columns:
            if feat_col['type'] == 'categorical':
                self.item_embedding_layers[feat_col['name']] = layers.Embedding(
                    input_dim=feat_col['vocab_size'],
                    output_dim=embedding_dim,
                    name=f"item_emb_{feat_col['name']}"
                )
        
        # User Tower的DNN层
        self.user_dnn_layers = []
        for units in user_hidden_units:
            self.user_dnn_layers.append(layers.Dense(units, activation='relu'))
            self.user_dnn_layers.append(layers.BatchNormalization())
            self.user_dnn_layers.append(layers.Dropout(dropout_rate))
        self.user_output_layer = layers.Dense(output_dim, activation=None, name='user_output')
        
        # Item Tower的DNN层
        self.item_dnn_layers = []
        for units in item_hidden_units:
            self.item_dnn_layers.append(layers.Dense(units, activation='relu'))
            self.item_dnn_layers.append(layers.BatchNormalization())
            self.item_dnn_layers.append(layers.Dropout(dropout_rate))
        self.item_output_layer = layers.Dense(output_dim, activation=None, name='item_output')
        
        # L2归一化层
        self.user_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
        self.item_norm = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))
    
    def user_tower(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        User Tower
        Args:
            inputs: 用户特征字典
            training: 是否训练模式
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
                user_embeddings.append(tf.expand_dims(inputs[feat_col['name']], axis=-1))
        
        # 拼接所有特征
        user_features = tf.concat(user_embeddings, axis=-1)
        
        # 通过DNN
        for layer in self.user_dnn_layers:
            user_features = layer(user_features, training=training)
        
        # 输出层
        user_vector = self.user_output_layer(user_features)
        
        # L2归一化
        user_vector = self.user_norm(user_vector)
        
        return user_vector
    
    def item_tower(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Item Tower
        Args:
            inputs: 物品特征字典
            training: 是否训练模式
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
                item_embeddings.append(tf.expand_dims(inputs[feat_col['name']], axis=-1))
        
        # 拼接所有特征
        item_features = tf.concat(item_embeddings, axis=-1)
        
        # 通过DNN
        for layer in self.item_dnn_layers:
            item_features = layer(item_features, training=training)
        
        # 输出层
        item_vector = self.item_output_layer(item_features)
        
        # L2归一化
        item_vector = self.item_norm(item_vector)
        
        return item_vector
    
    def call(self, inputs: Dict[str, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        前向传播
        Args:
            inputs: 包含用户和物品特征的字典
            training: 是否训练模式
        Returns:
            预测得分 (余弦相似度), shape: (batch_size,)
        """
        # 分离用户和物品特征
        user_inputs = {k: v for k, v in inputs.items() 
                      if any(k == fc['name'] for fc in self.user_feature_columns)}
        item_inputs = {k: v for k, v in inputs.items() 
                      if any(k == fc['name'] for fc in self.item_feature_columns)}
        
        # 计算用户和物品向量
        user_vector = self.user_tower(user_inputs, training=training)
        item_vector = self.item_tower(item_inputs, training=training)
        
        # 计算余弦相似度 (已归一化，直接内积)
        score = tf.reduce_sum(user_vector * item_vector, axis=1)
        
        return score


class TwoTowerTrainer:
    """双塔模型训练器"""
    
    def __init__(self, model: TwoTowerModel, config: Dict):
        self.model = model
        self.config = config
        
        # 优化器
        self.optimizer = keras.optimizers.Adam(
            learning_rate=config.get('learning_rate', 0.001)
        )
        
        # 使用对比学习损失
        self.temperature = config.get('temperature', 0.05)
        
        # 评估指标
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        self.val_loss_metric = keras.metrics.Mean(name='val_loss')
    
    def contrastive_loss(self, 
                        user_vectors: tf.Tensor,
                        item_vectors: tf.Tensor) -> tf.Tensor:
        """
        对比学习损失 (InfoNCE Loss)
        正样本：batch内的user-item配对
        负样本：batch内其他所有item
        """
        # 计算所有user-item对的相似度矩阵
        # user_vectors: (batch_size, emb_dim)
        # item_vectors: (batch_size, emb_dim)
        similarity_matrix = tf.matmul(user_vectors, item_vectors, transpose_b=True)  # (batch, batch)
        similarity_matrix = similarity_matrix / self.temperature
        
        # 对角线是正样本
        batch_size = tf.shape(user_vectors)[0]
        labels = tf.range(batch_size)
        
        # 计算交叉熵损失
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=similarity_matrix
        )
        
        return tf.reduce_mean(loss)
    
    @tf.function
    def train_step(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        """训练步骤"""
        with tf.GradientTape() as tape:
            # 分离用户和物品特征
            user_inputs = {k: v for k, v in inputs.items() 
                          if any(k == fc['name'] for fc in self.model.user_feature_columns)}
            item_inputs = {k: v for k, v in inputs.items() 
                          if any(k == fc['name'] for fc in self.model.item_feature_columns)}
            
            # 获取向量表示
            user_vectors = self.model.user_tower(user_inputs, training=True)
            item_vectors = self.model.item_tower(item_inputs, training=True)
            
            # 计算损失
            loss = self.contrastive_loss(user_vectors, item_vectors)
        
        # 更新模型
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss_metric.update_state(loss)
        
        return loss
    
    def train(self,
              train_dataset: tf.data.Dataset,
              val_dataset: tf.data.Dataset,
              epochs: int = 10,
              save_path: str = "models/two_tower"):
        """训练模型"""
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 重置指标
            self.train_loss_metric.reset_states()
            self.val_loss_metric.reset_states()
            
            # 训练
            for batch, inputs in enumerate(train_dataset):
                self.train_step(inputs)
                
                if batch % 100 == 0:
                    print(f"Batch {batch}, Loss: {self.train_loss_metric.result():.4f}")
            
            # 验证
            for inputs in val_dataset:
                user_inputs = {k: v for k, v in inputs.items() 
                              if any(k == fc['name'] for fc in self.model.user_feature_columns)}
                item_inputs = {k: v for k, v in inputs.items() 
                              if any(k == fc['name'] for fc in self.model.item_feature_columns)}
                
                user_vectors = self.model.user_tower(user_inputs, training=False)
                item_vectors = self.model.item_tower(item_inputs, training=False)
                loss = self.contrastive_loss(user_vectors, item_vectors)
                self.val_loss_metric.update_state(loss)
            
            # 打印结果
            train_loss = self.train_loss_metric.result()
            val_loss = self.val_loss_metric.result()
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_weights(f"{save_path}_best.h5")
                print(f"保存最佳模型，Val Loss: {val_loss:.4f}")
        
        print(f"\n训练完成！最佳Val Loss: {best_val_loss:.4f}")
