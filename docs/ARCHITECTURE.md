# 项目架构详细文档

## 1. 系统架构

### 1.1 整体架构
```
用户请求 
    ↓
API网关 (负载均衡)
    ↓
推荐服务层
    ├── 召回层 (Recall)
    ├── 粗排层 (Pre-Ranking) 
    ├── 精排层 (Ranking)
    └── 重排层 (Re-Ranking)
    ↓
特征服务
    ├── 实时特征 (Redis)
    └── 离线特征 (Hive/Parquet)
    ↓
存储层
    ├── 用户画像库
    ├── 物品画像库
    └── 行为日志库
```

### 1.2 数据流

#### 离线数据流
1. **原始数据采集** → Kafka
2. **数据清洗** → Flink/Spark
3. **特征工程** → 离线特征库
4. **模型训练** → TensorFlow/PyTorch
5. **模型部署** → TensorFlow Serving

#### 在线数据流
1. **用户请求** → 推荐服务
2. **特征获取** → Redis (实时特征)
3. **多路召回** → 候选集生成
4. **排序打分** → 模型推理
5. **结果返回** → 用户端
6. **行为日志** → Kafka → 特征更新

## 2. 召回层详解

### 2.1 ItemCF (物品协同过滤)
**算法原理:**
- 基于物品相似度矩阵
- 公式: $sim(i,j) = \frac{|N(i) \cap N(j)|}{\sqrt{|N(i)| \cdot |N(j)|}}$
- 适用场景: 短期兴趣捕捉

**实现要点:**
- 使用稀疏矩阵节省内存
- 只保留Top-K相似物品
- 增量更新相似度矩阵

### 2.2 双塔模型 (Two Tower)
**模型结构:**
```
User Tower:                Item Tower:
User Features →            Item Features →
    Embedding                  Embedding
        ↓                          ↓
    DNN Layers                 DNN Layers
        ↓                          ↓
    User Vector               Item Vector
        └──────── Dot Product ────────┘
                      ↓
                 Score (相似度)
```

**训练策略:**
- 对比学习 (Contrastive Learning)
- InfoNCE Loss
- 负采样: batch内负样本

**在线服务:**
- 离线生成用户/物品向量
- 使用Faiss进行ANN检索
- 响应时间 < 10ms

### 2.3 MIND (多兴趣召回)
**核心思想:**
- 用户有多个兴趣
- 使用胶囊网络建模多兴趣
- 每个兴趣对应一个向量

**优势:**
- 提升推荐多样性
- 适合兴趣广泛的用户

## 3. 排序层详解

### 3.1 DIN (Deep Interest Network)
**核心创新: 注意力机制**

```python
# 注意力计算过程
1. Query: 候选物品embedding
2. Keys: 用户历史行为序列embeddings
3. 计算attention score:
   score = MLP([query, key, query-key, query*key])
4. Softmax归一化
5. 加权求和得到用户兴趣表示
```

**特点:**
- 动态兴趣表示
- 局部激活
- CTR提升显著 (10-15%)

**改进方向 (面试可讨论):**
- DIEN: 增加兴趣演化层 (GRU)
- DSIN: 会话兴趣网络
- MIMN: 记忆网络

### 3.2 多目标学习
**场景:**
同时优化多个目标:
- 点击率 (CTR)
- 完播率 (VTR)
- 分享率 (Share Rate)
- 点赞率 (Like Rate)

**方法:**
1. **Hard Parameter Sharing**: 共享底层
2. **MMoE**: Multi-gate Mixture-of-Experts
3. **PLE**: Progressive Layered Extraction

## 4. 特征工程

### 4.1 特征分类

#### 用户特征
- **基础属性**: user_id, age, gender, city
- **活跃度**: 注册天数, 近7日行为数
- **偏好**: 喜欢的类目, 常看作者

#### 物品特征
- **基础属性**: item_id, category, author_id
- **内容特征**: 标题, 标签, 时长
- **统计特征**: CTR, 完播率, 点赞数

#### 上下文特征
- **时间**: hour, weekday, is_weekend
- **位置**: position (推荐位位置)
- **设备**: device_type, network_type

#### 交叉特征
- user_id × category
- age_group × category
- hour × category

### 4.2 实时特征架构

**Redis存储结构:**
```
Key: user_features:{user_id}
Value: {
    "click_7d": 50,
    "watch_time_7d": 1800,
    "last_category": "game"
}

Key: item_ctr:{item_id}
Value: clicks / impressions (实时计算)
```

**更新策略:**
- 用户行为实时写入Kafka
- Flink消费并更新Redis
- 设置TTL防止内存爆炸

## 5. 重排层详解

### 5.1 MMR算法
**公式:**
```
MMR = arg max [λ × Sim(D_i, Q) - (1-λ) × max Sim(D_i, D_j)]
                                               j∈S
```
- λ: 相关性和多样性的权重
- S: 已选择的文档集合

### 5.2 业务规则
1. **作者打散**: 连续推荐同一作者不超过2个
2. **类目控制**: 同类目不超过3个
3. **时效性**: 新内容加权
4. **安全过滤**: 低质内容过滤

## 6. 性能优化

### 6.1 缓存策略
```
L1: 本地内存缓存 (热门用户画像)
L2: Redis缓存 (实时特征)
L3: 数据库 (离线特征)
```

### 6.2 模型优化
- **量化**: FP32 → INT8
- **蒸馏**: 大模型 → 小模型
- **剪枝**: 去除冗余参数

### 6.3 服务优化
- **批处理**: 动态batching
- **异步**: 非阻塞IO
- **降级**: 多级降级策略

## 7. 监控与运维

### 7.1 核心指标
- **QPS**: 每秒请求数
- **RT**: 响应时间 (P50, P99)
- **错误率**: 4xx, 5xx错误
- **模型指标**: AUC, CTR, Diversity

### 7.2 告警规则
```yaml
- name: high_latency
  condition: P99 > 100ms
  duration: 5m
  action: alert

- name: low_ctr
  condition: CTR < baseline * 0.95
  duration: 10m
  action: alert_and_rollback
```

## 8. 待优化方向 (面试重点)

### 8.1 模型方面
1. **序列建模增强**
   - Transformer替代DIN的attention
   - BERT4Rec: 双向建模
   - SASRec: self-attention

2. **图神经网络**
   - GCN: 图卷积网络
   - GraphSAGE: 归纳式学习
   - 建模user-item-content多元关系

3. **多模态融合**
   - 视频内容理解 (CV)
   - 标题语义 (NLP)
   - 音频特征提取

### 8.2 工程方面
1. **在线学习**
   - 实时模型更新
   - 增量学习
   - 解决数据分布漂移

2. **因果推断**
   - IPS: Inverse Propensity Score
   - DR: Doubly Robust
   - 去除位置偏差、曝光偏差

3. **强化学习**
   - 长期价值优化
   - DQN, Actor-Critic
   - 探索与利用平衡

### 8.3 业务方面
1. **冷启动优化**
   - 内容冷启动: 基于内容画像
   - 用户冷启动: 迁移学习
   - 探索策略: Thompson Sampling

2. **实时性提升**
   - 分钟级模型更新
   - 流式特征计算
   - 边缘计算

3. **个性化深化**
   - 细粒度兴趣建模
   - 场景化推荐
   - 上下文感知

## 9. 面试问题准备

### 技术深度问题
1. **DIN的注意力机制为什么有效?**
   - 传统方法用固定的用户表示
   - DIN根据候选物品动态调整
   - 局部激活原理

2. **如何解决数据稀疏问题?**
   - Embedding技术
   - 迁移学习
   - 引入side information

3. **召回和排序的区别?**
   - 召回: 快速、粗糙、大规模
   - 排序: 精准、复杂、小规模
   - Trade-off: 效率 vs 准确性

### 系统设计问题
1. **如何设计一个高并发推荐系统?**
   - 负载均衡
   - 多级缓存
   - 服务降级
   - 异步处理

2. **特征延迟问题如何解决?**
   - 实时特征: Redis + Flink
   - 准实时: T+1更新
   - 离线特征: 定期批处理

3. **模型如何AB测试?**
   - 流量分层
   - 实验指标定义
   - 统计显著性检验

### 优化思路问题
1. **CTR上去了但时长下降怎么办?**
   - 多目标优化
   - 调整目标权重
   - 引入时长预估

2. **如何提升推荐多样性?**
   - MMR重排
   - DPP行列式点过程
   - 探索策略

3. **新内容如何获得曝光?**
   - 内容冷启动策略
   - 探索流量分配
   - Thompson Sampling
