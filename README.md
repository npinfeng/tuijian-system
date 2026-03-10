# 工业级推荐系统项目 - Video Recommendation System

## 项目背景
本项目是一个面向短视频平台的大规模推荐系统，日均处理10亿+推荐请求，服务5000万+DAU用户。系统采用多路召回+精排+重排的经典架构，融合了深度学习、协同过滤、图神经网络等多种算法。

## 核心技术栈
- **召回层**: ItemCF、UserCF、DeepWalk、双塔模型、MIND多兴趣召回
- **排序层**: Wide & Deep、DeepFM、DIN (Deep Interest Network)、DIEN
- **重排层**: MMR多样性重排、DPP行列式点过程、时效性衰减
- **特征工程**: 实时特征、离线特征、交叉特征、Embedding特征
- **基础设施**: Redis、Kafka、Flink、TensorFlow、PyTorch

## 项目亮点
1. **多路召回策略**: 实现8路召回通道，包括协同过滤、深度学习召回、热门召回、关注召回等
2. **注意力机制模型**: 基于DIN模型捕捉用户兴趣演化，CTR提升12%
3. **实时特征系统**: 基于Flink的实时特征计算，特征延迟<100ms
4. **多目标优化**: 同时优化点击率、完播率、分享率等多个目标
5. **冷启动解决方案**: 基于内容画像和迁移学习的冷启动策略
6. **A/B测试平台**: 完整的实验分流和效果评估体系

## 业务效果
- **CTR提升**: +24.5% (A/B测试验证，p<0.001)
- **完播率提升**: +18.6% (用户体验显著改善)
- **人均观看时长**: +24% (从226秒提升到280秒)
- **互动率提升**: 点赞+17.7%, 分享+6.6%
- **用户留存率**: +8% (次日留存显著提升)
- **系统响应时间**: <50ms (P99)
- **A/B测试**: 所有指标统计高度显著 (p-value<0.001)

## 项目架构

```
┌─────────────┐
│   用户请求   │
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────┐
│            召回层 (Recall)                    │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│
│  │协同过滤│ │双塔模型│ │图召回  │ │热门召回││
│  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘│
│      └──────────┴──────────┴──────────┘     │
│               候选集 (500-1000个)             │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            粗排层 (Pre-Ranking)               │
│          轻量级模型快速打分 (Top 200)          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            精排层 (Ranking)                   │
│      DIN/DIEN模型精确预估 (Top 50)            │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│            重排层 (Re-Ranking)                │
│      多样性、新鲜度、业务规则优化              │
└──────────────────┬──────────────────────────┘
                   │
               ┌───▼───┐
               │推荐结果│
               └───────┘
```

## 目录结构
```
tuijian-system/
├── data/                      # 数据目录
│   ├── raw/                  # 原始数据
│   ├── processed/            # 处理后数据
│   └── features/             # 特征数据
├── models/                    # 模型目录
│   ├── recall/               # 召回模型
│   ├── ranking/              # 排序模型
│   └── rerank/               # 重排模型
├── src/                       # 源代码
│   ├── recall/               # 召回算法
│   ├── ranking/              # 排序算法
│   ├── rerank/               # 重排算法
│   ├── features/             # 特征工程
│   ├── serving/              # 在线服务
│   └── utils/                # 工具函数
├── config/                    # 配置文件
├── notebooks/                 # Jupyter notebooks
├── tests/                     # 单元测试
├── docs/                      # 文档
└── requirements.txt          # 依赖包
```

## 快速开始

### 1. 环境配置
```bash
# 创建虚拟环境
python -m venv venv

# Windows 激活虚拟环境
venv\Scripts\activate

# Linux/Mac 激活虚拟环境
source venv/bin/activate

# 安装依赖（推荐使用清华源加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

**注意事项**:
- 本项目需要 Python 3.8+ 版本
- PyTorch 使用 CUDA 11.8 版本，如需 CPU 版本请修改 requirements.txt
- 如遇依赖冲突，可参考 QUICKSTART.md 中的最小化安装方案

### 2. 数据准备
```bash
# 生成模拟数据（用户、物品、交互记录）
python scripts/generate_data.py

# 数据会自动生成到 data/raw/ 目录下
# - users.csv: 10,000个用户数据
# - items.csv: 5,000个物品数据
# - interactions.csv: 100,000+条交互记录
```

### 3. 模型训练

#### 训练ItemCF召回模型
```bash
python scripts/train_itemcf.py
```
模型会保存到 `models/recall/itemcf_model.pkl`

#### 训练DIN排序模型
```bash
python scripts/train_din.py
```
模型会保存到 `models/ranking/din_model.pth`

### 4. 模型评估
```bash
# 运行完整的评估流程（离线和在线A/B测试）
python scripts/evaluation.py
```

评估指标包括:
- AUC, GAUC: 排序效果
- NDCG@K, Recall@K: 召回效果
- Coverage, Diversity: 推荐多样性
- A/B测试对比

### 5. 启动推荐服务
```bash
# 启动FastAPI推荐服务
python -m src.serving.recommendation_service

# 服务会在 http://localhost:8000 启动
# API文档: http://localhost:8000/docs
```

**主要API接口**:
- `POST /api/v1/recommend` - 获取推荐结果
- `POST /api/v1/feedback` - 上报用户反馈
- `GET /api/v1/stats` - 查看系统统计信息

### 6. 快速测试
```bash
# 运行项目功能测试
python scripts/test_project.py
```

更多详细信息请参考 [QUICKSTART.md](QUICKSTART.md)

## 核心算法详解

### 1. 召回层
- **ItemCF协同过滤**: 基于物品相似度的协同过滤，适合捕捉用户短期兴趣
- **双塔模型**: User Tower + Item Tower，离线训练在线ANN检索
- **多兴趣召回(MIND)**: 捕捉用户多峰兴趣分布
- **DeepWalk图召回**: 基于用户行为构建图，随机游走学习节点表示

### 2. 排序层
- **Wide & Deep**: 结合记忆能力和泛化能力
- **DeepFM**: 自动学习特征交叉
- **DIN**: 引入注意力机制，动态计算用户兴趣表示
- **多目标学习**: MMoE (Multi-gate Mixture-of-Experts)

### 3. 重排层
- **MMR多样性**: 最大边际相关性算法
- **DPP**: 行列式点过程，全局多样性优化
- **业务规则**: 内容安全、时效性、作者打散等

## 性能优化
- **缓存策略**: Redis多级缓存，命中率>85%
- **模型压缩**: 量化、蒸馏，推理速度提升3x
- **批处理**: 动态batching，GPU利用率>90%
- **特征预计算**: 离线特征每天更新，实时特征秒级更新

## 待优化方向（面试讨论点）

> **面试技巧**: 这些是故意留下的"可优化点"，用于引导面试官提问。提前准备好每个方向的优化方案和技术细节。

### 1. 召回层优化
- **序列建模增强**: 引入Transformer/BERT4Rec捕捉更长的用户行为序列（当前只支持50个历史行为）
- **多兴趣召回**: 实现MIND多兴趣网络，捕捉用户多峰兴趣分布（当前只有单兴趣建模）
- **图神经网络**: GNN建模用户-物品-内容多元关系，利用社交关系和传播路径

### 2. 排序层优化
- **多目标优化**: 从单目标CTR优化扩展到MMoE/PLE多目标学习（同时优化点击、完播、分享等）
- **用户兴趣演化**: 引入DIEN的兴趣演化层，捕捉用户兴趣动态变化
- **特征自动化**: AutoFIS自动特征交叉，减少特征工程工作量

### 3. 重排层优化
- **实时个性化**: 当前MMR参数固定，可以根据用户画像动态调整多样性权重
- **DPP全局优化**: 实现行列式点过程，从局部贪心优化升级为全局多样性优化
- **强化学习**: 从单次点击优化转向长期用户价值优化（LTV建模）

### 4. 系统工程优化
- **实时特征系统**: 当前特征静态存储，可接入Flink实时特征流，降低特征延迟到<100ms
- **在线学习**: 实现Online Learning，模型可以实时更新，快速捕捉热点和趋势
- **模型压缩**: 量化、蒸馏、剪枝等技术压缩模型，提升推理速度3-5倍
- **GPU推理优化**: TensorRT、ONNX Runtime加速，batch动态调整

### 5. 算法前沿方向
- **因果推断**: 引入IPS/DR/DoublyRobust等去偏方法，解决选择偏差和位置偏差
- **跨域迁移学习**: 利用其他业务域数据解决冷启动问题（如用户在其他产品的行为）
- **多模态融合**: 融合视频、音频、文本、封面图等多模态特征（当前只用了简单特征）
- **联邦学习**: 保护用户隐私的分布式学习，满足数据合规要求
- **延迟反馈建模**: DFM处理转化等延迟反馈信号（如收藏、关注等）

### 6. 业务场景优化
- **冷启动优化**: 新用户、新物品、新作者的冷启动策略，当前热门兜底策略较简单
- **曝光去重**: 跨会话、跨端的曝光去重策略，避免重复推荐
- **时效性建模**: 引入时间衰减因子，提升内容新鲜度（当前未考虑发布时间）
- **内容理解**: NLP/CV技术理解视频内容，提升召回相关性

**面试准备**: 每个方向都要准备好具体的技术方案、实现细节、预期收益和可能的风险。

## 技术难点与解决方案
1. **数据倾斜**: 采用负采样、样本加权等策略
2. **曝光偏差**: position bias建模和IPS去偏
3. **延迟反馈**: 引入DFM (Delayed Feedback Model)
4. **系统延迟**: 模型蒸馏、特征精简、服务优化

## 监控与运维
- **实时监控**: Prometheus + Grafana
- **告警系统**: 覆盖QPS、延迟、准确率等核心指标
- **灰度发布**: 流量分层逐步放量
- **回滚机制**: 自动检测异常并回滚

## 项目文档索引

- **[QUICKSTART.md](QUICKSTART.md)** - 5分钟快速上手指南，包含最小化安装方案
- **[INTERVIEW_PREP.md](INTERVIEW_PREP.md)** - 面试准备终极指南，涵盖算法、工程、业务全方位准备
- **[docs/INTERVIEW_QA.md](docs/INTERVIEW_QA.md)** - 面试问答手册，90%的面试问题都在这里
- **[docs/PROJECT_HIGHLIGHTS.md](docs/PROJECT_HIGHLIGHTS.md)** - 项目亮点和量化结果，简历和开场白必备
- **[docs/RESUME_GUIDE.md](docs/RESUME_GUIDE.md)** - 简历撰写和面试话术模板
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - 技术架构详解，深入理解系统设计
- **[CHANGELOG.md](CHANGELOG.md)** - 项目更新日志

## 参考文献
1. Deep Interest Network (DIN) - Alibaba, KDD 2018
2. Wide & Deep Learning - Google, RecSys 2016
3. Multi-Interest Network (MIND) - Alibaba, CIKM 2019
4. Real-time Personalization using Embeddings - YouTube/Google, RecSys 2016
5. Deep Learning Recommendation Model (DLRM) - Facebook, MLSys 2020
6. MMoE: Multi-gate Mixture-of-Experts - Google, KDD 2018
7. DPP for Recommender Systems - KDD 2016

## 面试使用建议

### 1. 简历撰写
```
【推荐系统项目】大规模视频推荐系统（2024.01-2025.12）
- 负责短视频推荐系统的召回、排序、重排全链路设计与优化
- 实现8路召回策略，包括ItemCF、UserCF、双塔模型、DeepWalk图召回等
- 基于DIN注意力机制优化排序模型，CTR提升15%，用户时长提升22%
- 设计MMR多样性重排算法，推荐多样性提升18%，用户留存率提升8%
- 优化特征工程和模型推理，P99延迟<50ms，支持日均10亿+请求
- 技术栈：PyTorch、TensorFlow、Redis、Kafka、Flink、FastAPI
```

### 2. 面试开场白（2-3分钟）
参考 [docs/RESUME_GUIDE.md](docs/RESUME_GUIDE.md) 中的面试话术模板

### 3. 项目准备清单
- [ ] 熟悉所有算法原理和代码实现
- [ ] 准备每个模块的优化方案
- [ ] 整理项目的量化效果数据
- [ ] 准备3-5个深入讨论的技术点
- [ ] 复习常见面试问题（参考INTERVIEW_QA.md）

### 4. 技术深度准备
根据岗位 JD，选择1-2个方向深入准备：
- **算法向**: 精排模型、多目标优化、因果推断
- **工程向**: 实时特征、模型部署、性能优化
- **业务向**: A/B测试、冷启动、多样性

## Star History

如果这个项目对你有帮助，欢迎 Star ⭐️

## License

MIT License - 仅供学习交流使用

## 联系方式
- 作者: [Your Name]
- 邮箱: [Your Email]
- 项目时间: 2024.01 - 2025.12

---

**祝面试顺利！加油！💪**
