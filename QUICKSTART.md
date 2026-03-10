# 快速开始指南

## 项目运行步骤

### 1. 环境准备

#### 1.1 安装Python环境
```powershell
# 确保Python版本 >= 3.8
python --version
```

#### 1.2 创建虚拟环境
```powershell
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 如果遇到权限问题，执行:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### 1.3 安装依赖
```powershell
# 安装所有依赖包
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果遇到TensorFlow安装问题，可以单独安装CPU版本:
pip install tensorflow-cpu==2.13.0
```

### 2. 数据准备

#### 2.1 生成模拟数据
```powershell
# 生成用户、物品、交互数据
python scripts/generate_data.py
```

生成的数据将保存在 `data/raw/` 目录下:
- `users.csv`: 用户数据 (10,000用户)
- `items.csv`: 物品数据 (50,000物品)
- `interactions.csv`: 交互数据 (1,000,000条)

### 3. 模型训练

#### 3.1 训练ItemCF召回模型
```powershell
# 训练基于物品的协同过滤模型
python scripts/train_itemcf.py
```

模型将保存在 `models/itemcf_model.pkl`

#### 3.2 训练DIN排序模型 (可选)
```powershell
# 训练深度兴趣网络模型
python scripts/train_din.py
```

**注意**: DIN模型训练需要较长时间，建议使用GPU。如果只是demo，可以跳过此步骤。

### 4. 模型评估

```powershell
# 运行评估脚本
python scripts/evaluation.py
```

评估指标包括:
- AUC、GAUC
- NDCG@10
- Recall@50
- Diversity、Coverage

### 5. 启动推荐服务

#### 5.1 准备Redis (可选)
如果要使用实时特征，需要启动Redis:

```powershell
# Windows用户可以下载Redis for Windows
# 或使用Docker:
docker run -d -p 6379:6379 redis
```

如果没有Redis，服务会自动降级，不影响基本功能。

#### 5.2 启动服务
```powershell
# 启动推荐服务
python src/serving/recommendation_service.py
```

服务将在 `http://localhost:8000` 启动

#### 5.3 测试API

打开浏览器访问: `http://localhost:8000/docs`

你会看到Swagger UI文档，可以直接测试API。

**测试推荐接口:**
```json
POST http://localhost:8000/api/v1/recommend

{
  "user_id": 123,
  "scene": "feed",
  "num": 10
}
```

**返回示例:**
```json
{
  "user_id": 123,
  "items": [
    {
      "item_id": 1234,
      "score": 0.85,
      "reason": "personalized"
    },
    ...
  ],
  "trace_id": "123_1234567890",
  "cost_time": 45
}
```

### 6. 查看系统状态

```powershell
# 查看服务状态
curl http://localhost:8000/

# 查看统计信息
curl http://localhost:8000/api/v1/stats
```

---

## 常见问题

### Q1: TensorFlow安装失败
**A**: 尝试安装CPU版本:
```powershell
pip install tensorflow-cpu==2.13.0
```

### Q2: 推荐服务启动失败，提示模型未找到
**A**: 确保先运行了模型训练脚本:
```powershell
python scripts/train_itemcf.py
```

### Q3: Redis连接失败
**A**: 实时特征是可选的，服务会自动降级。如需使用Redis:
- Windows: 下载 [Redis for Windows](https://github.com/microsoftarchive/redis/releases)
- 或使用Docker: `docker run -d -p 6379:6379 redis`

### Q4: 内存不足
**A**: 可以减少数据量:
```python
# 在 scripts/generate_data.py 中修改
generator = DataGenerator(
    n_users=1000,      # 减少到1000
    n_items=5000,      # 减少到5000
    ...
)
generator.generate_all(n_interactions=10000)  # 减少到10000
```

### Q5: 如何只运行最小demo?
**A**: 最小化步骤:
```powershell
# 1. 安装依赖
pip install fastapi uvicorn pandas numpy scikit-learn

# 2. 生成数据
python scripts/generate_data.py

# 3. 训练ItemCF
python scripts/train_itemcf.py

# 4. 启动服务
python src/serving/recommendation_service.py
```

---

## 项目结构说明

```
tuijian-system/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后数据
├── models/                  # 模型存储
│   └── itemcf_model.pkl   # ItemCF模型
├── src/                     # 源代码
│   ├── recall/             # 召回算法
│   ├── ranking/            # 排序算法
│   ├── rerank/             # 重排算法
│   ├── features/           # 特征工程
│   ├── serving/            # 在线服务
│   └── utils/              # 工具函数
├── scripts/                 # 脚本
│   ├── generate_data.py   # 数据生成
│   ├── train_itemcf.py    # 训练ItemCF
│   ├── train_din.py       # 训练DIN
│   └── evaluation.py      # 模型评估
├── config/                  # 配置文件
│   └── config.yaml        # 主配置
├── docs/                    # 文档
│   ├── ARCHITECTURE.md    # 架构文档
│   ├── INTERVIEW_QA.md    # 面试问答
│   └── PROJECT_HIGHLIGHTS.md  # 项目亮点
└── README.md               # 项目说明
```

---

## 下一步

1. **代码阅读**: 从 `src/serving/recommendation_service.py` 开始
2. **算法理解**: 阅读 `docs/ARCHITECTURE.md`
3. **面试准备**: 阅读 `docs/INTERVIEW_QA.md`
4. **扩展功能**: 实现双塔模型、DIN模型的实际应用

---

## 联系方式

如有问题，欢迎提Issue或联系作者。

祝你面试顺利！🚀
