"""
重排模块
  - MMRReranker      : MMR 最大边际相关性（贪心局部最优，纯 Python/numpy）
  - DiversityReranker: 综合多样性重排（类目、作者、时效性，纯 Python/numpy）
  - DPPReranker      : 行列式点过程（贪心全局最优，纯 Python/numpy）
  - HybridReranker   : 自适应混合重排器（根据候选集大小自动选择算法）
"""

# ---- 无外部重依赖，始终可用 ----
from src.rerank.diversity_rerank import MMRReranker, DiversityReranker

try:
    from src.rerank.dpp_rerank import DPPReranker, HybridReranker
except ImportError:
    DPPReranker = None
    HybridReranker = None
