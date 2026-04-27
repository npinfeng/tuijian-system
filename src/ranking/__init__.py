"""
排序模块
包含粗排和精排模型:
  - PreRanking  : 轻量级逻辑回归粗排（500 -> 200）
  - DIN         : Deep Interest Network 精排（注意力机制，PyTorch）
  - WideDeep    : Wide & Deep 排序模型（PyTorch）
  - DeepFM      : DeepFM 排序模型（PyTorch）
  - MMoE        : Multi-gate Mixture-of-Experts 多目标模型（PyTorch）
"""

from src.ranking.pre_ranking import PreRanking, PreRankingFeatureExtractor
from src.ranking.din_model import DIN, DINTrainer

__all__ = [
    'PreRanking', 'PreRankingFeatureExtractor',
    'DIN', 'DINTrainer',
]

# 可选模块：如果已实现则导入
try:
    from src.ranking.wide_deep import WideDeep, WideDeepTrainer
    __all__ += ['WideDeep', 'WideDeepTrainer']
except ImportError:
    pass

try:
    from src.ranking.deepfm import DeepFM, DeepFMTrainer, FMLayer
    __all__ += ['DeepFM', 'DeepFMTrainer', 'FMLayer']
except ImportError:
    pass

try:
    from src.ranking.mmoe import MMoE, MMoETrainer, ExpertNetwork, GatingNetwork
    __all__ += ['MMoE', 'MMoETrainer', 'ExpertNetwork', 'GatingNetwork']
except ImportError:
    pass
