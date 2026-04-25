"""
排序模块
包含粗排和精排模型:
  - PreRanking  : 轻量级逻辑回归粗排（500 -> 200）
  - DIN         : Deep Interest Network 精排（注意力机制，TF）
  - WideDeep    : Wide & Deep 排序模型（PyTorch）
  - DeepFM      : DeepFM 排序模型（PyTorch）
  - MMoE        : Multi-gate Mixture-of-Experts 多目标模型（PyTorch）
"""

from src.ranking.pre_ranking import PreRanking, PreRankingFeatureExtractor
from src.ranking.wide_deep import WideDeep, WideDeepTrainer
from src.ranking.deepfm import DeepFM, DeepFMTrainer, FMLayer
from src.ranking.mmoe import MMoE, MMoETrainer, ExpertNetwork, GatingNetwork

# DIN 依赖 TensorFlow，单独 try-import 防止环境中无 TF 时整个模块报错
try:
    from src.ranking.din_model import DIN, DINTrainer
    __all__ = [
        'PreRanking', 'PreRankingFeatureExtractor',
        'DIN', 'DINTrainer',
    ]
except ImportError:
    __all__ = [
        'PreRanking', 'PreRankingFeatureExtractor',
    ]
