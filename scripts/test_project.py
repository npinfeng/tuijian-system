"""
快速测试脚本
验证项目基本功能
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试模块导入"""
    print("测试模块导入...")
    
    try:
        from src.utils.common import load_config
        print("✓ utils.common")
    except Exception as e:
        print(f"✗ utils.common: {e}")
        return False
    
    try:
        from src.recall.collaborative_filtering import ItemCF, UserCF
        print("✓ recall.collaborative_filtering")
    except Exception as e:
        print(f"✗ recall.collaborative_filtering: {e}")
        return False
    
    try:
        from src.ranking.din_model import DIN
        print("✓ ranking.din_model")
    except Exception as e:
        print(f"✗ ranking.din_model: {e}")
        return False
    
    try:
        from src.rerank.diversity_rerank import MMRReranker, DiversityReranker
        print("✓ rerank.diversity_rerank")
    except Exception as e:
        print(f"✗ rerank.diversity_rerank: {e}")
        return False
    
    try:
        from src.features.feature_engineering import FeatureEngineer
        print("✓ features.feature_engineering")
    except Exception as e:
        print(f"✗ features.feature_engineering: {e}")
        return False
    
    print("\n所有模块导入成功！✓\n")
    return True


def test_data_generation():
    """测试数据生成"""
    print("测试数据生成...")
    
    try:
        from scripts.generate_data import DataGenerator
        
        # 生成小量数据用于测试
        generator = DataGenerator(
            n_users=100,
            n_items=500,
            n_categories=5,
            n_authors=20
        )
        
        users = generator.generate_users()
        items = generator.generate_items()
        interactions = generator.generate_interactions(1000)
        
        print(f"✓ 生成用户数据: {len(users)} 条")
        print(f"✓ 生成物品数据: {len(items)} 条")
        print(f"✓ 生成交互数据: {len(interactions)} 条")
        
        print("\n数据生成成功！✓\n")
        return True
        
    except Exception as e:
        print(f"✗ 数据生成失败: {e}")
        return False


def test_itemcf():
    """测试ItemCF模型"""
    print("测试ItemCF模型...")
    
    try:
        import pandas as pd
        from src.recall.collaborative_filtering import ItemCF
        
        # 创建测试数据
        test_data = pd.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            'item_id': [1, 2, 1, 3, 2, 3, 1, 4, 2, 4]
        })
        
        # 训练模型
        model = ItemCF(similarity_type='cosine', top_k_similar=3)
        model.fit(test_data)
        
        # 测试推荐
        recommendations = model.recommend(user_id=1, n=3)
        
        print(f"✓ ItemCF模型训练成功")
        print(f"✓ 为用户1推荐: {recommendations}")
        
        print("\nItemCF测试成功！✓\n")
        return True
        
    except Exception as e:
        print(f"✗ ItemCF测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmr():
    """测试MMR重排"""
    print("测试MMR重排算法...")
    
    try:
        import pandas as pd
        import numpy as np
        from src.rerank.diversity_rerank import MMRReranker
        
        # 创建测试数据
        items = [1, 2, 3, 4, 5]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        
        item_features = pd.DataFrame({
            'item_id': items,
            'embedding': [np.random.randn(10) for _ in items]
        })
        
        # 重排
        reranker = MMRReranker(lambda_param=0.5)
        reranked = reranker.rerank(items, scores, item_features, top_k=3)
        
        print(f"✓ MMR重排成功: {reranked}")
        
        print("\nMMR测试成功！✓\n")
        return True
        
    except Exception as e:
        print(f"✗ MMR测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试流程"""
    print("="*60)
    print("推荐系统项目 - 快速测试")
    print("="*60)
    print()
    
    # 测试模块导入
    if not test_imports():
        print("\n❌ 模块导入失败，请检查依赖安装")
        return
    
    # 测试数据生成
    if not test_data_generation():
        print("\n❌ 数据生成失败")
        return
    
    # 测试ItemCF
    if not test_itemcf():
        print("\n❌ ItemCF测试失败")
        return
    
    # 测试MMR
    if not test_mmr():
        print("\n❌ MMR测试失败")
        return
    
    print("="*60)
    print("✅ 所有测试通过！项目基本功能正常")
    print("="*60)
    print()
    print("下一步:")
    print("1. 运行 python scripts/generate_data.py 生成完整数据")
    print("2. 运行 python scripts/train_itemcf.py 训练模型")
    print("3. 运行 python src/serving/recommendation_service.py 启动服务")
    print()


if __name__ == '__main__':
    main()
