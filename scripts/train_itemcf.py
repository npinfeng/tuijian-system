"""
训练ItemCF模型
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from pathlib import Path
from src.recall.collaborative_filtering import ItemCF


def main():
    """主训练流程"""
    
    print("="*50)
    print("训练ItemCF模型")
    print("="*50)
    
    # 1. 加载数据
    print("\n加载交互数据...")
    interactions = pd.read_csv('data/raw/interactions.csv')
    
    # 只使用正样本（点击数据）
    positive_interactions = interactions[interactions['is_click'] == 1]
    print(f"正样本数量: {len(positive_interactions)}")
    
    # 2. 创建模型
    model = ItemCF(
        similarity_type='cosine',
        top_k_similar=100
    )
    
    # 3. 训练模型
    model.fit(positive_interactions)
    
    # 4. 保存模型
    Path('models').mkdir(exist_ok=True)
    model.save('models/itemcf_model.pkl')
    
    # 5. 测试推荐
    print("\n测试推荐...")
    test_user_id = 1
    recommendations = model.recommend(test_user_id, n=10)
    
    print(f"\n为用户 {test_user_id} 推荐的物品:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}, Score: {score:.4f}")
    
    print("\n训练完成！")


if __name__ == '__main__':
    main()
