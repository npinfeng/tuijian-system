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


from src.data.kuairand_loader import load_kuairand_splits


def main():
    """主训练流程"""
    
    print("="*50)
    print("训练ItemCF模型 (KuaiRand-1K)")
    print("="*50)
    
    # 1. 加载数据
    print("\n加载 KuaiRand-1K 交互数据...")
    train_log, val_log, test_log = load_kuairand_splits(
        use_early_log_as_train=True,
        val_ratio=0.1,
        max_train_rows=500000,
        verbose=True
    )
    
    # 只使用正样本（点击数据）进行协同过滤训练
    positive_interactions = train_log[train_log['is_click'] == 1]
    print(f"训练集正样本数量: {len(positive_interactions)}")
    
    # 2. 创建模型
    model = ItemCF(
        similarity_type='cosine',
        top_k_similar=100
    )
    
    # 3. 训练模型
    model.fit(positive_interactions)
    
    # 4. 保存模型
    Path('models').mkdir(exist_ok=True)
    model.save('models/itemcf_kuairand.pkl')
    
    # 5. 测试推荐
    print("\n测试推荐...")
    # 从测试集中取一个真实用户
    test_user_id = test_log['user_id'].iloc[0]
    recommendations = model.recommend(test_user_id, n=10)
    
    print(f"\n为用户 {test_user_id} 推荐的物品:")
    for i, (item_id, score) in enumerate(recommendations, 1):
        print(f"{i}. Item {item_id}, Score: {score:.4f}")
    
    print("\n训练完成！模型保存为 models/itemcf_kuairand.pkl")


if __name__ == '__main__':
    main()
