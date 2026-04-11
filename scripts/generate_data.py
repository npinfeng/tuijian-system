"""
数据生成脚本
用于生成推荐系统的模拟数据
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from pathlib import Path


class DataGenerator:
    """推荐系统数据生成器"""
    
    def __init__(self, 
                 n_users: int = 10000,
                 n_items: int = 50000,
                 n_categories: int = 20,
                 n_authors: int = 1000):
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        self.n_authors = n_authors
        
        # 创建数据目录
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    def generate_users(self) -> pd.DataFrame:
        """生成用户数据"""
        print("生成用户数据...")
        
        users = pd.DataFrame({
            'user_id': range(self.n_users),
            'age': np.random.randint(18, 60, self.n_users),
            'gender': np.random.choice(['M', 'F'], self.n_users),
            'city': np.random.choice([f'city_{i}' for i in range(50)], self.n_users),
            'device_type': np.random.choice(['iOS', 'Android'], self.n_users, p=[0.4, 0.6]),
            'register_date': [
                (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
                for _ in range(self.n_users)
            ]
        })
        
        users.to_csv('data/raw/users.csv', index=False)
        print(f"用户数据已保存: {len(users)} 条")
        return users
    
    def generate_items(self) -> pd.DataFrame:
        """生成物品数据"""
        print("生成物品数据...")
        
        items = pd.DataFrame({
            'item_id': range(self.n_items),
            'title': [f'video_title_{i}' for i in range(self.n_items)],
            'category': np.random.randint(0, self.n_categories, self.n_items),
            'author_id': np.random.randint(0, self.n_authors, self.n_items),
            'duration': np.random.randint(15, 300, self.n_items),  # 15秒到5分钟
            'publish_time': [
                (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat()
                for _ in range(self.n_items)
            ],
            'tag_list': [
                ','.join([f'tag_{random.randint(0, 100)}' for _ in range(random.randint(1, 5))])
                for _ in range(self.n_items)
            ]
        })
        
        items.to_csv('data/raw/items.csv', index=False)
        print(f"物品数据已保存: {len(items)} 条")
        return items
    
    def generate_interactions(self, n_samples: int = 1000000) -> pd.DataFrame:
        """
        生成用户-物品交互数据
        模拟真实的交互模式
        """
        print(f"生成交互数据 ({n_samples} 条)...")
        
        # 用户活跃度服从幂律分布
        user_probs = np.random.zipf(1.5, self.n_users)
        user_probs = user_probs / user_probs.sum()
        
        # 物品热度服从幂律分布
        item_probs = np.random.zipf(1.3, self.n_items)
        item_probs = item_probs / item_probs.sum()
        
        # 生成交互
        interactions = []
        
        start_time = datetime.now() - timedelta(days=30)
        
        for _ in range(n_samples):
            user_id = np.random.choice(self.n_users, p=user_probs)
            item_id = np.random.choice(self.n_items, p=item_probs)
            
            # 生成时间戳
            timestamp = start_time + timedelta(
                seconds=random.randint(0, 30*24*3600)
            )
            
            # 模拟点击概率（基于位置）
            position = random.randint(0, 20)
            base_ctr = 0.15
            position_bias = 1.0 / (1 + np.log(position + 1))
            is_click = random.random() < (base_ctr * position_bias)
            
            # 如果点击，生成播放相关数据
            if is_click:
                play_duration = random.randint(5, 300)
                is_play = 1
                is_finish = random.random() < 0.35  # 35%完播率
                is_like = random.random() < 0.08    # 8%点赞率
                is_share = random.random() < 0.03   # 3%分享率
            else:
                play_duration = 0
                is_play = 0
                is_finish = 0
                is_like = 0
                is_share = 0
            
            interactions.append({
                'user_id': user_id,
                'item_id': item_id,
                'timestamp': timestamp.isoformat(),
                'position': position,
                'is_click': int(is_click),
                'is_play': is_play,
                'play_duration': play_duration,
                'is_finish': int(is_finish),
                'is_like': int(is_like),
                'is_share': int(is_share),
                'hour': timestamp.hour,
                'weekday': timestamp.weekday()
            })
        
        df = pd.DataFrame(interactions)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        df.to_csv('data/raw/interactions.csv', index=False)
        print(f"交互数据已保存: {len(df)} 条")
        
        # 打印统计信息
        print("\n数据统计:")
        print(f"总交互数: {len(df)}")
        print(f"点击率: {df['is_click'].mean():.2%}")
        print(f"完播率: {df['is_finish'].mean():.2%}")
        print(f"点赞率: {df['is_like'].mean():.2%}")
        print(f"分享率: {df['is_share'].mean():.2%}")
        
        return df
    
    def generate_item_embeddings(self, dim: int = 64) -> pd.DataFrame:
        """
        生成物品embeddings（用于相似度计算）
        实际场景中应该由模型生成
        """
        print("生成物品embeddings...")
        
        embeddings = []
        for item_id in range(self.n_items):
            # 同一类目的物品embedding相似
            category = item_id % self.n_categories
            base_emb = np.random.randn(dim)
            category_emb = np.random.randn(dim) * 0.3
            
            # 混合
            emb = base_emb + category_emb * (category / self.n_categories)
            emb = emb / np.linalg.norm(emb)  # L2归一化
            
            embeddings.append({
                'item_id': item_id,
                'embedding': emb.tolist()
            })
        
        df = pd.DataFrame(embeddings)
        df.to_csv('data/processed/item_embeddings.csv', index=False)
        print(f"Embeddings已保存: {len(df)} 条")
        
        return df
    
    def generate_all(self, n_interactions: int = 1000000):
        """生成所有数据"""
        print("="*50)
        print("开始生成推荐系统数据")
        print("="*50)
        
        # 生成用户数据
        users = self.generate_users()
        
        # 生成物品数据
        items = self.generate_items()
        
        # 生成交互数据
        interactions = self.generate_interactions(n_interactions)
        
        # 生成embeddings
        embeddings = self.generate_item_embeddings()
        
        print("\n" + "="*50)
        print("数据生成完成！")
        print("="*50)
        
        return users, items, interactions, embeddings


def main():
    """主函数"""
    generator = DataGenerator(
        n_users=1000,
        n_items=5000,
        n_categories=20,
        n_authors=1000
    )
    
    generator.generate_all(n_interactions=50000)


if __name__ == '__main__':
    main()
