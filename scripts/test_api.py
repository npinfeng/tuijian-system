"""
测试推荐服务API
"""

import requests
import json
import time

# API基础地址
BASE_URL = "http://127.0.0.1:8000"


def test_health_check():
    """测试健康检查接口"""
    print("\n" + "="*60)
    print("测试 1: 健康检查")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    print("✓ 健康检查通过")


def test_recommend():
    """测试推荐接口"""
    print("\n" + "="*60)
    print("测试 2: 获取推荐结果")
    print("="*60)
    
    # 请求推荐
    request_data = {
        "user_id": 1,
        "scene": "feed",
        "num": 10,
        "context": {
            "device": "ios",
            "city": "beijing"
        }
    }
    
    print(f"请求参数: {json.dumps(request_data, indent=2, ensure_ascii=False)}")
    
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/api/v1/recommend",
        json=request_data
    )
    cost_time = int((time.time() - start_time) * 1000)
    
    print(f"\n状态码: {response.status_code}")
    print(f"实际响应时间: {cost_time}ms")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n推荐结果:")
        print(f"- 用户ID: {result['user_id']}")
        print(f"- 推荐数量: {len(result['items'])}")
        print(f"- 服务响应时间: {result['cost_time']}ms")
        print(f"- 追踪ID: {result['trace_id']}")
        
        print(f"\n推荐物品列表:")
        for i, item in enumerate(result['items'][:5], 1):  # 只显示前5个
            print(f"  {i}. Item {item['item_id']}, Score: {item['score']:.4f}, Reason: {item['reason']}")
        
        if len(result['items']) > 5:
            print(f"  ... 还有 {len(result['items']) - 5} 个物品")
        
        assert len(result['items']) == 10, "推荐数量不符合预期"
        assert result['user_id'] == 1, "用户ID不匹配"
        print("\n✓ 推荐接口测试通过")
    else:
        print(f"错误: {response.text}")
        raise Exception("推荐接口调用失败")


def test_feedback():
    """测试反馈接口"""
    print("\n" + "="*60)
    print("测试 3: 用户反馈")
    print("="*60)
    
    # 测试点击反馈
    response = requests.post(
        f"{BASE_URL}/api/v1/feedback",
        params={
            "user_id": 1,
            "item_id": 100,
            "action": "click"
        }
    )
    
    print(f"点击反馈 - 状态码: {response.status_code}")
    print(f"点击反馈 - 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    # 测试曝光反馈
    response = requests.post(
        f"{BASE_URL}/api/v1/feedback",
        params={
            "user_id": 1,
            "item_id": 101,
            "action": "impression"
        }
    )
    
    print(f"曝光反馈 - 状态码: {response.status_code}")
    print(f"曝光反馈 - 响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
    
    assert response.status_code == 200
    print("\n✓ 反馈接口测试通过")


def test_stats():
    """测试统计信息接口"""
    print("\n" + "="*60)
    print("测试 4: 系统统计信息")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/api/v1/stats")
    print(f"状态码: {response.status_code}")
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\n系统统计信息:")
        print(f"- 总用户数: {stats['total_users']}")
        print(f"- 总物品数: {stats['total_items']}")
        print(f"- 日均请求: {stats['daily_requests']}")
        print(f"- 平均响应时间: {stats['avg_response_time']}")
        
        print(f"\n模型加载状态:")
        for model_name, loaded in stats['models'].items():
            status = "✓ 已加载" if loaded else "✗ 未加载"
            print(f"  - {model_name}: {status}")
        
        print("\n✓ 统计信息接口测试通过")


def test_multiple_users():
    """测试多个用户的推荐"""
    print("\n" + "="*60)
    print("测试 5: 多用户推荐差异性")
    print("="*60)

    user_ids = [i for i in range(1, 101)]
    results = {}
    
    for user_id in user_ids:
        response = requests.post(
            f"{BASE_URL}/api/v1/recommend",
            json={"user_id": user_id, "num": 5}
        )
        
        if response.status_code == 200:
            result = response.json()
            item_ids = [item['item_id'] for item in result['items']]
            results[user_id] = item_ids
            print(f"用户 {user_id}: {item_ids[:3]}... (响应时间: {result['cost_time']}ms)")
    
    # 检查推荐结果的差异性
    all_items = set()
    for items in results.values():
        all_items.update(items)
    
    print(f"\n推荐结果分析:")
    print(f"- 测试用户数: {len(user_ids)}")
    print(f"- 不同物品总数: {len(all_items)}")
    print(f"- 平均每用户推荐: {sum(len(items) for items in results.values()) / len(results):.1f}个")
    
    print("\n✓ 多用户推荐测试通过")


def test_performance():
    """测试性能"""
    print("\n" + "="*60)
    print("测试 6: 性能测试 (100次请求)")
    print("="*60)
    
    num_requests = 100
    response_times = []
    
    print(f"发送 {num_requests} 个推荐请求...")
    
    for i in range(num_requests):
        user_id = i % 100 + 1  # 循环使用用户ID
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/api/v1/recommend",
            json={"user_id": user_id, "num": 10}
        )
        
        cost_time = (time.time() - start_time) * 1000
        response_times.append(cost_time)
        
        if (i + 1) % 20 == 0:
            print(f"  已完成 {i + 1}/{num_requests} 个请求")
    
    # 计算性能指标
    avg_time = sum(response_times) / len(response_times)
    p50 = sorted(response_times)[int(len(response_times) * 0.5)]
    p95 = sorted(response_times)[int(len(response_times) * 0.95)]
    p99 = sorted(response_times)[int(len(response_times) * 0.99)]
    max_time = max(response_times)
    min_time = min(response_times)
    
    print(f"\n性能测试结果:")
    print(f"- 总请求数: {num_requests}")
    print(f"- 平均响应时间: {avg_time:.2f}ms")
    print(f"- P50响应时间: {p50:.2f}ms")
    print(f"- P95响应时间: {p95:.2f}ms")
    print(f"- P99响应时间: {p99:.2f}ms")
    print(f"- 最大响应时间: {max_time:.2f}ms")
    print(f"- 最小响应时间: {min_time:.2f}ms")
    
    # 判断性能是否达标
    if p99 < 100:
        print(f"\n✓ 性能优秀! P99 < 100ms")
    elif p99 < 200:
        print(f"\n✓ 性能良好! P99 < 200ms")
    else:
        print(f"\n⚠ 性能需要优化, P99 = {p99:.2f}ms")
    
    print("\n✓ 性能测试完成")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("推荐系统API自动化测试")
    print("="*60)
    print(f"API地址: {BASE_URL}")
    print("请确保推荐服务已启动 (python -m src.serving.recommendation_service)")
    print("="*60)
    
    try:
        # 依次运行测试
        test_health_check()
        test_recommend()
        test_feedback()
        test_stats()
        test_multiple_users()
        test_performance()
        
        # 测试总结
        print("\n" + "="*60)
        print("✓ 所有测试通过!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ 错误: 无法连接到推荐服务")
        print("请先启动服务: python -m src.serving.recommendation_service")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
