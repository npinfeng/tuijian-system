import time
import cProfile
import pstats
import io
from src.serving.recommendation_service import RecommendationService

def run_benchmark(service, num_requests=1000):
    start_time = time.time()
    for i in range(num_requests):
        user_id = i % 100 + 1
        service.recommend(user_id=user_id, scene="feed", num=10)
    end_time = time.time()
    
    total_time = end_time - start_time
    qps = num_requests / total_time
    print(f"\n--- Benchmark Results ---")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"QPS: {qps:.2f} req/s")
    print(f"Average Latency: {(total_time / num_requests * 1000):.2f} ms")


def main():
    print("Initializing service...")
    service = RecommendationService()
    
    print("Warming up...")
    run_benchmark(service, num_requests=10)
    
    print("\nStarting Main Benchmark & Profiling...")
    pr = cProfile.Profile()
    pr.enable()
    run_benchmark(service, num_requests=500)
    pr.disable()
    
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # print top 30
    
    print("\n--- Profiling Output (Top 30 by cumulative time) ---")
    print(s.getvalue())

if __name__ == "__main__":
    main()
