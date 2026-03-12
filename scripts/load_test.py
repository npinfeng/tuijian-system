import asyncio
import httpx
import time
import statistics
import logging

# Disable httpx access logs etc to keep console clean
logging.getLogger("httpx").setLevel(logging.WARNING)

URL = "http://127.0.0.1:8000/api/v1/recommend"
CONCURRENCY = 50
TOTAL_REQUESTS = 500

async def make_request(client, user_id):
    payload = {
        "user_id": user_id,
        "scene": "feed",
        "num": 10
    }
    start = time.perf_counter()
    try:
        response = await client.post(URL, json=payload, timeout=20.0)
        success = response.status_code == 200
        if not success:
            print(f"Request failed with status: {response.status_code}, body: {response.text}")
    except Exception as e:
        success = False
        print(f"Request failed with error: {e}")
    end = time.perf_counter()
    return success, (end - start) * 1000  # ms

async def worker(client, queue, results):
    while True:
        try:
            user_id = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        
        success, latency = await make_request(client, user_id)
        results.append((success, latency))
        queue.task_done()

async def main():
    print(f"Starting load test...")
    print(f"URL: {URL}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Total Requests: {TOTAL_REQUESTS}")
    
    queue = asyncio.Queue()
    for i in range(TOTAL_REQUESTS):
        queue.put_nowait(i % 1000 + 1)
        
    results = []
    start_time = time.time()
    
    limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = []
        for _ in range(CONCURRENCY):
            task = asyncio.create_task(worker(client, queue, results))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    end_time = time.time()
    total_time = end_time - start_time
    
    success_count = sum(1 for r in results if r[0])
    fail_count = len(results) - success_count
    latencies = [r[1] for r in results if r[0]]
    
    print("\n--- Results ---")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Total Requests: {len(results)}")
    print(f"Successful Requests: {success_count}")
    print(f"Failed Requests: {fail_count}")
    print(f"Requests per second (QPS): {len(results) / total_time:.2f}")
    
    if latencies:
        print(f"\n--- Latency (ms) ---")
        print(f"Min: {min(latencies):.2f}")
        print(f"Max: {max(latencies):.2f}")
        print(f"Avg: {sum(latencies)/len(latencies):.2f}")
        print(f"P50: {statistics.median(latencies):.2f}")
        try:
            # For python 3.8+
            print(f"P95: {statistics.quantiles(latencies, n=100)[94]:.2f}")
            print(f"P99: {statistics.quantiles(latencies, n=100)[98]:.2f}")
        except AttributeError:
            # Fallback if quantiles not available
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            if p95_idx < len(sorted_latencies):
                print(f"P95: {sorted_latencies[p95_idx]:.2f}")
            if p99_idx < len(sorted_latencies):
                print(f"P99: {sorted_latencies[p99_idx]:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
