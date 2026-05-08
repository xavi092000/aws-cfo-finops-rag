import time
from observability import log_event, check_latency_slo

start = time.time()

time.sleep(1)

latency = round(time.time() - start, 3)
slo_status = check_latency_slo(latency)

log_event({
    "component": "test",
    "query_type": "hybrid",
    "status": "success",
    "latency_seconds": latency,
    "slo_latency_status": slo_status,
    "fallback_used": True
})

print("Observability test completed.")
print(f"Latency: {latency}s")
print(f"SLO status: {slo_status}")
print("Log file: outputs/pipeline_events.jsonl")