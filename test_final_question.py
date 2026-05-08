from cfo_finops_athena_rag_final import run_finops_cfo_pipeline

selection = {
    "mode": "days",
    "block": "C",
    "week": 9,
    "days": 3,
    "period": "Monday to Wednesday",
}

question = "what was the service with the most cost during that period of time"

result = run_finops_cfo_pipeline(
    selection=selection,
    question=question,
    enable_audio=False,
    auto_open_audio=False,
)

print("\n==============================")
print("ROUTE")
print("==============================")
print(result["route"])

print("\n==============================")
print("ANSWER")
print("==============================")
print(result["answer"])