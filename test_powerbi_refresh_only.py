from agent_entree_finopsrag import trigger_powerbi_refresh_if_needed, force_aws_profile

force_aws_profile()

selection = {
    "mode": "days",
    "block": "C",
    "week": 9,
    "days": 3,
    "period": "Monday to Wednesday",
}

print("Starting targeted Power BI refresh test...")
result = trigger_powerbi_refresh_if_needed(selection)
print("Refresh helper result:", result)