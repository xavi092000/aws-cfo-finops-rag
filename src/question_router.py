def classify_question(question: str) -> str:
    question = question.lower().strip()

    analytics_keywords = [
        "variance",
        "forecast",
        "trend",
        "trends",
        "total",
        "average",
        "highest",
        "lowest",
        "over budget",
        "under budget",
        "budget used",
        "budget usage",
        "unit economics",
        "roi",
        "break-even",
        "break even",
        "per unit",
        "by division",
        "by product",
        "by project",
        "this week",
        "last week",
        "this month",
        "last month",
        "today",
        "yesterday",
        "how much",
        "which division",
        "which product",
        "which team",
        "what is the variance",
        "what is the budget",
        "what is the spend",
        "spent the most",
        "spent the least",
        "actual",
        "actuals",
        "cost",
        "costs",
        "spend",
        "overspend",
        "underspend",
        "risk",
        "at risk",
        "critical",
        "warning",
        "watch",
        "driver",
        "cost driver",
        "allocation",
        "allocated budget",
        "top 5",
        "top 10",
    ]

    hybrid_keywords = [
        "why",
        "recommend",
        "recommendation",
        "optimize",
        "optimization",
        "justify",
        "justified",
        "what should we do",
        "what do we do",
        "how do we reduce",
        "how can we reduce",
        "how should we respond",
    ]

    document_keywords = [
        "what is finops",
        "finops maturity",
        "best practices",
        "principles",
        "governance",
        "ownership",
        "accountability",
        "habits",
        "how to keep cloud cost down",
        "how can i keep cloud cost down",
        "how do i keep cloud cost down",
        "what are the best ways",
        "what should i do to keep cloud cost down",
    ]

    has_analytics = any(word in question for word in analytics_keywords)
    has_hybrid = any(word in question for word in hybrid_keywords)
    has_document = any(word in question for word in document_keywords)

    if has_document and not has_analytics and not has_hybrid:
        return "document"

    if has_hybrid and has_analytics:
        return "hybrid"

    if has_hybrid and ("budget" in question or "cost" in question or "spend" in question or "risk" in question):
        return "hybrid"

    if has_analytics:
        return "analytics"

    return "document"