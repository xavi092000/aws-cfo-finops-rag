from cfo_finops_athena_rag_final import handle_rag_failure

analytical_answer = """The top 3 highest-cost services during the selected period were:
- Bedrock: $762.80
- CloudWatch: $257.29
- Redshift: $50.22"""

general_answer = """Hybrid CFO Answer

1. Analytical result

The top 3 highest-cost services during the selected period were:
- Bedrock: $762.80
- CloudWatch: $257.29
- Redshift: $50.22

2. FinOps interpretation

The retrieved documents do not provide enough specific guidance for this question.
However, based on the selected-period analysis, the company should focus on practical FinOps controls.

Recommended actions:
- Review the top cost drivers for the selected period.
- Improve cost allocation tags by product, team, environment, and owner.

3. CFO priority

Start with the largest variance drivers, then assign clear ownership and corrective actions.
"""

clean_general_answer = (
    general_answer
    .replace("1. Executive answer", "Executive answer")
    .replace("2. Key insights", "Key insights")
    .replace("3. CFO actions", "CFO actions")
)

if "FinOps interpretation" in general_answer:
    clean_general_answer = general_answer.split("FinOps interpretation")[-1]
else:
    clean_general_answer = general_answer

clean_general_answer = (
    clean_general_answer
    .replace("Hybrid CFO Answer", "")
    .replace("3. CFO priority", "")
    .replace("Start with the largest variance drivers, then assign clear ownership and corrective actions.", "")
    .strip()
)

combined_answer = (
    "Hybrid CFO Answer\n\n"
    "1. Analytical Answer restricted to the selected period:\n"
    f"{analytical_answer}\n\n"
    "2. FinOps Interpretation linked to the analytical result:\n"
    f"{clean_general_answer}\n\n"
    "3. CFO Priority:\n"
    "Use the analytical result first, then apply the FinOps interpretation to guide action."
)

print(combined_answer)