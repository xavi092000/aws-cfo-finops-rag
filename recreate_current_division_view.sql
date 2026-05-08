CREATE OR REPLACE VIEW cfo_finops_db.current_division_view AS
WITH base AS (
    SELECT
        CAST(date AS date) AS date,
        division,
        service,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_pct AS variance_percent
    FROM cfo_finops_db.mart_division_cfo_daily
    WHERE CAST(date AS date) BETWEEN DATE '2025-02-24' AND DATE '2025-02-26'
),

calendar_fixed AS (
    SELECT
        date,

        -- 🔥 RECOMPUTE DAY
        date_format(date, '%W') AS day,

        -- 🔥 RECOMPUTE WEEK
        CASE
            WHEN date BETWEEN DATE '2025-01-01' AND DATE '2025-01-05'
                THEN 'Week 1'
            ELSE CONCAT(
                'Week ',
                CAST(
                    2 + CAST(date_diff('day', DATE '2025-01-06', date) / 7 AS integer)
                    AS varchar
                )
            )
        END AS week,

        -- 🔥 RECOMPUTE MONTH
        CASE
            WHEN date <= DATE '2025-01-26' THEN 'Month 1'
            WHEN date <= DATE '2025-02-23' THEN 'Month 2'
            ELSE 'Month 3'
        END AS month,

        division,
        service,
        actual_cost_usd,
        allocated_budget_usd,
        variance_usd,
        variance_percent

    FROM base
)

SELECT
    date AS period_value,
    *
FROM calendar_fixed