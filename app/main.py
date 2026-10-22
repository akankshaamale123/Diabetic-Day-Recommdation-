# app/main.py — generate chart files (each chart separate)
import pandas as pd
from storage import db
from pipeline.glucose_analysis import analyze_meal_responses, summarize_patterns
from pipeline.glucose_charts import save_all, compare_stats, default_login_date

db.init_db()

if __name__ == "__main__":
    days = db.available_days()
    if not days:
        raise SystemExit("No data yet — run scripts/load_cgmacros.py first.")

    login = default_login_date()
    last_day = pd.Timestamp(days[-1])
    end = last_day + pd.Timedelta(days=1)

    paths = save_all(last_day, login, end)
    for name, p in paths.items():
        print(f"{name:28s} -> {p}")

    print("\n", compare_stats(last_day, login, end).to_string())
    print(analyze_meal_responses(login, end).head(20).to_string(index=False))
    print(summarize_patterns(login, end))