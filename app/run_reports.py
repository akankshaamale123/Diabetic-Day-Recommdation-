# app/run_reports.py  — run from the `app` folder:  python run_reports.py [YYYY-MM-DD]
import sys
import pandas as pd
from storage import db
from pipeline.glucose_charts import save_all, compare_stats, default_login_date
 
db.init_db()
days = db.available_days()
if not days:
    raise SystemExit("No data yet — run scripts/load_cgmacros.py first.")
 
login_date = default_login_date()                    # first day the user has data
day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else days[-1])
end = pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
 
paths = save_all(day, login_date, end)               # every chart = its own HTML file
for name, p in paths.items():
    print(f"{name:28s} -> {p}")
 
print("\n", compare_stats(day, login_date, end).to_string())
 