# app/run_reports.py  — run from the `app` folder:  python run_reports.py [YYYY-MM-DD]
import logging
import sys

import pandas as pd

from storage import db
from pipeline.glucose_charts import save_all, compare_stats, default_login_date

logger = logging.getLogger(__name__)


def main() -> int:
    logger.info("run_reports.py: starting")

    db.init_db()
    days = db.available_days()

    if not days:
        logger.error("run_reports.py: no data in database — run "
                     "scripts/load_cgmacros.py first")
        return 1

    logger.info("run_reports.py: %d day(s) available (%s → %s)",
                len(days), days[0], days[-1])

    try:
        login_date = default_login_date()
    except Exception:
        logger.exception("run_reports.py: could not resolve default login date")
        return 1

    day = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else days[-1])
    end = pd.Timestamp(days[-1]) + pd.Timedelta(days=1)
    logger.info("run_reports.py: selected day=%s, login_date=%s, end=%s",
                day.date(), login_date.date(), end.date())

    # --- write every chart as its own HTML file ---
    try:
        paths = save_all(day, login_date, end)
    except Exception:
        logger.exception("run_reports.py: save_all() failed for day=%s", day.date())
        return 1

    logger.info("run_reports.py: wrote %d chart file(s)", len(paths))
    for name, p in paths.items():
        # CLI output: keep as print so the report paths are readable to the user
        print(f"{name:28s} -> {p}")
        logger.debug("run_reports.py: chart %s -> %s", name, p)

    # --- numbers behind the comparison ---
    try:
        stats = compare_stats(day, login_date, end)
    except Exception:
        logger.exception("run_reports.py: compare_stats() failed for day=%s",
                         day.date())
        return 1

    if stats is None or (hasattr(stats, "empty") and stats.empty):
        logger.warning("run_reports.py: compare_stats() returned no rows — "
                       "nothing to display")
    else:
        logger.info("run_reports.py: compare table has %d row(s)",
                    len(stats))
        print("\n", stats.to_string())

    logger.info("run_reports.py: done")
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    sys.exit(main())