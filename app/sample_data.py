from datetime import datetime, timedelta
from storage.db import init_db, _conn
from storage import user_db
import json
# ---- templates: (food_name, portion_g, kcal, carbs, sugar, protein,
#                  fat, fiber, cholesterol, calcium, source, confidence) ----

BREAKFAST_TPL = [
    ("Idli with sambar",         250, 200, 35, 3, 6,  2,  3, 0,  40,  "database", 0.95),
    ("Poha with peanuts",        180, 260, 42, 2, 5,  9,  2, 0,  20,  "database", 0.92),
    ("Vegetable upma",           180, 210, 32, 1, 4,  7,  3, 0,  25,  "database", 0.90),
    ("Oats with milk",           220, 190, 30, 6, 7,  4,  4, 8,  120, "database", 0.90),
    ("Besan chilla (2 pcs)",     150, 220, 18, 1, 10, 11, 4, 0,  40,  "database", 0.88),
    ("Plain dosa with chutney",  180, 220, 36, 2, 4,  6,  2, 0,  20,  "database", 0.92),
    ("Aloo paratha with curd",   200, 340, 48, 3, 8,  13, 4, 6,  80,  "database", 0.90),
    ("Boiled eggs with toast",   160, 240, 15, 1, 14, 13, 1, 210, 40, "database", 0.93),
    ("Masala omelette with roti",180, 290, 24, 2, 15, 15, 3, 220, 50, "database", 0.91),
    ("Pesarattu with chutney",   200, 230, 34, 2, 8,  6,  4, 0,  30,  "database", 0.90),
]

LUNCH_TPL = [
    ("Dal, rice, mixed sabzi",   400, 460, 70, 4, 12, 10, 6, 0,  80,  "database", 0.92),
    ("Rajma with brown rice",    380, 420, 62, 3, 14,  8, 8, 0,  90,  "database", 0.90),
    ("Chole with 2 rotis",       320, 400, 58, 3, 13, 10, 7, 0,  70,  "database", 0.90),
    ("Paneer curry with roti",   280, 420, 38, 5, 18, 22, 3, 15, 280, "database", 0.92),
    ("Vegetable biryani",        320, 480, 62, 5, 10, 16, 4, 0,  60,  "database", 0.88),
    ("Roti, dal, aloo sabzi",    350, 380, 55, 4, 10,  9, 6, 0,  60,  "database", 0.92),
    ("Khichdi with kadhi",       300, 320, 48, 2, 10,  7, 4, 10, 120, "database", 0.90),
    ("Fish curry with rice",     350, 440, 58, 2, 24, 14, 4, 40, 120, "database", 0.90),
    ("Chicken curry with roti",  320, 450, 42, 3, 28, 20, 3, 70, 80,  "database", 0.92),
    ("Sambar rice with poriyal", 380, 400, 66, 4, 10,  9, 8, 0,  90,  "database", 0.90),
]

DINNER_TPL = [
    ("Roti, dal, bhindi sabzi",  320, 330, 46, 3, 9,  8,  5, 0,  60,  "database", 0.92),
    ("Vegetable khichdi",        260, 260, 42, 2, 8,  5,  5, 0,  40,  "database", 0.90),
    ("Grilled paneer salad",     220, 260, 12, 4, 20, 15, 3, 20, 260, "database", 0.88),
    ("Dal soup with 2 rotis",    300, 300, 44, 3, 10, 6,  5, 0,  60,  "database", 0.92),
    ("Mixed veg curry with roti",300, 290, 38, 3, 8,  10, 5, 0,  50,  "database", 0.90),
    ("Palak paneer with roti",   280, 380, 32, 4, 16, 21, 4, 15, 250, "database", 0.92),
    ("Egg curry with rice",      320, 380, 52, 2, 18, 15, 3, 210, 80, "database", 0.91),
    ("Methi chicken with roti",  300, 400, 34, 3, 30, 20, 3, 80, 60,  "database", 0.90),
    ("Vegetable soup + roti",    280, 240, 36, 3, 6,  7,  5, 0,  50,  "database", 0.90),
]

SNACK_TPL = [
    ("Buttermilk (chaas)",       200, 60,  5,  3, 3,  2,  0, 6,  80,  "database", 0.93),
    ("Roasted chana",            40,  160, 22, 1, 8,  4,  6, 0,  20,  "database", 0.92),
    ("Apple",                    150, 80,  20, 15, 0.5, 0.3, 3, 0, 8,  "database", 0.95),
    ("Green tea",                200, 2,   0,  0, 0,  0,  0, 0,  5,   "database", 0.98),
    ("Handful almonds",          20,  120, 4,  1, 4,  10, 3, 0,  40,  "database", 0.93),
    ("Sprouts chaat",            100, 110, 16, 2, 7,  2,  4, 0,  30,  "database", 0.90),
    ("Greek yogurt with berries",150, 130, 12, 9, 10, 4,  2, 5,  100, "database", 0.92),
    ("Masala tea with biscuit",  150, 80,  14, 8, 2,  2,  0, 8,  40,  "database", 0.93),
]


def seed_sample_data(days: int = 15, force: bool = False) -> None:
    """
    Insert realistic Indian meals (breakfast/lunch/snack/dinner) and glucose
    readings for the last `days` days — including TODAY.

    Glucose readings are placed within 30 minutes of each meal, so
    plot_scatter()'s merge_asof(tolerance=30min) will always find pairs.
    """
    init_db()
    with _conn() as c:
        if force:
            c.execute("DELETE FROM meals")
            c.execute("DELETE FROM glucose")
            print("seed_sample_data(): cleared existing rows (force=True)")
        else:
            existing = c.execute("SELECT COUNT(*) FROM meals").fetchone()[0]
            if existing > 0:
                print(f"seed_sample_data(): {existing} meal rows already exist — "
                      f"use force=True to reseed")
                return

        now = datetime.now()
        today = now.date()

        meal_rows:    list[tuple] = []
        glucose_rows: list[tuple] = []

        for d in range(days):
            day = today - timedelta(days=d)
            ds  = day.strftime("%Y-%m-%d")

            # rotate templates per day so consecutive days look different
            b  = BREAKFAST_TPL[d % len(BREAKFAST_TPL)]
            l  = LUNCH_TPL    [d % len(LUNCH_TPL)]
            sn = SNACK_TPL    [d % len(SNACK_TPL)]
            dn = DINNER_TPL   [d % len(DINNER_TPL)]

            # -------- meals: 4 per day --------
            meal_rows.append((f"{ds} 08:00:00", *b))    # breakfast
            meal_rows.append((f"{ds} 13:00:00", *l))    # lunch
            meal_rows.append((f"{ds} 17:00:00", *sn))   # evening snack
            meal_rows.append((f"{ds} 20:00:00", *dn))   # dinner

            # -------- glucose: 7 readings, each within 30 min of a meal --------
            pre_b  = 105 + (d * 3) % 20     # 105–124
            post_b = 155 + (d * 7) % 45     # 155–199
            pre_l  = 110 + (d * 5) % 20     # 110–129
            post_l = 165 + (d * 9) % 50     # 165–214
            pre_s  = 120 + (d * 2) % 15     # 120–134
            pre_d  = 115 + (d * 4) % 20     # 115–134
            post_d = 160 + (d * 6) % 45     # 160–204

            glucose_rows += [
                (f"{ds} 07:30:00", float(pre_b)),   # 30 min before 08:00 breakfast
                (f"{ds} 10:00:00", float(post_b)),  # 2 h after breakfast
                (f"{ds} 12:30:00", float(pre_l)),   # 30 min before 13:00 lunch
                (f"{ds} 15:30:00", float(post_l)),  # 2.5 h after lunch
                (f"{ds} 16:30:00", float(pre_s)),   # 30 min before 17:00 snack
                (f"{ds} 19:30:00", float(pre_d)),   # 30 min before 20:00 dinner
                (f"{ds} 21:30:00", float(post_d)),  # 1.5 h after dinner
            ]

        c.executemany(
            "INSERT INTO meals ("
            "timestamp, food_name, portion_grams, calories_kcal, carbs_g, sugar_g, "
            "protein_g, fat_g, fiber_g, cholesterol_mg, calcium_mg, source, confidence"
            ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            meal_rows,
        )
        c.executemany(
            "INSERT INTO glucose (timestamp, glucose_mgdl) VALUES (?,?)",
            glucose_rows,
        )

        print(f"seed_sample_data(): inserted {len(meal_rows)} meal rows "
              f"({days} days × 4 meals) and {len(glucose_rows)} glucose rows "
              f"({days} days × 7 readings), today={today}")

def seed_user_db(user_id: int = 1,
                 days: int = 15,
                 force: bool = False,
                 login_days_ago: int = 20) -> None:
    """
    Seed storage.user_db tables (users, user_meals, user_glucose).
    """
    now   = datetime.now()
    today = now.date()
    created_ts = (today - timedelta(days=login_days_ago)).strftime("%Y-%m-%d %H:%M:%S")

    with user_db._session() as c:
        # ---- user row ----
        if force:
            c.execute("DELETE FROM user_meals   WHERE user_id = ?", (user_id,))
            c.execute("DELETE FROM user_glucose WHERE user_id = ?", (user_id,))
            print(f"seed_user_db(): cleared user_id={user_id}")

        exists = c.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone()
        if exists is None:
            c.execute(
                "INSERT INTO users (id, username, created_at, diabetes_type, baseline_mgdl) "
                "VALUES (?,?,?,?,?)",
                (user_id, f"user{user_id}", created_ts, "type2", 130.0),
            )
            print(f"seed_user_db(): created user_id={user_id}")
        else:
            c.execute("UPDATE users SET created_at = ? WHERE id = ?",
                      (created_ts, user_id))
            print(f"seed_user_db(): users.created_at → {created_ts}")

        # ---- build rows ----
        meal_rows: list[dict] = []
        gluc_rows: list[dict] = []

        for d in range(days):
            day      = today - timedelta(days=d)
            ds       = day.strftime("%Y-%m-%d")
            is_today = (day == today)

            def add_meal(hour: int, meal_type: str, tpl: tuple) -> None:
                if is_today and now.hour < hour:
                    return
                
                # Unpack everything including cholesterol (_chol) and calcium (_calc)
                (name, portion, kcal, carbs, sugar, prot,
                 fat, fib, _chol, _calc, src, conf) = tpl
                 
                meal_rows.append({
                    "user_id":       user_id,
                    "timestamp":     f"{ds} {hour:02d}:00:00",
                    "meal_type":     meal_type,
                    "foods_json":    json.dumps([{
                        "food_name":     name,
                        "portion_grams": portion,
                        "source":        src,
                        "confidence":    conf,
                    }]),
                    "calories_kcal": float(kcal),
                    "carbs_g":       float(carbs),
                    "sugar_g":       float(sugar),
                    "fiber_g":       float(fib),
                    "protein_g":     float(prot),
                    "fat_g":         float(fat),
                    "cholesterol_mg": float(_chol),  # <--- FIXED HERE
                    "calcium_mg":     float(_calc),  # <--- FIXED HERE
                })

            add_meal( 8, "breakfast", BREAKFAST_TPL[d % len(BREAKFAST_TPL)])
            add_meal(13, "lunch",     LUNCH_TPL    [d % len(LUNCH_TPL)])
            add_meal(17, "snack",     SNACK_TPL    [d % len(SNACK_TPL)])
            add_meal(20, "dinner",    DINNER_TPL   [d % len(DINNER_TPL)])

            # glucose readings
            pre_b  = 105 + (d * 3) % 20
            post_b = 155 + (d * 7) % 45
            pre_l  = 110 + (d * 5) % 20
            post_l = 165 + (d * 9) % 50
            pre_s  = 120 + (d * 2) % 15
            pre_d  = 115 + (d * 4) % 20
            post_d = 160 + (d * 6) % 45

            for hh, mm, val in [
                ( 7, 30, pre_b),
                (10,  0, post_b),
                (12, 30, pre_l),
                (15, 30, post_l),
                (16, 30, pre_s),
                (19, 30, pre_d),
                (21, 30, post_d),
            ]:
                if is_today and (now.hour < hh or
                                 (now.hour == hh and now.minute < mm)):
                    continue
                gluc_rows.append({
                    "user_id":      user_id,
                    "timestamp":    f"{ds} {hh:02d}:{mm:02d}:00",
                    "glucose_mgdl": float(val),
                })

        # ---- insert ----
        if meal_rows:
            cols = list(meal_rows[0].keys())
            sql  = (f"INSERT INTO user_meals ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(cols))})")
            c.executemany(sql, [tuple(r[k] for k in cols) for r in meal_rows])

        if gluc_rows:
            gcols = list(gluc_rows[0].keys())
            gsql  = (f"INSERT INTO user_glucose ({','.join(gcols)}) "
                     f"VALUES ({','.join('?' * len(gcols))})")
            c.executemany(gsql, [tuple(r[k] for k in gcols) for r in gluc_rows])

        print(f"seed_user_db(): user_id={user_id} → "
              f"{len(meal_rows)} meals, {len(gluc_rows)} glucose rows")

if __name__ == "__main__":
    seed_sample_data(days=20, force=True)                # storage.db   (scatter)
    seed_user_db(user_id=1, days=15, force=True,         # storage.user_db (other charts)
                 login_days_ago=20)  
