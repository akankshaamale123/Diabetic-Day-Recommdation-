# seed_users.py
from storage import user_db

def create_sample_users():
    user_db.init_user_tables()
    
    users = [
        ("alice",   "type1",       120.0, "2026-10-01T08:00:00"),
        ("bob",     "type2",       140.0, "2026-10-15T08:00:00"),
        ("charlie", "prediabetes", 110.0, "2026-10-20T08:00:00"),
    ]
    
    for name, dtype, base, created in users:
        u = user_db.get_or_create_user(name, dtype, base, created_at=created)
        print(f"✅ User ready: ID={u['id']} | {name} ({dtype})")

if __name__ == "__main__":
    create_sample_users()