from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query
import tempfile
import pandas as pd
import shutil

import json
from typing import List
from pathlib import Path
from pipeline.vision_pipeline import detect_multi, ocr_menu, ocr_packaged
from pipeline.portion import vision_to_rows, rows_to_items
from pipeline.nutrition_calculator import calculate_nutrition
from pipeline.meal_advisor import assess_meal, breakfast_advice, to_markdown
from pipeline.user_charts import (
    journey_days, journey_start, model_note, whatif_text,
    plot_today, plot_whatif, plot_journey, plot_today_vs_past, compare_table
)
from pipeline.meal_coach import coach_reply
from pipeline.safety import with_disclaimer
from storage import user_db
from models.models import DetectItem, CalculateRequest, ChatRequest, SaveMealRequest

router = APIRouter()

def get_user_from_db(user_id: int) -> dict:
    """Fetch user by ID from the database."""
    with user_db._session() as c:
        row = c.execute("SELECT id, username, diabetes_type, baseline_mgdl, created_at FROM users WHERE id=?", (user_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return {
        "id": row[0], "username": row[1], "diabetes_type": row[2],
        "baseline_mgdl": row[3], "journey_start": row[4]
    }


def save_upload_files(uploads: List[UploadFile]) -> List[str]:
    temp_dir = tempfile.mkdtemp()
    paths = []
    for file in uploads:
        file_path = Path(temp_dir) / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        paths.append(str(file_path))
    return paths



@router.post("/detect")
async def api_detect(item: DetectItem):
    image_path = Path(item.image_path)
    mode = item.mode

    if not image_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Image file not found."
        )

    paths = [str(image_path)]

    if mode == "Restaurant Food":
        out = ocr_menu(paths[0])
        if out.get("error"):
            raise HTTPException(status_code=400, detail=out["error"])

        rows = [
            [d.get("name", ""), 1, d.get("notes", "")]
            for d in out.get("dishes", [])
        ]

        return {
            "rows": rows,
            "markdown": f"Menu parsed (confidence {out.get('confidence', 0):.0%})",
            "quality": None
        }

    if mode == "Packaged food":
        out = ocr_packaged(paths[0])
        if out.get("error"):
            raise HTTPException(status_code=400, detail=out["error"])

        n = out.get("per_serving", {})
        rows = [[out.get("product", "product"), out.get("serving_size_g", 100), "from label"]]

        return {
            "rows": rows,
            "markdown": f"Label parsed\nper serving: {n.get('carbs_g', '?')} g carbs",
            "quality": None
        }

    out = detect_multi(paths)
    rows = vision_to_rows(out)
    quality = out.get("quality", {})

    if not rows:
        return {
            "rows": [["", 100, "AI could not detect food."]],
            "markdown": "No food detected.",
            "quality": quality
        }

    warn = "\n".join(f"- {w}" for w in out.get("warnings", []))
    need_more = bool(quality.get("should_ask_another_photo"))

    md = "### Review foods\n" + (
        f"**Warnings:**\n{warn}\n\n" if warn else ""
    )

    if need_more:
        reasons = "\n".join(
            f"- {r}" for r in quality.get("reasons", [])
        )
        md += (
            f"\n⚠️ **I might have missed items:**\n{reasons}\n\n"
            f"**{quality.get('suggestion', '')}**"
        )

    return {"rows": rows, "markdown": md, "quality": quality}

@router.post("/calculate")
def api_calculate(req: CalculateRequest):
    table = [[item.name, item.grams, item.note] for item in req.foods]
    items = rows_to_items(table)
    if not items: raise HTTPException(status_code=400, detail="Add at least one food.")

    nutrition = calculate_nutrition(items)
    totals = nutrition.get("total_nutrition", {})
    if not any(totals.values()): raise HTTPException(status_code=400, detail="Nutrition could not be calculated.")

    md = "### 📊 Total Nutrition\n" + "\n".join([f"- **{k}**: {v}" for k, v in totals.items()])
    if req.meal_type == "breakfast":
        fake_vision = {"food_items": [{"food_name": i["name"]} for i in items]}
        md += "\n---\n" + to_markdown(breakfast_advice({"vision": fake_vision, "nutrition": nutrition}))
    else:
        a = assess_meal(totals, "lunch")
        md += f"\n### Verdict: {a['label']}\n" + "\n".join([f"- **{rule['title']}** — {rule['detail']}" for rule in a.get("rules", [])])

    # return {"advice_md": md, "totals": totals, "foods": items, "pending_context": {"totals": totals, "foods": items}}
    return {"advice_md": md, "totals": totals, "foods": items}

@router.post("/save-meal")
def api_save_meal(req: SaveMealRequest,
                  user_id: int = Query(...)):
    user = get_user_from_db(user_id)
    try:
        ts = pd.Timestamp(req.when) if req.when else pd.Timestamp.now()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format.")

    user_db.add_meal(user["id"], ts, req.meal_type, req.totals, req.foods)
    day = ts.strftime("%Y-%m-%d")
    note = f"✅ Saved {req.meal_type} at {ts:%Y-%m-%d %H:%M}.\n{whatif_text(user, day)}\n*{model_note()}*"

    fig_today = plot_today(user, day)
    fig_cmp = plot_today_vs_past(user, day)
    return {
        "message": with_disclaimer(note),
        "charts": {
            "today": json.loads(fig_today.to_json()) if fig_today else None,
            "compare": json.loads(fig_cmp.to_json()) if fig_cmp else None
        }
    }

@router.get("/charts")
def api_get_charts(day: str,
                   user_id: int = Query(...)):
    user = get_user_from_db(user_id)
    try:
        return {
            "whatif_text": whatif_text(user, day),
            "charts": {
                "today": json.loads(plot_today(user, day).to_json()),
                "whatif": json.loads(plot_whatif(user, day).to_json()),
                "compare": json.loads(plot_today_vs_past(user, day).to_json()),
                "journey": json.loads(plot_journey(user).to_json()),
            },
            "stats": compare_table(user, day).to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat")
def api_chat(req: ChatRequest,
             user_id: int = Query(...)):
    user = get_user_from_db(user_id)
    try:
        reply = coach_reply(user, req.history, req.message, extra_context=req.context)
    except Exception:
        reply = "Sorry, I couldn't process that."
    return {"reply": reply} 