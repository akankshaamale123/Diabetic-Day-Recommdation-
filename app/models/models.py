from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class DetectItem(BaseModel):
    image_path: str="F:/Akanksha/diabetic and Cancer day/Diabetic Diet plan/170302114154-masala-dosa.jpg"
    mode: str = "Meal (single/multi)"


class MealItem(BaseModel):
    name: str
    grams: float
    note: str = ""

class CalculateRequest(BaseModel):
    foods: List[MealItem]
    meal_type: str

class SaveMealRequest(BaseModel):
    meal_type: str
    when: Optional[str] = None
    totals: Dict[str, Any]
    foods: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    message: str
    history: List[Dict[str, str]]
    context: Optional[Dict[str, Any]] = None
