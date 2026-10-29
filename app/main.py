# app/main.py
import logging
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the Gradio UI we built in app.py
from app import demo as gradio_demo

# Import the FastAPI router we built in routes.py
from models.routes import router as api_router

# Set up FastAPI
app = FastAPI(title="Diabetic Diet API & UI")

# Allow frontend (React/Vue) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Add all API routes under /api
app.include_router(api_router, prefix="/api")

# 2. Mount the Gradio App under /gradio
app = gr.mount_gradio_app(app, gradio_demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    
    print("🚀 Starting App...")
    print("📊 Gradio UI:    http://127.0.0.1:8000/gradio")
    print("⚙️  API Endpoints: http://127.0.0.1:8000/api/...")
    print("📚 API Docs:      http://127.0.0.1:8000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8000)