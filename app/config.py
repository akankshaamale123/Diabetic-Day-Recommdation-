from dotenv import load_dotenv
import os

load_dotenv()

Model = os.getenv("MODEL_NAME", " ")
Model_URL = os.getenv("MODEL_URL")

