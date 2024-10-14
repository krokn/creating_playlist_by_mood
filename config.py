from dotenv import load_dotenv
import os

load_dotenv()

API_KEY=os.environ.get("API_KEY")
FOLDER_ID=os.environ.get("FOLDER_ID")
YANDEXGPT_API_URL=os.environ.get("YANDEXGPT_API_URL")