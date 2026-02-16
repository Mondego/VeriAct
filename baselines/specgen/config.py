import os
from dotenv import load_dotenv

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BASE_DIR, "config", ".env"))

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
