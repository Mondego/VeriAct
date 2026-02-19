import os
from pathlib import Path
from dotenv import load_dotenv

_BASE_DIR = Path(__file__).parent.parent.parent # base directory of the project specsysns
load_dotenv(dotenv_path=str(_BASE_DIR / "config" / ".env"))
