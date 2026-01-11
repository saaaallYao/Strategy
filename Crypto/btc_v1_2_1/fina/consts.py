import hashlib
import os
import pathlib

from dotenv import load_dotenv

# Load environment variables from .env file
# Specify the path to .env file relative to this consts.py file
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(env_path)
# print("OpenAI key:", os.getenv('OPENAI_API_KEY'))

import pandas as pd

TRUE_CUR_PATH = os.path.dirname(__file__)
TRUE_CUR_PATH = TRUE_CUR_PATH if TRUE_CUR_PATH != "" else "."

# TRUE_CUR_PATH = pathlib.Path.cwd()

pl_TRUE_CUR_PATH = pathlib.Path(TRUE_CUR_PATH)
BASE_PATH = pl_TRUE_CUR_PATH.parent

# Check the auto root path is correct
try:
    assert BASE_PATH.name == "fina"
except Exception as e:
    print("Base dir should be something like ....../fina/: ", BASE_PATH)
    print("Important! Ensure TRUE_CUR_PATH is your working directory:", TRUE_CUR_PATH)
    print("Exception:", e)

DATA_PATH = BASE_PATH / "data"
PROD_PATH = DATA_PATH / "prod"
LOG_PATH = DATA_PATH / "logs"
LLM_RESPONSE_CACHE = PROD_PATH / "llm_res"
CACHE_PATH = DATA_PATH / "cache"

for p in [DATA_PATH, LLM_RESPONSE_CACHE, LOG_PATH, CACHE_PATH]:
    p.mkdir(parents=True, exist_ok=True)

assert DATA_PATH.exists()
