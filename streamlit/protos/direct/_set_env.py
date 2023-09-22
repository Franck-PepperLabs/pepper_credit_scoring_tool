# update the PYTHONPATH and PROJECT_DIR from .env file
import os, sys
from dotenv import load_dotenv


def setup_python_path():
    # Update the PYTHONPATH and PROJECT_DIR from .env file
    load_dotenv()
    if python_path := os.getenv("PYTHONPATH"):
        python_path_list = python_path.split(";")
        for path in python_path_list:
            sys.path.insert(0, path)
