from fastapi import FastAPI
import logging
logging.basicConfig(level=logging.INFO)

# update the PYTHONPATH and PROJECT_DIR from .env file
import os, sys
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.
if python_path := os.getenv("PYTHONPATH"):
    python_path_list = python_path.split(";")
    for path in python_path_list:
        sys.path.insert(0, path)

from get_table import router as get_table_router
from get_table_names import router as get_table_names_router

app = FastAPI()

logging.info("Fast API started")

@app.get("/")
def hello_world():
    # Ajoutez un message de journalisation pour indiquer que la fonction est appelée
    logging.info("hello_world() called.")
    return {"message": "Hello, World!"}


# Include routers or individual routes here if necessary
app.include_router(get_table_router)
app.include_router(get_table_names_router)

# Other FastAPI configurations and main routes here
