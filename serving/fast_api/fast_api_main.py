# from fastapi import FastAPI
# import logging
# logging.basicConfig(level=logging.INFO)

# update the PYTHONPATH and PROJECT_DIR from .env file
# import os, sys
# from dotenv import load_dotenv
# load_dotenv()  # take environment variables from .env.
# if python_path := os.getenv("PYTHONPATH"):
#    python_path_list = python_path.split(";")
#    for path in python_path_list:
#        sys.path.insert(0, path)

from _router_commons import *

from get_table_names import router as table_names_router
from get_table import router as table_router
from get_target import router as target_router
from get_predict import router as predict_router
from get_main_map import router as main_map_router
from get_client_data import router as client_data_router


app = FastAPI()

logging.info("**Fast API** started")

@app.get("/")
def welcome():
    # Ajoutez un message de journalisation pour indiquer que la fonction est appelée
    logging.info("welcome() called.")
    return {"message": "Welcome to Home Credit Dashboard"}


# Include routers or individual routes here if necessary
app.include_router(table_names_router)
app.include_router(table_router)
app.include_router(target_router)
app.include_router(predict_router)
app.include_router(main_map_router)
app.include_router(client_data_router)

# Other FastAPI configurations and main routes here
