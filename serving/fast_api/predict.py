# from fastapi import APIRouter
# import logging
# logging.basicConfig(level=logging.INFO)

# from typing import List, Union
from _router_commons import *
from home_credit.api import predict as _predict


router = APIRouter()
logging.info("<predict> router started")


@router.get("/api/predict")
async def predict(
    sk_curr_id: int,
    proba: bool = False
) -> Union[int, float]:
    """...."""
    log_call_info("predict", locals().copy())
    return _predict(sk_curr_id, proba)

# TODO Old Flask code à réviser / intégrer

"""
# Chargement du jeu de données
from home_credit.load import load_prep_dataset
data = load_prep_dataset("baseline_v1")
print(f"baseline_v1 loaded {data.shape}")

# Charger le classifieur
from home_credit.model_persist import load_model
model = load_model("lgbm_baseline_default_third_party")
print(f"classifier loaded {model}")

from home_credit.best_model_search import train_preproc
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify
def old_predict(customer_id):
    print("customer id:", customer_id)
    customer = data[data.SK_ID_CURR == customer_id]
    print(customer)
    # TODO FATAL ERROR: Erreur de conception critique : je ne peux pas aller plus loin
    # Sans reprendre beaucoup de choses en arrière, donc 1 à 2 jours au moins
    # L'erreur magistrale a été de ne pas faire une pipeline (imputer, scaler, classifier)
    # Et là, je m'en mange les doigts.
    x, y_true = train_preproc(customer, MinMaxScaler(), keep_test_samples=True)
    y_prob = model.predict_proba(x)[:, 1]
    y_pred = int(y_prob > .4)
    print("y_true:", y_true)
    print("y_prob:", y_prob)
    print("y_pred:", y_pred)
    return jsonify({
        "y_true": y_true,
        "y_prob": y_prob,
        "y_pred": y_pred
    })
"""