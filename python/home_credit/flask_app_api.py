from flask import Flask, jsonify
from home_credit.load import load_prep_dataset
from home_credit.persist import load_model
from home_credit.best_model_search import train_preproc
from sklearn.preprocessing import MinMaxScaler

# Création de l'application Flask
app = Flask(__name__)

# Chargement du jeu de données
data = load_prep_dataset("baseline_v1")
print(f"baseline_v1 loaded {data.shape}")

# Charger le classifieur
model = load_model("lgbm_baseline_default_third_party")
print(f"classifier loaded {model}")

@app.route("/customer/<int:customer_id>")  #, methods=["GET"])
def customer(customer_id):
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


# Assure que `run` n'est exécuté que si le script est lancé directement
# Et non via l'indirection d'un import
if __name__ == "__main__":
    app.run()

