# Isolé ici (du reste de prod.py) car python 3.10
from typing import Union
import time
import mlflow
from sklearn.base import ClassifierMixin
from sklearn import preprocessing
from home_credit.load import load_prep_dataset, get_mlflow_dir
from home_credit.best_model_search import kfold_train_and_eval_model_v2, post_train_eval
from imblearn.base import BaseSampler
from imblearn import under_sampling


def do_first_experiment(
    trainset_name: str,
    clf: ClassifierMixin,
    imb_sampler: Union[BaseSampler, None] = under_sampling.RandomUnderSampler(random_state=42)
) -> None:
    """
    Train and evaluate a given classifier on a specified dataset using
    k-fold cross validation, and log the results and relevant metrics to mlflow.
    Saves the trained classifier as a reusable model artifact.

    The trained model can be reloaded and used later, without the need for
    retraining, by specifying the appropriate artifact URI obtained from the
    MLflow UI or API.

    Parameters
    ----------
    trainset_name : str
        Name of the training dataset.
    clf : ClassifierMixin
        Classifier to train and evaluate.
    """
    print("default `mlruns/`:", mlflow.get_tracking_uri())
    mlflow.set_tracking_uri(f"file://{get_mlflow_dir()}")
    print("this project `mlruns/`:", mlflow.get_tracking_uri())

    #experiment = "train_baseline_10k"
    data = load_prep_dataset(trainset_name)
    scaler = preprocessing.MinMaxScaler()
    """clf = ensemble.RandomForestClassifier(
        n_estimators=100, random_state=42,
        verbose=1, n_jobs=-1
    )"""
    clf_name = clf.__class__.__name__
    mlflow.set_experiment(f"{clf_name}_{trainset_name}")
    with mlflow.start_run() as run:
        # Log clf hyperparameters
        mlflow.log_params(clf.get_params())

        # Train and evaluate model with k-fold cross validation
        train_time = -time.time()
        res = kfold_train_and_eval_model_v2(data, clf, verbosity=3)
        train_time += time.time()
        print(f"Train time (min.): {train_time/60:.1f}")

        eval_time = -time.time()

        # Log k-fold cross validation metrics
        for sc_name, sc_val in res["scores"].items():
            mlflow.log_metric(f"{sc_name}_overall", sc_val["overall"])
            snof = f"{sc_name}_over_folds"
            svof = sc_val["over_folds"]
            [mlflow.log_metric(f"{snof}_{i}", v) for i, v in enumerate(svof)]

        # Evaluate model with different thresholds and log relevant metrics
        for t in range(45, 34, -5):  # 45 %, 40 %, 35 %
            # ...
            eval_res = post_train_eval(data, res, clf, scaler, t/100)
            (
                comb_im_fp, roc_im_fp, conf_im_fp,
                auc_rs, auc_s, auc, auc_v, auc_rv, cm
            ) = eval_res

            # Log post eval AUCs
            mlflow.log_metric(f"AUC_RS_{t}PCT", auc_rs)
            mlflow.log_metric(f"AUC_S_{t}PCT", auc_s)
            mlflow.log_metric(f"AUC_{t}PCT", auc)
            mlflow.log_metric(f"AUC_V_{t}PCT", auc_v)
            mlflow.log_metric(f"AUC_RV_{t}PCT", auc_rv)

            # Log final confusion relevant metrics
            tn, fn, tp, fp = cm[0, 0], cm[1, 0], cm[1, 1], cm[0, 1]
            fnr, fpr = fn / tn, fp / tp
            mlflow.log_metric(f"TN_RV_{t}PCT", tn)
            mlflow.log_metric(f"FN_RV_{t}PCT", fn)
            mlflow.log_metric(f"TP_RV_{t}PCT", tp)
            mlflow.log_metric(f"FP_RV_{t}PCT", fp)
            mlflow.log_metric(f"FNR_RV_{t}PCT", fnr)
            mlflow.log_metric(f"FPR_RV_{t}PCT", fpr)
            print(f"{t} % exp res: AUC ({auc_rv}), FNR ({fnr}), FPR ({fpr})")


            # Log plots (overview, final AUC and confusion mx)
            mlflow.log_artifact(comb_im_fp)
            mlflow.log_artifact(roc_im_fp)
            mlflow.log_artifact(conf_im_fp)

        eval_time += time.time()

        # Log times
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metric("eval_time", eval_time)

        # Save the model
        mlflow.sklearn.log_model(clf, f"model_{clf_name}")


    # Return the run_id
    return run.info.run_id
