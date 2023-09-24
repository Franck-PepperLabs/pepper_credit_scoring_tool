import unittest

import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss

from home_credit.model.facade import (
    predict_facade,
    predict_proba_facade,
    predict_facade,
    fit_facade,
    get_feature_importances_facade
)


def custom_lgbm_log_loss(self, y_pred: pd.Series, sample_weight, group):
    # Utiliser log_loss à l'intérieur de cette fonction
    print(f"y_true shape: {self.shape}")
    print(f"y_pred shape: {y_pred.shape}")
    return log_loss(self, y_pred, sample_weight=sample_weight)


class TestFitFacade(unittest.TestCase):

    def test_fit_lgbm_classifier(self):
        # Create a dummy dataset for the example
        X, y = load_iris(return_X_y=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize a LightGBM model
        clf = LGBMClassifier()

        # Test fitting with the LightGBM model
        fit_facade(clf, (X_train, y_train), (X_valid, y_valid), custom_lgbm_log_loss)

        # Check that the model has been fitted correctly
        self.assertTrue(hasattr(clf, "best_iteration_"))

    def test_fit_sklearn_classifier(self):
        # Create a dummy dataset for the example
        X, y = load_iris(return_X_y=True)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize a scikit-learn model
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier()

        # Test fitting with the scikit-learn model
        fit_facade(clf, (X_train, y_train), (X_valid, y_valid), log_loss)

        # Check that the model has been fitted correctly
        self.assertTrue(hasattr(clf, "n_estimators"))


class TestPredictFacade(unittest.TestCase):
    # Class attribute to store the LGBM classifier trained with the Iris dataset
    clf = None

    @classmethod
    def setUpClass(cls):
        # Load the Iris dataset once
        X, y = load_iris(return_X_y=True)

        # Initialize a LightGBM model and fit it
        cls.clf = LGBMClassifier()
        cls.clf.fit(X, y)

    def test_predict_facade(self, X_test):
        predictions = predict_facade(self.clf, X_test)
        # Check if predictions are of the correct shape and data type
        self.assertEqual(predictions_np.shape, (2,))
        self.assertTrue(all(isinstance(pred, int) for pred in predictions))


    def test_predict_with_numpy_array(self):
        # Test the predict_facade function with a NumPy array
        X_test_np = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]])
        test_predict_facade(self, X_test_np)

    def test_predict_with_pandas_dataframe(self):
        # Test the predict_facade function with a Pandas DataFrame
        X_test_df = pd.DataFrame({
            'feature1': [5.1, 6.2],
            'feature2': [3.5, 2.9],
            'feature3': [1.4, 4.3],
            'feature4': [0.2, 1.3]
        })
        test_predict_facade(self, X_test_df)


class TestPredictProbaFacade(unittest.TestCase):

    def test_predict_proba_facade_with_lgbm(self):
        # Create a dummy LGBMClassifier
        clf = LGBMClassifier()

        # Create a sample input dataset
        X_test = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 2.9, 4.3, 1.3]])

        # Make predictions
        predictions = predict_proba_facade(clf, X_test)

        # Assertions
        self.assertTrue(isinstance(predictions, np.ndarray))
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.all(predictions >= 0.0) and np.all(predictions <= 1.0))

    def test_predict_proba_facade_with_other_classifier(self):
        # Create a dummy classifier that is not LGBM
        class DummyClassifier:
            def predict_proba(self, X):
                return np.array([[0.2, 0.8], [0.6, 0.4]])

        # Create a sample input dataset
        X_test = np.array([[1.1, 2.2], [3.3, 4.4]])

        # Make predictions
        predictions = predict_proba_facade(DummyClassifier(), X_test)

        # Assertions
        self.assertTrue(isinstance(predictions, np.ndarray))
        self.assertEqual(predictions.shape, (2,))
        self.assertTrue(np.all(predictions >= 0.0) and np.all(predictions <= 1.0))


class TestGetFeatureImportancesFacade(unittest.TestCase):
    # Class attribute to store the Iris dataset
    iris = None

    @classmethod
    def setUpClass(cls):
        # Load the Iris dataset once
        cls.iris = load_iris()

    def train_classifier(self, clf):
        # Check if Iris dataset is loaded
        self.assertIsNotNone(self.iris)

        # Train the classifier with the Iris dataset
        clf.fit(self.iris.data, self.iris.target)

    def test_not_dummy_classifier(self, clf_class):
        # Initialize a clf_class classifier
        clf = clf_class()
        
        # Train the classifier
        self.train_classifier(clf)
        
        # Ensure feature importances are obtained as a list
        importances = get_feature_importances_facade(clf)
        self.assertIsInstance(importances, list)

    def test_lgbm_classifier(self):
        # Test a LightGBM model
        self.test_not_dummy_classifier(LGBMClassifier)

    def test_random_forest_classifier(self):
        # Test a RandomForestClassifier
        self.test_not_dummy_classifier(RandomForestClassifier)

    def test_logistic_regression(self):
        # Test a LogisticRegression model
        self.test_not_dummy_classifier(LogisticRegression)

    def test_dummy_classifier(self):
        # Initialize a DummyClassifier
        clf = DummyClassifier(strategy='uniform')
        self.train_classifier(clf)
        # Ensure feature importances are obtained as a list with equal values
        importances = get_feature_importances_facade(clf)
        expected_importances = [1.0 / len(self.iris.data[0])] * len(self.iris.data[0])
        self.assertEqual(importances, expected_importances)


if __name__ == '__main__':
    unittest.main()
