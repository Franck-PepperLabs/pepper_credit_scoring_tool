
from typing import Tuple, Optional
from sklearn.base import BaseEstimator
import pandas as pd
from home_credit.impute import default_imputation


def train_preproc(
    data: pd.DataFrame,
    scaler: Optional[BaseEstimator] = None,
    keep_test_samples: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the training data for machine learning.

    This function preprocesses the input data for training machine learning models.
    It performs the following steps:
    1. Separates the training data from the test data if `keep_test_samples` is False.
    2. Imputes missing values using the `default_imputation` function.
    3. Extracts the features (X) and target variable (y).
    4. Optionally scales the features using the specified `scaler`.

    Parameters:
    -----------
    data : pd.DataFrame
        The input data containing both training and test samples.

    scaler : BaseEstimator or None, optional (default=None)
        An optional scaler to standardize the features. If None, no scaling is performed.

    keep_test_samples : bool, optional (default=False)
        If True, keeps test samples in the training data; otherwise, excludes them.

    Returns:
    --------
    X : pd.DataFrame, shape (n_samples, n_features)
        The preprocessed training features.

    y : pd.Series, shape (n_samples,)
        The corresponding target variable.

    Notes:
    ------
    - This function assumes that the target variable is named 'TARGET'
        and the non-feature columns include 'TARGET', 'SK_ID_CURR',
        'SK_ID_BUREAU', 'SK_ID_PREV', and 'index'.
    - Missing values are imputed using the `default_imputation` function.

    Example:
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> data = load_data()  # Load your dataset
    >>> scaler = StandardScaler()
    >>> X_train, y_train = train_preproc(data, scaler=scaler, keep_test_samples=False)
    """
    data_train = data if keep_test_samples else data[data.TARGET > -1]
    
    # Impute missing values
    """
    # Default imputation, if necessary
    # (should have been handled by feature engineering)
    if (data.isna() | np.isinf(data)).any().any():
        data = default_imputation(data)
    """
    data_train = default_imputation(data_train)

    # Exclude non-feature columns from training and test features
    not_feat_names = [
        "TARGET", "SK_ID_CURR", "SK_ID_BUREAU", "SK_ID_PREV", "index"
    ]
    feat_names = data_train.columns.difference(not_feat_names)
    
    # Extract features (X) and target variable (y)
    X = data_train[feat_names]
    y = data_train.TARGET

    # Scale the data
    if scaler is not None:
        scaler.fit(X)
        X = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
    
    return X, y
