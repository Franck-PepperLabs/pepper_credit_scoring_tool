import unittest

import pandas as pd
import numpy as np

from home_credit.impute import default_imputation


class TestDefaultImputation(unittest.TestCase):
    def test_default_imputation(self):
        # Create a test DataFrame with missing values
        data = pd.DataFrame({
            'Feature1': [1.0, 2.0, np.nan, 4.0],
            'Feature2': [5.0, 6.0, 7.0, 8.0],
            'TARGET': [1, 0, 1, -1]  # Example TARGET column
        })

        # Call the default imputation function
        imputed_data = default_imputation(data)

        # Check that the result has the same shape as the input data
        self.assertEqual(imputed_data.shape, data.shape)

        # Check that missing values have been imputed correctly
        expected_imputed_data = pd.DataFrame({
            'Feature1': [1.0, 2.0, 1.5, 4.0],
            'Feature2': [5.0, 6.0, 7.0, 8.0],
            'TARGET': [1.0, 0.0, 1.0, -1.0]
        })
        pd.testing.assert_frame_equal(imputed_data, expected_imputed_data)


if __name__ == '__main__':
    unittest.main()