import unittest
import numpy as np
from pepper.np_utils import subindex

# Définir une liste de cas de test sous forme de tuples (entrée, sortie attendue)
test_cases = [
    (
        np.array([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11]),
        np.array([0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 3])
    ),
    (
        np.array([1, 11, 5, 11, 1, 11, 5, 0, 3, 11, 0, 1]),
        np.array([0, 0, 0, 1, 1, 2, 1, 0, 0, 3, 1, 2])
    ),
    # Ajouter d'autres cas de test ici
]

class TestSubindex(unittest.TestCase):

    def test_subindex_multi(self):
        for input_array, expected_result in test_cases:
            with self.subTest(input_array=input_array, expected_result=expected_result):
                result = subindex(input_array)
                self.assertTrue(np.array_equal(result, expected_result))

    def test_subindex_sorted_multi(self):
        for input_array, expected_result in test_cases:
            with self.subTest(input_array=input_array, expected_result=expected_result):
                result = subindex(input_array, sorted=True)
                self.assertTrue(np.array_equal(result, expected_result))

    def test_subindex_unsorted(self):
        a = np.array([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11])
        result = subindex(a)
        expected_result = np.array([0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 3])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_subindex_sorted(self):
        a = np.array([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11])
        result = subindex(a, sorted=True)
        expected_result = np.array([0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 3])
        self.assertTrue(np.array_equal(result, expected_result))

    def test_subindex_raises_error_for_non_1d_array(self):
        a = np.array([[0, 1], [2, 3]])
        with self.assertRaises(ValueError):
            subindex(a)

if __name__ == '__main__':
    unittest.main()
