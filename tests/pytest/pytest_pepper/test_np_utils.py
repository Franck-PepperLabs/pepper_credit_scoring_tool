import pytest
import numpy as np
from pepper.np_utils import subindex


# Définir une liste de cas de test sous forme de tuples (entrée, sortie attendue)
test_cases = [
    (
        np.array([0, 0, 1, 1, 1, 3, 5, 5, 11, 11, 11, 11]),
        np.array([0, 1, 0, 1, 2, 0, 0, 1, 0, 1, 2, 3])),
    (
        np.array([1, 11, 5, 11, 1, 11, 5, 0, 3, 11, 0, 1]),
        np.array([0, 0, 0, 1, 1, 2, 1, 0, 0, 3, 1, 2])
    ),
    # Ajouter d'autres cas de test ici
]

@pytest.mark.parametrize("input_array, expected_result", test_cases)
def test_subindex(input_array, expected_result):
    result = subindex(input_array)
    assert np.array_equal(result, expected_result)

@pytest.mark.parametrize("input_array, expected_result", test_cases)
def test_subindex_sorted(input_array, expected_result):
    result = subindex(input_array, sorted=True)
    assert np.array_equal(result, expected_result)
