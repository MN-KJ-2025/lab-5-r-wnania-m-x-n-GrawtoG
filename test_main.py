import pickle
from typing import Any

import main
import numpy as np
import pytest

try:
    with open("expected", "rb") as f:
        expected = pickle.load(f)
except FileNotFoundError:
    print(
        "Error: The 'expected' file was not found. Please ensure it is in the correct directory."
    )
    expected = {"square_from_rectan": [], "residual_norm": [], "spare_matrix_Abt": []}
print(expected["spare_matrix_Abt"][7
                                   ],sep="\npjaisd\n")

# --- Data Preparation ---

valid_spare_matrix_Abt = [
    (m, n, res) for m, n, res in expected["spare_matrix_Abt"] if res is not None
]
invalid_spare_matrix_Abt = [
    (m, n, res) for m, n, res in expected["spare_matrix_Abt"] if res is None
]

valid_residual_norm = [
    (A, x, b, res) for A, x, b, res in expected["residual_norm"] if res is not None
]
invalid_residual_norm = [
    (A, x, b, res) for A, x, b, res in expected["residual_norm"] if res is None
]

valid_square_from_rectan = [
    (A, b, res) for A, b, res in expected["square_from_rectan"] if res is not None
]
invalid_square_from_rectan = [
    (A, b, res) for A, b, res in expected["square_from_rectan"] if res is None
]


# --- Tests for spare_matrix_Abt ---


@pytest.mark.parametrize("m, n, expected_result", invalid_spare_matrix_Abt)
def test_spare_matrix_Abt_invalid_input(m: Any, n: Any, expected_result: None):
    """Tests if spare_matrix_Abt correctly handles invalid input data by returning None."""
    actual = main.spare_matrix_Abt(m, n)
    assert actual is None, (
        f"For invalid input, expected None but got {actual}."
    )


@pytest.mark.parametrize("m, n, expected_result", valid_spare_matrix_Abt)
def test_spare_matrix_Abt_correct_solution(
    m: int, n: int, expected_result: tuple[np.ndarray, np.ndarray]
):
    """Tests if spare_matrix_Abt produces the correct matrix A and vector b for valid inputs."""
    A, b = main.spare_matrix_Abt(m, n)
    expected_A, expected_b = expected_result

    assert A == pytest.approx(expected_A), (
        f"Matrix A is incorrect for input ({m}, {n})."
    )
    assert b == pytest.approx(expected_b), (
        f"Vector b is incorrect for input ({m}, {n})."
    )


# --- Tests for residual_norm ---


@pytest.mark.parametrize("A, x, b, expected_result", invalid_residual_norm)
def test_residual_norm_invalid_input(
    A: np.ndarray, x: np.ndarray, b: np.ndarray, expected_result: None
):
    """Tests if residual_norm correctly handles invalid input data by returning None."""
    actual = main.residual_norm(A, x, b)
    assert actual is None, f"For invalid input, expected None but got {actual}."


@pytest.mark.parametrize("A, x, b, expected_result", valid_residual_norm)
def test_residual_norm_correct_solution(
    A: np.ndarray, x: np.ndarray, b: np.ndarray, expected_result: float
):
    """Tests if residual_norm calculates the correct residual norm for valid inputs."""
    actual_result = main.residual_norm(A, x, b)
    assert actual_result == pytest.approx(expected_result), (
        f"Expected norm {expected_result}, but got {actual_result}."
    )


# # --- Tests for square_from_rectan ---


# @pytest.mark.parametrize("A, b, expected_result", invalid_square_from_rectan)
# def test_square_from_rectan_invalid_input(
#     A: np.ndarray, b: np.ndarray, expected_result: None
# ):
#     """Tests if square_from_rectan correctly handles invalid input data by returning None."""
#     actual = main.square_from_rectan(A, b)
#     assert actual is None, f"For invalid input, expected None but got {actual}."


# @pytest.mark.parametrize("A, b, expected_result", valid_square_from_rectan)
# def test_square_from_rectan_correct_solution(
#     A: np.ndarray, b: np.ndarray, expected_result: tuple[np.ndarray, np.ndarray]
# ):
#     """Tests if square_from_rectan produces the correct square matrix and vector for valid inputs."""
#     At, bt = main.square_from_rectan(A, b)
#     expected_At, expected_bt = expected_result

#     assert At == pytest.approx(expected_At), "Transformed matrix A_new is incorrect."
#     assert bt == pytest.approx(expected_bt), "Transformed vector b_new is incorrect."
