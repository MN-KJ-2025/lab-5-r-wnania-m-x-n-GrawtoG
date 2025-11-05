# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np

print(np.linspace(1, 10))


def spare_matrix_Abt(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja tworząca zestaw składający się z macierzy A (m,n) i
    wektora b (m,) na podstawie pomocniczego wektora t (m,).

    Args:
        m (int): Liczba wierszy macierzy A.
        n (int): Liczba kolumn macierzy A.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A o rozmiarze (m,n),
            - Wektor b (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(m, int) or not isinstance(n, int):
            return None
        if m <= 0 or n <= 0:
            return None
        t = np.linspace(0, 1, m)
        return (
            np.asarray(np.vander(t, n, True)),
            np.asarray([np.cos(4 * _t) for _t in t]),
        )
    except Exception:
        return None


def square_from_rectan(
    A: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja przekształcająca układ równań z prostokątną macierzą współczynników
    na kwadratowy układ równań.
    A^T * A * x = A^T * b  ->  A_new * x = b_new

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (tuple[np.ndarray, np.ndarray]):
            - Macierz A_new o rozmiarze (n,n),
            - Wektor b_new (n,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        A = np.asarray(A)
        b = np.asarray(b)

        if A.ndim != 2 or b.ndim != 1:
            return None

        m = A.shape[0]
        if b.shape[0] != m:
            return None

        return A.T @ A, A.T @ b
    except Exception:
        return None


def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci:
    Ax = b

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej stronie równania.

    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not (
            isinstance(A, np.ndarray)
            and isinstance(x, np.ndarray)
            and isinstance(b, np.ndarray)
        ):
            return None

        if b.ndim == 2 and b.shape[1] == 1:
            b = b.ravel()
        if x.ndim == 2 and x.shape[1] == 1:
            x = x.ravel()

        if (A.shape)[1] != (x.shape)[0]:
            return None

        if A.ndim != 2 or b.ndim != 1 or x.ndim != 1:
            return None

        return np.linalg.norm(b - (A @ x))
    except Exception:
        return None
