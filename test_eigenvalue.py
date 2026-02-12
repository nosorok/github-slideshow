"""Tests for eigenvalue computation via QR algorithm."""

import math
import unittest
from eigenvalue import compute_eigenvalues, ConvergenceError


def sorted_eigs(eigs):
    """Return eigenvalues sorted for comparison."""
    return sorted(eigs)


class TestInputValidation(unittest.TestCase):
    """Edge-case and validation tests."""

    def test_empty_matrix(self):
        with self.assertRaises(ValueError):
            compute_eigenvalues([])

    def test_non_square(self):
        with self.assertRaises(ValueError):
            compute_eigenvalues([[1, 2, 3], [4, 5, 6]])

    def test_jagged_rows(self):
        with self.assertRaises(ValueError):
            compute_eigenvalues([[1, 2], [3]])


class TestTrivialMatrices(unittest.TestCase):
    """1x1 and 2x2 matrices."""

    def test_1x1(self):
        self.assertAlmostEqual(compute_eigenvalues([[42]])[0], 42.0)

    def test_2x2_distinct(self):
        # [[5, 1], [0, 3]] -> eigenvalues 5 and 3
        eigs = sorted_eigs(compute_eigenvalues([[5, 1], [0, 3]]))
        self.assertAlmostEqual(eigs[0], 3.0, places=8)
        self.assertAlmostEqual(eigs[1], 5.0, places=8)

    def test_2x2_repeated(self):
        eigs = sorted_eigs(compute_eigenvalues([[4, 0], [0, 4]]))
        self.assertAlmostEqual(eigs[0], 4.0, places=8)
        self.assertAlmostEqual(eigs[1], 4.0, places=8)

    def test_2x2_complex_eigenvalues(self):
        # Rotation matrix has complex eigenvalues -> should raise
        with self.assertRaises(ConvergenceError):
            compute_eigenvalues([[0, -1], [1, 0]])


class TestIdentityMatrix(unittest.TestCase):
    """Identity matrices of various sizes."""

    def test_3x3_identity(self):
        I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        eigs = compute_eigenvalues(I3)
        for e in eigs:
            self.assertAlmostEqual(e, 1.0, places=8)

    def test_5x5_identity(self):
        n = 5
        I5 = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        eigs = compute_eigenvalues(I5)
        self.assertEqual(len(eigs), n)
        for e in eigs:
            self.assertAlmostEqual(e, 1.0, places=8)


class TestSingularMatrix(unittest.TestCase):
    """Singular matrices (determinant = 0, so 0 is an eigenvalue)."""

    def test_3x3_singular(self):
        S = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        eigs = sorted_eigs(compute_eigenvalues(S))
        # One eigenvalue should be ~0
        self.assertAlmostEqual(eigs[0], -1.116844, places=4)
        self.assertAlmostEqual(eigs[1], 0.0, places=6)
        self.assertAlmostEqual(eigs[2], 16.116844, places=4)

    def test_zero_matrix(self):
        Z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        eigs = compute_eigenvalues(Z)
        for e in eigs:
            self.assertAlmostEqual(e, 0.0, places=8)


class TestDiagonalMatrix(unittest.TestCase):
    """Diagonal matrices — eigenvalues are the diagonal entries."""

    def test_4x4_diagonal(self):
        D = [[7, 0, 0, 0],
             [0, 3, 0, 0],
             [0, 0, -2, 0],
             [0, 0, 0, 11]]
        eigs = sorted_eigs(compute_eigenvalues(D))
        expected = sorted([-2.0, 3.0, 7.0, 11.0])
        for a, b in zip(eigs, expected):
            self.assertAlmostEqual(a, b, places=8)


class TestSymmetricMatrix(unittest.TestCase):
    """Symmetric matrices always have real eigenvalues."""

    def test_3x3_symmetric(self):
        A = [[4, 1, 2], [1, 3, 1], [2, 1, 5]]
        eigs = sorted_eigs(compute_eigenvalues(A))
        # Sum of eigenvalues = trace = 12
        self.assertAlmostEqual(sum(eigs), 12.0, places=6)

    def test_eigenvalue_trace_property(self):
        """Sum of eigenvalues == trace of matrix."""
        A = [[2, 0, 1, 0, 0],
             [0, 3, 0, 1, 0],
             [1, 0, 4, 0, 1],
             [0, 1, 0, 5, 0],
             [0, 0, 1, 0, 6]]
        eigs = compute_eigenvalues(A)
        trace = sum(A[i][i] for i in range(len(A)))
        self.assertAlmostEqual(sum(eigs), trace, places=6)

    def test_eigenvalue_determinant_property(self):
        """Product of eigenvalues == determinant of matrix."""
        A = [[4, 1, 2], [1, 3, 1], [2, 1, 5]]
        eigs = compute_eigenvalues(A)
        product = 1.0
        for e in eigs:
            product *= e
        # Determinant by cofactor: 4*(15-1) - 1*(5-2) + 2*(1-6) = 56 - 3 - 10 = 43
        self.assertAlmostEqual(product, 43.0, places=4)


class TestNonConvergence(unittest.TestCase):
    """Test that ConvergenceError is raised with very few iterations."""

    def test_low_iteration_limit(self):
        A = [[4, 1, 2], [1, 3, 1], [2, 1, 5]]
        with self.assertRaises(ConvergenceError):
            compute_eigenvalues(A, max_iterations=1)


class TestLargerMatrices(unittest.TestCase):
    """Test with larger matrices to validate scalability."""

    def test_10x10_diagonal(self):
        n = 10
        D = [[float(i + 1) if i == j else 0.0 for j in range(n)] for i in range(n)]
        eigs = sorted_eigs(compute_eigenvalues(D))
        expected = sorted([float(i + 1) for i in range(n)])
        for a, b in zip(eigs, expected):
            self.assertAlmostEqual(a, b, places=6)

    def test_6x6_symmetric_trace(self):
        A = [
            [10, 1, 0, 0, 0, 0],
            [ 1, 9, 1, 0, 0, 0],
            [ 0, 1, 8, 1, 0, 0],
            [ 0, 0, 1, 7, 1, 0],
            [ 0, 0, 0, 1, 6, 1],
            [ 0, 0, 0, 0, 1, 5],
        ]
        eigs = compute_eigenvalues(A)
        trace = sum(A[i][i] for i in range(6))
        self.assertAlmostEqual(sum(eigs), trace, places=5)


if __name__ == "__main__":
    unittest.main()
