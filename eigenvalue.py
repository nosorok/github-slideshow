"""
Eigenvalue computation for square NxN matrices using the QR algorithm.

Mathematical approach:
    The QR algorithm iteratively decomposes a matrix A into Q (orthogonal) and
    R (upper-triangular) factors, then forms A_next = R @ Q. As iterations
    proceed, A converges toward an upper-triangular (Schur) form whose diagonal
    entries are the eigenvalues.

    Before iterating, the matrix is reduced to upper-Hessenberg form using
    Householder reflections, which dramatically speeds convergence.  Wilkinson
    shifts are applied at each step to improve convergence for difficult spectra.

Non-convergence risk:
    For high-dimensional or ill-conditioned matrices the QR iteration may not
    converge within the allowed number of iterations.  In such cases, a
    ``ConvergenceError`` is raised so callers can detect and handle the
    situation instead of silently returning incorrect results.
"""

import math
from copy import deepcopy


class ConvergenceError(Exception):
    """Raised when the QR algorithm fails to converge."""


# ---------------------------------------------------------------------------
# Low-level helpers (pure-Python, no external dependencies)
# ---------------------------------------------------------------------------

def _mat_zeros(n, m):
    """Return an n x m zero matrix."""
    return [[0.0] * m for _ in range(n)]


def _mat_eye(n):
    """Return an n x n identity matrix."""
    m = _mat_zeros(n, n)
    for i in range(n):
        m[i][i] = 1.0
    return m


def _mat_copy(A):
    return [row[:] for row in A]


def _mat_mul(A, B):
    """Multiply two matrices A (n x m) and B (m x p) -> n x p."""
    n = len(A)
    m = len(B)
    p = len(B[0])
    C = _mat_zeros(n, p)
    for i in range(n):
        for k in range(m):
            a_ik = A[i][k]
            for j in range(p):
                C[i][j] += a_ik * B[k][j]
    return C


def _mat_transpose(A):
    n = len(A)
    m = len(A[0])
    T = _mat_zeros(m, n)
    for i in range(n):
        for j in range(m):
            T[j][i] = A[i][j]
    return T


def _vec_norm(v):
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------
# Householder QR factorisation
# ---------------------------------------------------------------------------

def _householder_qr(A):
    """Return (Q, R) via Householder reflections for an n x n matrix A."""
    n = len(A)
    R = _mat_copy(A)
    Q = _mat_eye(n)

    for k in range(n - 1):
        # Build the Householder vector for column k
        x = [R[i][k] for i in range(k, n)]
        alpha = _vec_norm(x)
        if x[0] >= 0:
            alpha = -alpha
        x[0] -= alpha
        norm_x = _vec_norm(x)
        if norm_x < 1e-15:
            continue
        v = [xi / norm_x for xi in x]

        # Apply H = I - 2*v*v^T to R (rows k..n, cols k..n)
        for j in range(k, n):
            dot = sum(v[i - k] * R[i][j] for i in range(k, n))
            for i in range(k, n):
                R[i][j] -= 2.0 * v[i - k] * dot

        # Accumulate into Q
        for j in range(n):
            dot = sum(v[i - k] * Q[i][j] for i in range(k, n))
            for i in range(k, n):
                Q[i][j] -= 2.0 * v[i - k] * dot

    Q = _mat_transpose(Q)  # Q was built as Q^T
    return Q, R


# ---------------------------------------------------------------------------
# Hessenberg reduction
# ---------------------------------------------------------------------------

def _hessenberg(A):
    """Reduce A to upper-Hessenberg form via Householder reflections.

    Returns H (upper-Hessenberg) such that H = P^T A P for some orthogonal P.
    """
    n = len(A)
    H = _mat_copy(A)

    for k in range(n - 2):
        x = [H[i][k] for i in range(k + 1, n)]
        alpha = _vec_norm(x)
        if alpha < 1e-15:
            continue
        if x[0] >= 0:
            alpha = -alpha
        x[0] -= alpha
        norm_x = _vec_norm(x)
        if norm_x < 1e-15:
            continue
        v = [xi / norm_x for xi in x]

        # H <- H - 2 v (v^T H)  (left multiply by reflector)
        for j in range(n):
            dot = sum(v[i] * H[i + k + 1][j] for i in range(len(v)))
            for i in range(len(v)):
                H[i + k + 1][j] -= 2.0 * v[i] * dot

        # H <- H - 2 (H v) v^T  (right multiply by reflector)
        for i in range(n):
            dot = sum(v[j] * H[i][j + k + 1] for j in range(len(v)))
            for j in range(len(v)):
                H[i][j + k + 1] -= 2.0 * dot * v[j]

    return H


# ---------------------------------------------------------------------------
# QR algorithm with Wilkinson shift
# ---------------------------------------------------------------------------

def _wilkinson_shift(H, n):
    """Compute the Wilkinson shift from the trailing 2x2 block of H."""
    a = H[n - 2][n - 2]
    b = H[n - 2][n - 1]
    c = H[n - 1][n - 2]
    d = H[n - 1][n - 1]
    tr = a + d
    det = a * d - b * c
    disc = tr * tr - 4.0 * det
    if disc < 0:
        # Complex eigenvalues — use d as the shift (closest to the corner)
        return d
    sqrt_disc = math.sqrt(disc)
    mu1 = (tr + sqrt_disc) / 2.0
    mu2 = (tr - sqrt_disc) / 2.0
    # Pick the eigenvalue closer to d
    if abs(mu1 - d) <= abs(mu2 - d):
        return mu1
    return mu2


def compute_eigenvalues(matrix, max_iterations=10000, tol=1e-10):
    """Compute eigenvalues of a square NxN matrix using the QR algorithm.

    Parameters
    ----------
    matrix : list[list[float]]
        A square NxN matrix represented as a list of lists.
    max_iterations : int
        Maximum number of QR iterations before raising ``ConvergenceError``.
    tol : float
        Convergence tolerance.  An off-diagonal element is considered zero
        when its absolute value is below ``tol``.

    Returns
    -------
    list[float]
        Eigenvalues in no particular order.

    Raises
    ------
    ValueError
        If the input is not a square matrix.
    ConvergenceError
        If the algorithm does not converge within *max_iterations*.
    """
    # --- Input validation ---------------------------------------------------
    n = len(matrix)
    if n == 0:
        raise ValueError("Matrix must be non-empty")
    for i, row in enumerate(matrix):
        if len(row) != n:
            raise ValueError(
                f"Matrix is not square: row {i} has {len(row)} columns, "
                f"expected {n}"
            )

    # --- Trivial sizes ------------------------------------------------------
    if n == 1:
        return [float(matrix[0][0])]

    if n == 2:
        a, b = matrix[0]
        c, d = matrix[1]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4.0 * det
        if disc < 0:
            raise ConvergenceError(
                "2x2 matrix has complex eigenvalues; only real eigenvalues "
                "are supported"
            )
        sqrt_disc = math.sqrt(disc)
        return [(tr + sqrt_disc) / 2.0, (tr - sqrt_disc) / 2.0]

    # --- Reduce to Hessenberg form -----------------------------------------
    H = _hessenberg([[float(x) for x in row] for row in matrix])

    eigenvalues = []
    m = n  # active sub-matrix size

    total_iters = 0

    while m > 2:
        iters = 0
        while iters < max_iterations:
            total_iters += 1
            # Check for convergence of the bottom-right corner
            if abs(H[m - 1][m - 2]) < tol * (abs(H[m - 1][m - 1]) + abs(H[m - 2][m - 2]) + 1e-30):
                eigenvalues.append(H[m - 1][m - 1])
                m -= 1
                break

            # Wilkinson shift
            mu = _wilkinson_shift(H, m)

            # Shift
            for i in range(m):
                H[i][i] -= mu

            # QR decomposition of the active sub-matrix
            # We operate only on the top-left m x m block
            sub = [H[i][:m] for i in range(m)]
            Q, R = _householder_qr(sub)
            sub = _mat_mul(R, Q)

            # Un-shift
            for i in range(m):
                sub[i][i] += mu

            # Write back
            for i in range(m):
                for j in range(m):
                    H[i][j] = sub[i][j]

            iters += 1
        else:
            raise ConvergenceError(
                f"QR algorithm did not converge after {max_iterations} "
                f"iterations (matrix size {n}x{n}, active block {m}x{m}). "
                "This can happen with high-dimensional or ill-conditioned "
                "matrices."
            )

    # Handle remaining 2x2 or 1x1 block
    if m == 2:
        a, b = H[0][0], H[0][1]
        c, d = H[1][0], H[1][1]
        tr = a + d
        det = a * d - b * c
        disc = tr * tr - 4.0 * det
        if disc < 0:
            raise ConvergenceError(
                "Remaining 2x2 block has complex eigenvalues; only real "
                "eigenvalues are supported"
            )
        sqrt_disc = math.sqrt(disc)
        eigenvalues.append((tr + sqrt_disc) / 2.0)
        eigenvalues.append((tr - sqrt_disc) / 2.0)
    elif m == 1:
        eigenvalues.append(H[0][0])

    return eigenvalues


# ---------------------------------------------------------------------------
# Convenience / demo
# ---------------------------------------------------------------------------

def print_matrix(matrix, label="Matrix"):
    """Pretty-print a matrix."""
    print(f"{label}:")
    for row in matrix:
        print("  [" + ", ".join(f"{x:10.4f}" for x in row) + "]")


if __name__ == "__main__":
    # --- Identity matrix ---
    I3 = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
    print_matrix(I3, "Identity 3x3")
    eigs = compute_eigenvalues(I3)
    print(f"  Eigenvalues: {[round(e, 6) for e in eigs]}\n")

    # --- Singular matrix (det = 0, so 0 is an eigenvalue) ---
    S = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    print_matrix(S, "Singular 3x3")
    eigs = compute_eigenvalues(S)
    print(f"  Eigenvalues: {[round(e, 6) for e in eigs]}\n")

    # --- Symmetric matrix (guaranteed real eigenvalues) ---
    A = [[4, 1, 2],
         [1, 3, 1],
         [2, 1, 5]]
    print_matrix(A, "Symmetric 3x3")
    eigs = compute_eigenvalues(A)
    print(f"  Eigenvalues: {[round(e, 6) for e in eigs]}\n")

    # --- Larger 5x5 matrix ---
    B = [
        [ 2,  0,  1,  0,  0],
        [ 0,  3,  0,  1,  0],
        [ 1,  0,  4,  0,  1],
        [ 0,  1,  0,  5,  0],
        [ 0,  0,  1,  0,  6],
    ]
    print_matrix(B, "Symmetric 5x5")
    eigs = compute_eigenvalues(B)
    print(f"  Eigenvalues: {[round(e, 6) for e in eigs]}\n")

    # --- Diagonal matrix ---
    D = [[7, 0, 0, 0],
         [0, 3, 0, 0],
         [0, 0, -2, 0],
         [0, 0, 0, 11]]
    print_matrix(D, "Diagonal 4x4")
    eigs = compute_eigenvalues(D)
    print(f"  Eigenvalues: {[round(e, 6) for e in eigs]}\n")
