"""
Matrix Engine v3.0 — Determinant, Inversion & Heatmap

Computes the determinant of any NxN square matrix using recursive
Laplace (cofactor) expansion, and the inverse using Gauss-Jordan
elimination with partial pivoting for numerical stability.

Includes a terminal heatmap UI that color-codes matrix cells based
on their value relative to the matrix's value range.

Edge Case Governance
--------------------
Every edge case is tagged with an [EDGE-CASE-nn] comment in the source
so a human auditor can find them with a simple search.
"""

import math
import sys


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# [EDGE-CASE-06] Floating-point tolerance for near-zero comparisons.
# IEEE-754 double has ~15-16 significant digits; 1e-12 gives 3-4 digits
# of margin while still catching truly degenerate pivots.
EPSILON = 1e-12

# [EDGE-CASE-09] Maximum matrix dimension allowed.  Gauss-Jordan is O(n^3)
# so a 500x500 matrix requires ~125 million operations — well within reason.
# The recursive determinant is O(n!) so we cap it separately at 12.
MAX_INVERSION_SIZE = 500
MAX_DETERMINANT_SIZE = 12


# ---------------------------------------------------------------------------
# Input validation (shared)
# ---------------------------------------------------------------------------

def _validate_square_matrix(matrix, caller="operation"):
    """Validate that the input is a non-empty, square, finite-valued matrix.

    Raises ValueError with a descriptive message on any violation.
    """
    # [EDGE-CASE-01] Empty matrix: no rows at all.
    if not matrix or not isinstance(matrix, (list, tuple)):
        raise ValueError(f"{caller}: matrix must be a non-empty list of rows")

    n = len(matrix)

    for i, row in enumerate(matrix):
        # [EDGE-CASE-02] Ragged rows: not all rows have the same length.
        if not isinstance(row, (list, tuple)) or len(row) != n:
            raise ValueError(
                f"{caller}: matrix must be square — row {i} has "
                f"{len(row) if isinstance(row, (list, tuple)) else '?'} "
                f"elements, expected {n}"
            )
        for j, val in enumerate(row):
            # [EDGE-CASE-03] Non-numeric entries.
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"{caller}: non-numeric value at [{i}][{j}]: {val!r}"
                )
            # [EDGE-CASE-04] NaN / Inf entries poison all arithmetic silently.
            if math.isnan(val) or math.isinf(val):
                raise ValueError(
                    f"{caller}: non-finite value at [{i}][{j}]: {val}"
                )

    return n


# ---------------------------------------------------------------------------
# Core determinant logic  (recursive Laplace — kept from v2)
# ---------------------------------------------------------------------------

def get_minor(matrix, row, col):
    """Return the (n-1)x(n-1) minor obtained by deleting the given row and column."""
    return [
        [matrix[r][c] for c in range(len(matrix)) if c != col]
        for r in range(len(matrix)) if r != row
    ]


def determinant(matrix):
    """Compute the determinant of an NxN matrix via recursive Laplace expansion.

    Algorithm
    ---------
    - Base case 1x1: det([a]) = a
    - Base case 2x2: det([[a,b],[c,d]]) = ad - bc
    - Recursive case: expand along the first row
          det(A) = sum over j of (-1)^j * A[0][j] * det(M(0,j))
      where M(0,j) is the (n-1)x(n-1) minor with row 0 and column j removed.
    """
    n = _validate_square_matrix(matrix, "determinant")

    # [EDGE-CASE-10] The recursive determinant is O(n!).  For n>12 this will
    # hang or exhaust memory.  Refuse early with a clear message.
    if n > MAX_DETERMINANT_SIZE:
        raise ValueError(
            f"determinant: matrix is {n}x{n} which exceeds the "
            f"{MAX_DETERMINANT_SIZE}x{MAX_DETERMINANT_SIZE} limit for the "
            f"recursive algorithm (O(n!) time).  Use a different method for "
            f"large matrices."
        )

    # Base cases
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    # Recursive Laplace expansion along the first row
    det = 0
    for col in range(n):
        sign = (-1) ** col
        minor = get_minor(matrix, 0, col)
        det += sign * matrix[0][col] * determinant(minor)
    return det


# ---------------------------------------------------------------------------
# Matrix inversion via Gauss-Jordan elimination with partial pivoting
# ---------------------------------------------------------------------------

class SingularMatrixError(ValueError):
    """Raised when inversion is attempted on a singular (non-invertible) matrix."""
    pass


def inverse(matrix):
    """Compute the inverse of an NxN matrix via Gauss-Jordan elimination.

    Returns a new NxN list-of-lists representing A^{-1}, or raises
    SingularMatrixError if the matrix is not invertible.

    Algorithm
    ---------
    1. Form the augmented matrix [A | I_n].
    2. For each column k = 0..n-1:
       a. Partial pivot: find the row with the largest absolute value in
          column k (at or below row k), and swap it into row k.
       b. If the pivot is effectively zero, the matrix is singular.
       c. Scale row k so the pivot becomes 1.
       d. Eliminate column k from all other rows.
    3. The right half of the augmented matrix is now A^{-1}.
    """
    n = _validate_square_matrix(matrix, "inverse")

    # [EDGE-CASE-09] Guard against very large matrices that would be slow.
    if n > MAX_INVERSION_SIZE:
        raise ValueError(
            f"inverse: matrix is {n}x{n} which exceeds the "
            f"{MAX_INVERSION_SIZE}x{MAX_INVERSION_SIZE} safety limit.  "
            f"Gauss-Jordan is O(n^3); this would require ~{n**3:,} operations."
        )

    # [EDGE-CASE-05] 1x1 special case: det = a, inverse = 1/a.
    # Avoids a division-by-zero if the single element is zero.
    if n == 1:
        val = matrix[0][0]
        if abs(val) < EPSILON:
            raise SingularMatrixError(
                "inverse: 1x1 matrix [[0]] is singular (element is zero)"
            )
        return [[1.0 / val]]

    # Build augmented matrix [A | I] using floats to avoid integer division.
    # [EDGE-CASE-07] Integer inputs are promoted to float here so that
    # division produces correct fractional results.
    aug = [
        [float(matrix[r][c]) for c in range(n)]
        + [1.0 if c == r else 0.0 for c in range(n)]
        for r in range(n)
    ]

    # Forward + backward elimination with partial pivoting.
    for k in range(n):
        # --- Partial pivoting -------------------------------------------
        # [EDGE-CASE-08] Without partial pivoting, a zero or near-zero
        # diagonal element causes division by ~0 and catastrophic loss of
        # precision.  We swap in the row with the largest absolute value
        # in column k to maximize numerical stability.
        max_val = abs(aug[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(aug[i][k]) > max_val:
                max_val = abs(aug[i][k])
                max_row = i
        if max_row != k:
            aug[k], aug[max_row] = aug[max_row], aug[k]

        pivot = aug[k][k]

        # [EDGE-CASE-06] If the best pivot is still effectively zero after
        # partial pivoting, the matrix is singular (or so close to singular
        # that the result would be numerically meaningless).
        if abs(pivot) < EPSILON:
            raise SingularMatrixError(
                f"inverse: matrix is singular or near-singular "
                f"(pivot at column {k} is {pivot:.2e}, below epsilon {EPSILON:.0e})"
            )

        # --- Scale pivot row so pivot becomes 1.0 -----------------------
        inv_pivot = 1.0 / pivot
        for j in range(2 * n):
            aug[k][j] *= inv_pivot

        # --- Eliminate column k from every other row --------------------
        for i in range(n):
            if i == k:
                continue
            factor = aug[i][k]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[i][j] -= factor * aug[k][j]

    # Extract the right half — that's A^{-1}.
    result = [[aug[r][n + c] for c in range(n)] for r in range(n)]

    # [EDGE-CASE-11] Snap near-zero entries to exactly 0.0 to avoid
    # confusing -3.6e-16 style artefacts in the output.
    for r in range(n):
        for c in range(n):
            if abs(result[r][c]) < EPSILON:
                result[r][c] = 0.0

    return result


# ---------------------------------------------------------------------------
# Verification utility
# ---------------------------------------------------------------------------

def multiply(a, b):
    """Multiply two NxN matrices and return the product."""
    n = len(a)
    return [
        [sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)]
        for i in range(n)
    ]


def is_identity(matrix, tol=1e-9):
    """Return True if the matrix is approximately the identity matrix."""
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(matrix[i][j] - expected) > tol:
                return False
    return True


# ---------------------------------------------------------------------------
# Heatmap UI
# ---------------------------------------------------------------------------

def _lerp_color(ratio):
    """Map a 0.0-1.0 ratio to an (R, G, B) gradient: blue -> green -> yellow -> red."""
    if ratio <= 0.33:
        t = ratio / 0.33
        r, g, b = int(0 + t * 0), int(100 + t * 155), int(255 - t * 155)
    elif ratio <= 0.66:
        t = (ratio - 0.33) / 0.33
        r, g, b = int(0 + t * 255), int(255), int(100 - t * 100)
    else:
        t = (ratio - 0.66) / 0.34
        r, g, b = 255, int(255 - t * 200), 0
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def _ansi_bg(r, g, b):
    """Return the ANSI escape sequence for a 24-bit background color."""
    return f"\033[48;2;{r};{g};{b}m"


ANSI_RESET = "\033[0m"


def print_heatmap(matrix, title="Heatmap"):
    """Print the matrix as a terminal heatmap with cells colored by value.

    The color scale maps the minimum value in the matrix to blue and the
    maximum value to red, with green and yellow in between.
    """
    n = len(matrix)
    flat = [v for row in matrix for v in row]
    lo, hi = min(flat), max(flat)
    val_range = hi - lo if hi != lo else 1

    col_width = max(len(f"{v:g}") for v in flat) + 2

    print(f"\n  {title}  (color: blue=min  green  yellow  red=max)")
    print(f"  {'─' * (col_width * n + 2)}")

    for row in matrix:
        cells = []
        for v in row:
            ratio = (v - lo) / val_range
            r, g, b = _lerp_color(ratio)
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "\033[30m" if brightness > 128 else "\033[97m"
            cell_text = f"{v:g}".center(col_width)
            cells.append(f"{_ansi_bg(r, g, b)}{fg}{cell_text}{ANSI_RESET}")
        print(f"  |{''.join(cells)}|")

    print(f"  {'─' * (col_width * n + 2)}")
    print(f"  Range: [{lo:g} .. {hi:g}]")


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def _fmt_matrix(matrix, precision=6):
    """Return a list of formatted row strings for a matrix."""
    lines = []
    for row in matrix:
        cells = []
        for v in row:
            if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
                cells.append(f"{int(v):>8}")
            elif isinstance(v, int):
                cells.append(f"{v:>8}")
            else:
                cells.append(f"{v:>10.{precision}f}")

        lines.append("  [" + ", ".join(cells) + "]")
    return lines


def _print_matrix(matrix, label="Matrix"):
    """Pretty-print a matrix with its determinant (if small enough)."""
    n = len(matrix)
    print(f"\n{label} ({n}x{n}):")
    for line in _fmt_matrix(matrix):
        print(line)
    if n <= MAX_DETERMINANT_SIZE:
        print(f"  Determinant = {determinant(matrix)}")


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def _demo_inversion(matrix, label):
    """Run inversion on a matrix and verify A * A^{-1} = I."""
    n = len(matrix)
    print(f"\n{'='*60}")
    print(f"  {label} ({n}x{n})")
    print(f"{'='*60}")

    print("\n  A =")
    for line in _fmt_matrix(matrix):
        print(line)

    try:
        inv = inverse(matrix)
    except (SingularMatrixError, ValueError) as e:
        print(f"\n  RESULT: {e}")
        return

    print("\n  A^(-1) =")
    for line in _fmt_matrix(inv):
        print(line)

    # Verification: A * A^{-1} should equal I
    product = multiply(matrix, inv)
    identity_ok = is_identity(product)
    status = "PASS" if identity_ok else "FAIL"
    print(f"\n  Verification  A * A^(-1) = I ?  [{status}]")

    if not identity_ok:
        print("  A * A^(-1) =")
        for line in _fmt_matrix(product):
            print(line)


def main():
    print("=" * 60)
    print("  Matrix Engine v3.0 — Determinant, Inversion & Heatmap")
    print("=" * 60)

    # ----------------------------------------------------------------
    # PART 1: Edge-case demonstrations
    # ----------------------------------------------------------------
    print("\n" + "#" * 60)
    print("  PART 1: Edge-Case Demonstrations")
    print("#" * 60)

    # [EDGE-CASE-01] Empty matrix
    print("\n--- [EDGE-CASE-01] Empty matrix ---")
    try:
        inverse([])
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-02] Non-square (ragged) matrix
    print("\n--- [EDGE-CASE-02] Non-square matrix ---")
    try:
        inverse([[1, 2, 3], [4, 5]])
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-03] Non-numeric entry
    print("\n--- [EDGE-CASE-03] Non-numeric entry ---")
    try:
        inverse([[1, "x"], [3, 4]])
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-04] NaN / Inf entries
    print("\n--- [EDGE-CASE-04] NaN entry ---")
    try:
        inverse([[1, float('nan')], [3, 4]])
    except ValueError as e:
        print(f"  Caught: {e}")

    print("\n--- [EDGE-CASE-04] Inf entry ---")
    try:
        inverse([[float('inf'), 2], [3, 4]])
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-05] 1x1 singular ([[0]])
    print("\n--- [EDGE-CASE-05] 1x1 singular matrix [[0]] ---")
    try:
        inverse([[0]])
    except SingularMatrixError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-06] Singular matrix (det = 0)
    print("\n--- [EDGE-CASE-06] Singular 3x3 (linearly dependent rows) ---")
    try:
        inverse([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    except SingularMatrixError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-07] Integer matrix — should produce correct float inverse
    _demo_inversion([[2, 1], [7, 4]], "[EDGE-CASE-07] Integer 2x2")

    # [EDGE-CASE-08] Near-zero pivot without pivoting would fail
    _demo_inversion(
        [[1e-18, 1], [1, 1]],
        "[EDGE-CASE-08] Near-zero leading element (needs pivoting)"
    )

    # [EDGE-CASE-09] Large matrix size guard
    print("\n--- [EDGE-CASE-09] Oversized matrix guard ---")
    try:
        huge = [[0] * 501 for _ in range(501)]
        inverse(huge)
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-10] Determinant size guard
    print("\n--- [EDGE-CASE-10] Determinant size guard ---")
    try:
        big = [[1 if i == j else 0 for j in range(13)] for i in range(13)]
        determinant(big)
    except ValueError as e:
        print(f"  Caught: {e}")

    # [EDGE-CASE-11] Floating-point dust cleanup
    print("\n--- [EDGE-CASE-11] Float precision cleanup ---")
    inv = inverse([[1, 2], [3, 4]])
    print(f"  inverse([[1,2],[3,4]]) =")
    for line in _fmt_matrix(inv):
        print(line)
    print(f"  (no -0.0 or 3.6e-16 artefacts in output)")

    # ----------------------------------------------------------------
    # PART 2: Successful inversions at various sizes
    # ----------------------------------------------------------------
    print("\n\n" + "#" * 60)
    print("  PART 2: Successful Inversions")
    print("#" * 60)

    _demo_inversion([[42]], "1x1 Matrix")

    _demo_inversion([[4, 7], [2, 6]], "2x2 Matrix")

    _demo_inversion(
        [[6, 1, 1], [4, -2, 5], [2, 8, 7]],
        "3x3 Matrix"
    )

    _demo_inversion(
        [[3, 2, 0, 1], [4, 0, 1, 2], [3, 0, 2, 1], [9, 2, 3, 1]],
        "4x4 Matrix"
    )

    _demo_inversion(
        [
            [2, 3, 1, 5, 4],
            [1, 0, 3, 2, 1],
            [0, 2, 4, 1, 3],
            [3, 1, 2, 0, 2],
            [1, 4, 0, 3, 1],
        ],
        "5x5 Matrix"
    )

    m6 = [
        [5, -3, 2, 7, 1, -4],
        [-2, 8, 0, 3, -1, 6],
        [4, 1, -5, 2, 9, 0],
        [0, -7, 3, 1, 4, -2],
        [6, 2, -1, -8, 0, 5],
        [-3, 4, 7, 0, -6, 1],
    ]
    _demo_inversion(m6, "6x6 Matrix")

    # ----------------------------------------------------------------
    # PART 3: Heatmap visualization
    # ----------------------------------------------------------------
    print("\n\n" + "#" * 60)
    print("  PART 3: Heatmap Visualization")
    print("#" * 60)

    m3 = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    print_heatmap(m3, "3x3 Original")
    print_heatmap(inverse(m3), "3x3 Inverse")
    print_heatmap(m6, "6x6 Original")
    print_heatmap(inverse(m6), "6x6 Inverse")


if __name__ == "__main__":
    main()
