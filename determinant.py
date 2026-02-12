"""
Matrix Determinant Calculator v2.0

Computes the determinant of any NxN square matrix using recursive
Laplace (cofactor) expansion along the first row.

Includes a terminal heatmap UI that color-codes matrix cells based
on their value relative to the matrix's value range.
"""

import sys


# ---------------------------------------------------------------------------
# Core determinant logic
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
    n = len(matrix)
    if n == 0:
        raise ValueError("Matrix must not be empty")
    if any(len(row) != n for row in matrix):
        raise ValueError(f"Matrix must be square (got rows of varying length)")

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
            # Use dark text on light backgrounds, light text on dark
            brightness = 0.299 * r + 0.587 * g + 0.114 * b
            fg = "\033[30m" if brightness > 128 else "\033[97m"
            cell_text = f"{v:g}".center(col_width)
            cells.append(f"{_ansi_bg(r, g, b)}{fg}{cell_text}{ANSI_RESET}")
        print(f"  |{''.join(cells)}|")

    print(f"  {'─' * (col_width * n + 2)}")
    print(f"  Range: [{lo:g} .. {hi:g}]")


# ---------------------------------------------------------------------------
# Demo / CLI
# ---------------------------------------------------------------------------

def _print_matrix(matrix, label="Matrix"):
    """Pretty-print a matrix with its determinant."""
    n = len(matrix)
    print(f"\n{label} ({n}x{n}):")
    for row in matrix:
        print(f"  {row}")
    print(f"  Determinant = {determinant(matrix)}")


def main():
    # --- 1x1 ---
    m1 = [[42]]
    _print_matrix(m1, "1x1 Matrix")

    # --- 2x2 ---
    m2 = [[4, 6], [3, 8]]
    _print_matrix(m2, "2x2 Matrix")

    # --- 3x3 (singular) ---
    m3a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    _print_matrix(m3a, "3x3 Matrix (singular)")

    # --- 3x3 ---
    m3b = [[6, 1, 1], [4, -2, 5], [2, 8, 7]]
    _print_matrix(m3b, "3x3 Matrix")

    # --- 4x4 (singular) ---
    m4a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    _print_matrix(m4a, "4x4 Matrix (singular)")

    # --- 4x4 ---
    m4b = [[3, 2, 0, 1], [4, 0, 1, 2], [3, 0, 2, 1], [9, 2, 3, 1]]
    _print_matrix(m4b, "4x4 Matrix")

    # --- 5x5 ---
    m5 = [
        [2, 3, 1, 5, 4],
        [1, 0, 3, 2, 1],
        [0, 2, 4, 1, 3],
        [3, 1, 2, 0, 2],
        [1, 4, 0, 3, 1],
    ]
    _print_matrix(m5, "5x5 Matrix")

    # --- Heatmap demos ---
    print("\n" + "=" * 50)
    print("  HEATMAP VISUALIZATION")
    print("=" * 50)

    print_heatmap(m3b, "3x3 Heatmap")
    print_heatmap(m4b, "4x4 Heatmap")
    print_heatmap(m5, "5x5 Heatmap")

    # A 6x6 matrix to showcase larger heatmap
    m6 = [
        [5, -3, 2, 7, 1, -4],
        [-2, 8, 0, 3, -1, 6],
        [4, 1, -5, 2, 9, 0],
        [0, -7, 3, 1, 4, -2],
        [6, 2, -1, -8, 0, 5],
        [-3, 4, 7, 0, -6, 1],
    ]
    _print_matrix(m6, "6x6 Matrix")
    print_heatmap(m6, "6x6 Heatmap")


if __name__ == "__main__":
    main()
