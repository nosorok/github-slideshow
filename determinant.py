def determinant_3x3(matrix):
    """Compute the determinant of a 3x3 matrix using the rule of Sarrus."""
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Input must be a 3x3 matrix")

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def minor_3x3(matrix, row, col):
    """Return the 3x3 minor of a 4x4 matrix by removing the given row and column."""
    return [
        [matrix[r][c] for c in range(4) if c != col]
        for r in range(4) if r != row
    ]


def determinant_4x4(matrix):
    """Compute the determinant of a 4x4 matrix using Laplace expansion along the first row.

    For each element a[0][j] in the first row, we compute its cofactor:
        C(0, j) = (-1)^j * det(M(0, j))
    where M(0, j) is the 3x3 minor obtained by deleting row 0 and column j.

    The determinant is the sum:  a[0][0]*C(0,0) + a[0][1]*C(0,1) + a[0][2]*C(0,2) + a[0][3]*C(0,3)
    """
    if len(matrix) != 4 or any(len(row) != 4 for row in matrix):
        raise ValueError("Input must be a 4x4 matrix")

    det = 0
    for col in range(4):
        sign = (-1) ** col
        minor = minor_3x3(matrix, 0, col)
        det += sign * matrix[0][col] * determinant_3x3(minor)
    return det


if __name__ == "__main__":
    matrix = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    print(f"Matrix:")
    for row in matrix:
        print(f"  {row}")
    print(f"Determinant: {determinant_3x3(matrix)}")

    matrix2 = [
        [6, 1, 1],
        [4, -2, 5],
        [2, 8, 7],
    ]
    print(f"\nMatrix:")
    for row in matrix2:
        print(f"  {row}")
    print(f"Determinant: {determinant_3x3(matrix2)}")

    # 4x4 examples
    matrix3 = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]
    print(f"\n4x4 Matrix:")
    for row in matrix3:
        print(f"  {row}")
    print(f"Determinant: {determinant_4x4(matrix3)}")

    matrix4 = [
        [3, 2, 0, 1],
        [4, 0, 1, 2],
        [3, 0, 2, 1],
        [9, 2, 3, 1],
    ]
    print(f"\n4x4 Matrix:")
    for row in matrix4:
        print(f"  {row}")
    print(f"Determinant: {determinant_4x4(matrix4)}")
