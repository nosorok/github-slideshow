def determinant_3x3(matrix):
    """Compute the determinant of a 3x3 matrix using the rule of Sarrus."""
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("Input must be a 3x3 matrix")

    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


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
