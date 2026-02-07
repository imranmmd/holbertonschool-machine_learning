#!/usr/bin/env python3
"""
Adjugate matrix
"""


def determinant(matrix):
    """Helper function to compute determinant"""

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(n):
        minor = [
            row[:col] + row[col + 1:]
            for row in matrix[1:]
        ]
        det += ((-1) ** col) * matrix[0][col] * determinant(minor)

    return det


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix"""

    # Type check
    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    # Square & non-empty check
    n = len(matrix)
    if n == 0 or not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    # Special case: 1x1 matrix
    if n == 1:
        return [[1]]

    # Compute cofactor matrix
    cofactor = []
    for i in range(n):
        row = []
        for j in range(n):
            submatrix = [
                r[:j] + r[j + 1:]
                for k, r in enumerate(matrix) if k != i
            ]
            row.append(((-1) ** (i + j)) * determinant(submatrix))
        cofactor.append(row)

    # Transpose cofactor matrix → adjugate
    adjugate_matrix = [
        [cofactor[j][i] for j in range(n)]
        for i in range(n)
    ]

    return adjugate_matrix
