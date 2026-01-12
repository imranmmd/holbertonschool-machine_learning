#!/usr/bin/env python3
"""Module for polynomial derivative"""


def poly_derivative(poly):
    """Calculates the derivative of a polynomial"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    derivative = [i * poly[i] for i in range(1, len(poly))]

    if not derivative or all(c == 0 for c in derivative):
        return [0]

    return derivative
