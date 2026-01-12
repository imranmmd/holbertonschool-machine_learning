#!/usr/bin/env python3
"""Module for polynomial integration"""


def poly_integral(poly, C=0):
    """Calculates the integral of a polynomial"""
    if not isinstance(poly, list) or not isinstance(C, (int, float)):
        return None

    if not all(isinstance(c, (int, float)) for c in poly):
        return None

    integral = [C]

    for i, coef in enumerate(poly):
        value = coef / (i + 1)
        if value.is_integer():
            value = int(value)
        integral.append(value)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
