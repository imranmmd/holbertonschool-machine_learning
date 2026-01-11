#!/usr/bin/env python3
"""
Module to plot a histogram of student grades for Project A.
"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades with:
    - X-axis: 'Grades'
    - Y-axis: 'Number of Students'
    - Bins every 10 units (0-100)
    - Bars outlined in black
    - Title: 'Project A'
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    plt.hist(
        student_grades,
        bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        edgecolor='black'
    )
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title("Project A")
    plt.xlim(0, 100)
    plt.xticks(np.arange(0, 101, 10))
    plt.ylim(0, 30)
    plt.show()
