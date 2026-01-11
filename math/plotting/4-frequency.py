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

    # Create bins from 0 to 100 in steps of 10
    bins = np.arange(0, 110, 10)  # 0, 10, 20, ..., 100

    # Plot histogram
    plt.hist(student_grades, bins=bins, edgecolor='black')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.show()
