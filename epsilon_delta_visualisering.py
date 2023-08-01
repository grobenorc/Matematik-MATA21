# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes
"""
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def f(x):
    return np.sin(x)

def plot_epsilon_delta(epsilon, delta):
    # Define the limit point L
    L = 0

    # Define the range of x values and the point c
    xmin, xmax = -5, 5
    c = 0

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the function and the limit point
    x = np.linspace(xmin, xmax, 1000)
    y = f(x)
    ax.plot(x, y, label='f(x)')
    ax.axhline(L, color='k', linestyle='--', label='Limit L')

    # Plot the horizontal lines at L +/- epsilon
    ax.axhline(L + epsilon, color='r', linestyle='--', label='L + epsilon')
    ax.axhline(L - epsilon, color='r', linestyle='--', label='L - epsilon')

    # Plot the vertical line at c
    ax.axvline(c, color='g', linestyle='--', label='x approaches c')

    # Plot the buffer zone around c
    ax.axvline(c + delta, color='b', linestyle='--', label='c + delta')
    ax.axvline(c - delta, color='b', linestyle='--', label='c - delta')

    # Plot the region where |f(x) - L| < epsilon
    x_region = np.linspace(c - delta, c + delta, 100)
    y_region = f(x_region)
    ax.fill_between(x_region, L - epsilon, L + epsilon, where=np.abs(y_region - L) < epsilon, alpha=0.3, color='gray', label='|f(x) - L| < epsilon')

    # Set the plot title and axis labels
    ax.set_title('Epsilon-Delta Criterion')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')

    # Add a legend to the plot
    ax.legend()

    plt.show()

# Create sliders for epsilon and delta
epsilon_slider = FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5, description='Epsilon:')
delta_slider = FloatSlider(min=0.1, max=1.0, step=0.1, value=0.5, description='Delta:')

# Create the interactive plot
interact(plot_epsilon_delta, epsilon=epsilon_slider, delta=delta_slider)