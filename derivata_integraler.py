# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes

Den här koden behandlar i huvudsak konceptet kring derviata och integraler. På ett intuitivt sätt försöks här förklaras bakgrinden till det, vad Riemannsummor säger/gör medmera.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math as m


def visualize_limit(function, x_start, x_end, n, epsilon=None):
  """This function visualizes the limit of a function.

  Args:
    function: The function to visualize.
    x_start: The starting value of x.
    x_end: The ending value of x.
    n: The number of points to plot.
    epsilon: The tolerance.

  Returns:
    A plot of the function.
  """

  x = np.linspace(x_start, x_end, n)
  y = [function(x) for x in x]

  # Plot the function
  plt.figure(figsize=(10, 5))
  plt.plot(x, y)
  plt.axhline(y=epsilon, color="blue") if epsilon else None
  plt.xlabel("x")
  plt.ylabel("y")
  plt.title("Limit of the function")

  # Add the expression to the middle of the plot at the bottom center
  plt.figtext(0.5, 0.02, function, va="top", ha="center", fontsize=14)

  plt.tight_layout()
  plt.show()



def rectangle_rule(expression, N, epsilon=None, series_start=1):
  """This function calculates the value of a definite integral using the
    rectangle rule.

  Args:
    expression: The mathematical expression to integrate.
    N: The number of rectangles.
    epsilon: The desired precision.
    series_start: The k-value from which the series should start.

  Returns:
    The value of the integral.
  """

  k = series_start
  S = 0

  continue_loop = True

  while continue_loop:
    
    S = (expression(k) / (1 + (k / N)**2)) * (1 / N) + S
    k = k + 1

    if k == N:
      continue_loop = False

  if epsilon is not None:
    S = S + epsilon

  return S




def rectangle_rule2(expression, N=100, epsilon=None):
  """This function calculates the value of a definite integral using the
    rectangle rule.

  Args:
    expression: The mathematical expression to integrate.
    N: The number of rectangles.
    epsilon: The desired precision.

  Returns:
    The value of the integral.
  """

  if epsilon is None:
    # Use a default precision of 1e-6.
    epsilon = 1e-6

  S_old = 0
  k = 1
  S = 0

  continue_loop = True

  while continue_loop:
    
    S = (expression(k) / (1 + (k / N)**2)) * (1 / N) + S
    k = k + 1

    if k == N:
      continue_loop = False

    # Check if the precision is reached.
    if abs(S - S_old) < epsilon:
      break

    S_old = S

  return S


def main():
  # Calculate the value of the integral using different values of N.
  N_values = []
  errors = []
  for N in range(1, 1001):
    S = rectangle_rule(lambda x: 1 / (1 + x**2), N, epsilon=1e-6)
    error = abs(S - m.pi / 4)
    N_values.append(N)
    errors.append(error)

  # Plot the errors.
  plt.plot(N_values, errors)
  plt.xlabel("Number of rectangles")
  plt.ylabel("Error")
  plt.show()
