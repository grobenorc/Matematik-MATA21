# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:24:57 2023

@author: claes
"""

import matplotlib.pyplot as plt
import numpy as np

"""
np.log(x) = ln(x)
np.log10(x) = log_10(x)
np.exp(x) = e^x

"""



def plot_math_expression(expression, x_range=(-10, 10), num_points=1000, Grid = True):
    x = np.linspace(x_range[0], x_range[1], num_points)
    try:
        y = eval(expression)
        plt.plot(x, y)
        plt.hlines(y = 0, xmin= x_range[0], xmax = x_range[1], colors="black", linewidth = 0.5)
        plt.vlines(x = 0, ymin = min(y), ymax = max(y), colors="black", linewidth = 0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Graph of {expression}')
        
        if Grid:
            plt.grid(linewidth = 0.3)
        else:
            plt.grid(False)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error:", e)


def utvärdera_matematisk_funktion(funktion, x):
  """Utvärderar den givna matematiska funktionen för den givna x.

  Argument:
    funktion: Den matematiska funktionen att utvärdera.
    x: Värdet att utvärdera funktionen vid.

  Returnerar:
    Funktionens värde vid x.
  """

  try:
    return eval(funktion)
  except Exception as e:
    print("Fel:", e)
    return None
  print(utvärdera_matematisk_funktion(funktion, x))

