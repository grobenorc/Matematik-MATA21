# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import math
'''
Matematiska funktioner med math:
    math.fabs(x) = aboslutvärdet
    math.factorial(x) = x!
    math.sqrt(x) = kvadratroten
    math.log(x) = ln(x)   ( math.log(x, bas) = log(x) med bastal 'bas')
    math.exp(x) = e^x
    math.cos(x) = cosinus av x
    math.acos(x) = arc cosinus av x
    math.sin(x) = sinus av x
    math.asin(x) = arc sinus av x
    math.tan(x) = tangens av x
    math.atan(x) = arc tangens av x
'''

# def calculate_series_terms(expression, n, series_start=1):
#     """Denna funktion beräknar termerna i en matematisk serie med en given expression och ett givet antal termer.
    
#     Argument:
#       expression: Den matematiska expressionen som ska beräknas. Använd math.expression() i expression-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
#       n: Antalet termer som ska beräknas.
#       series_start: Från vilken k-värde serien ska starta.
    
#     Returnerar:
#       En lista med beräknade termer.
#     """

#     x = range(series_start, n + 1)
#     terms = [eval(expression.replace("k", str(int(k)))) for k in x]
#     # Print the first 9 sums and the last sum
#     if n <= 10:
#         print("Serie: ", terms)
#     else:
#         print("Start serie: ", terms[:10])
#         print("Slut serie: ", terms[-1])


def calculate_series_terms(expression, n, series_start=1):
    """Denna funktion beräknar termerna i en matematisk serie med en given expression och ett givet antal termer.
    
    Argument:
      expression: Den matematiska expressionen som ska beräknas. Använd math.expression() i expression-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska beräknas.
      series_start: Från vilken k-värde serien ska starta.
    
    Returnerar:
      En lista med beräknade termer.
    """

    if n <= 10:
        print("Serie: ", [eval(expression.replace("k", str(int(k)))) for k in range(series_start, n + 1)])
    else:
        first_10_terms = [eval(expression.replace("k", str(int(k)))) for k in range(series_start, 11)]
        last_term = eval(expression.replace("k", str(n)))
        print("Start serie: ", first_10_terms)
        print("Slut serie: ", last_term)



def visualize_series_terms(expression, n, epsilon=None, series_start = 1, manuell_linje = None):
    """Denna funktion visualiserar en matematisk serie med ett given uttryck (serie a_k), ett givet antal termer och ett givet epsilon.
    
    Argument:
      expression: Den matematiska uttrycket som ska visualiseras. Använd math.expression() i expression-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska visualiseras.
      epsilon: Värdet av epsilon. Om epsilon inte anges kommer det att ignoreras.
      series_start: Från vilket k-värde serien ska starta.
    
    Returnerar:
      En plot av serien.
    """

    x = range(series_start, n + 1)
    y = [eval(expression.replace("k", str(int(k)))) for k in x]
    plt.plot(x, y, "ro")
    plt.axhline(y=epsilon, color="blue") if epsilon else None
    plt.xlabel("k")
    plt.ylabel("a_k")
    plt.suptitle("Matematisk serie")
    plt.title(expression)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()


def calculate_series_sums(expression, n, series_start=1):
    """Denna funktion beräknar summan av en matematisk serie med ett givet uttryck och ett givet antal termer.
    
    Argument:
      uttryck: Det matematiska uttrycket som ska beräknas. Använd math.uttryck() i uttryck-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska beräknas.
      series_start: Från vilket k-värde serien ska starta.
    
    Returnerar:
      None. Skriver ut de beräknade seriesummorna.
    """
    x = range(series_start, n + 1)
    y = [eval(expression.replace("k", str(int(k)))) for k in x]
    cumulative_sum = np.cumsum(y)
    
    # Print the first 9 sums and the last sum
    if n <= 10:
        print("Serie: ", cumulative_sum)
    else:
        print("Start serie: ", cumulative_sum[:10])
        print("Slut serie: ", cumulative_sum[-1])


def visualize_series_sum(expression, n, series_start = 1, manuell_linje= None):
    """Denna funktion visualiserar summan av en matematisk serie med ett givet uttryck och ett givet antal termer.
    
    Argument:
      uttryck: Det matematiska uttrycket som ska visualiseras. Använd math.uttryck() i uttryck-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska visualiseras.
      series_start: Från vilket k-värde serien ska starta.
    
    Returnerar:
      En plot av seriesumman.
    """

    x = range(series_start, n + 1)
    y = [eval(expression.replace("k", str(int(k)))) for k in x]
    cumulative_sum = np.cumsum(y)
    plt.plot(x, cumulative_sum, "ro")
    plt.axhline(y = manuell_linje, color = "blue") if manuell_linje else None
    plt.xlabel("k")
    plt.ylabel("Summa av serien")
    plt.title("Summa av den Matematiska serien")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))  # Set x-axis tick locations to integers
    plt.show()


def visualize_series(expression, n, epsilon=None, series_start=1, manuell_linje=None):



    """Denna funktion visualiserar en matematisk serie med ett givet uttryck, ett givet antal termer och ett givet epsilon.

    Argument:
      expression: Det matematiska uttrycket som ska visualiseras. Använd math.uttryck() i uttryck-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska visualiseras.
      epsilon: Värdet av epsilon. Om epsilon inte anges kommer det att ignoreras.
      series_start: Från vilket k-värde serien ska starta.

    Returnerar:
      En plot av serien.
    """

    x = range(series_start, n + 1)
    y = [eval(expression.replace("k", str(int(k)))) for k in x]
    cumulative_sum = np.cumsum(y)

    # Plot the terms of the series
    plt.figure(figsize=(10, 5))
    plt.figtext(0.5, 1, "Matematisk serie", va="bottom", ha="center", fontsize=20)
    plt.subplot(121)
    plt.plot(x, y, "ro")
    plt.axhline(y=epsilon, color="blue") if epsilon else None
    plt.xlabel("k")
    plt.ylabel("a_k")
    plt.title("Värdet på uttrycket")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the sum of the series
    plt.subplot(122)
    plt.plot(x, cumulative_sum, "ro")
    plt.axhline(y=manuell_linje, color="blue") if manuell_linje else None
    plt.xlabel("k")
    plt.ylabel("Summa")
    plt.title("Summa av serien")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the expression to the middle of the plots at the bottom center
    plt.figtext(0.5, 0.02, expression, va="top", ha="center", fontsize=14)

    plt.tight_layout()
    plt.show()


# def find_n(expression, epsilon):
#   """Denna funktion hittar n för vilket uttrycket är mindre än epsilon.

#   Argument:
#     expression: Det matematiska uttrycket.
#     epsilon: Tolerans.

#   Returnerar:
#     Det minsta n för vilket uttrycket är mindre än epsilon.
#   """

#   continue_searching = True
#   n = 0

#   while continue_searching:
#     n = n + 1
#     value = eval(expression.replace("k", str(n)))
#     if value < epsilon:
#       continue_searching = False

#   print("Då n >=", n, "är uttrycket", expression, " < epsilon = ", epsilon)


def find_n(expression, epsilon, n = 0):
  """Denna funktion hittar n för vilket uttrycket är mindre än epsilon.

  Argument:
    expression: Det matematiska uttrycket.
    epsilon: Tolerans.

  Returnerar:
    Det minsta n för vilket uttrycket är mindre än epsilon.
  """

  step_size = 10
  n = n

  # First, try every tenth value of n
  while True:
    n += step_size
    value = eval(expression.replace("k", str(n)))
    if value < epsilon:
      # Refine the solution within the last 10 values of n
      for i in range(n - step_size, n + 1):
        value = eval(expression.replace("k", str(i)))
        if value < epsilon:
          return "Då k >= {} är uttrycket {} < epsilon = {}".format(i, expression, epsilon)
      # If we reach here, then the expression is still greater than epsilon
      # for all values of n between n - step_size and n + 1.
      # Therefore, we need to increase the step size.
      step_size *= 10


