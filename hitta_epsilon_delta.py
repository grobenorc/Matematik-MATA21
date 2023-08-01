# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes
"""
import numpy as np
import math
import matplotlib.pyplot as plt

continue_searching = True


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



def find_n(expression, epsilon):
  """Denna funktion hittar n för vilket uttrycket är mindre än epsilon.

  Argument:
    expression: Det matematiska uttrycket.
    epsilon: Tolerans.

  Returnerar:
    Det minsta n för vilket uttrycket är mindre än epsilon.
  """

  step_size = 10
  n = 0

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
     









