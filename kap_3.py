# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes
"""

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

## I denna kod visualiseras vad som händer med serier, olika värden och hur de kan komma att konvergera, eller divergerar

# Först skapar vi en variabel som håller koll på den totala summan.

S = 0

# Följande variabel håller koll på hur många termer vi ska summera

N = 5

# Följande variabel ska hålla koll på hur många termer vi hittils har summerat.
# Vi stoppar när k=N (dvs., vi summerar ej termen då k=N).

k = 0

# Följande variabel ska stoppa while-loopen när det är dags.

continue_loop = True

# Nu summerar vi.

while continue_loop:
  a = k**2
  S = S + a
  plt.plot(k,S,"bo")
  k = k + 1
  if k == 5:
    continue_loop = False

# Vi visar fram resultatet numeriskt.

print("Med N = ",N,"blir S = ",S)

# Vi visar fram resultatet visuellt. 

plt.show()


def visualize_harmonic_series(expression, n):
    sequence = np.arange(1, n+1)
    harmonic_sum = np.cumsum(1 / sequence)

    plt.plot(sequence, harmonic_sum)
    plt.xlabel('n')
    plt.ylabel(expression)
    plt.title('Harmonic Series Visualization')
    plt.grid(False)
    plt.show()

n = int(input("Enter the number of terms (n): "))
visualize_harmonic_series("1/n", n)


def visualize_alternating_harmonic_series(n):
    sequence = np.arange(1, n+1)
    alternating_harmonic_sum = np.cumsum((-1)**(sequence+1) / sequence)

    plt.plot(sequence, alternating_harmonic_sum, "o")
    plt.xlabel('n')
    plt.ylabel('Partial Sum')
    plt.title('Alternating Harmonic Series Visualization')
    plt.grid(True)
    plt.show()

n = int(input("Enter the number of terms (n): "))
visualize_alternating_harmonic_series(n)




def find_n_for_sum(expression, target_value):
    def harmonic_partial_sum(n):
        sequence = np.arange(1, n+1)
        partial_sum = np.sum(eval(expression))
        return partial_sum

    n = 1
    while True:
        partial_sum = harmonic_partial_sum(n)
        if partial_sum >= target_value:
            return n
        n += 1

expression = input("Enter the expression for the partial sum: ")
target_value = float(input("Enter the target value for the partial sum: "))
n_required = find_n_for_sum(expression, target_value)
print("The number of terms needed: ", n_required)

