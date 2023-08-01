# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sympy as sp
import math

def riemann_sum_visualize(expression, lower_bound, upper_bound, N=100, epsilon = None):
    f = lambda x: eval(expression)
    value = integrate.quad(f, lower_bound, upper_bound)[0]
    riemann_sum = lambda a, b, n: sum(f((a + (i / n) * (b - a))) * (b - a) / n for i in range(n))
    n_values = np.arange(1, N + 1)
    riemann_sums = [riemann_sum(lower_bound, upper_bound, n) for n in n_values]
    

    # Create two subplots
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the function
    x_values = np.arange(lower_bound, upper_bound, 0.01)
    y_values = [f(x) for x in x_values]
    ax1.plot(x_values, y_values)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(r'Funktionen $f(x) = {}$'.format(expression))
    ax1.grid(True)

    # Plot the Riemann sums
    ax2.plot(n_values, riemann_sums, marker='o')
    ax2.set_xlabel('Antal rektanglar (n)')
    plt.axhline(y=value + epsilon, color="red", linestyle = 'dashed') if epsilon else None
    plt.axhline(y=value + epsilon, color="red", linestyle = 'dashed') if epsilon else None
    ax2.set_ylabel('Riemann Sum')
    ax2.set_title("Riemann-summa")
    ax2.grid(True)
    
    plt.show()

    # Calculate the exact value using scipy.integrate.quad()
    
    print('Det exakta integrerade värdet är:', value)
    print('Antal indelningar = ', n_values[-1])
    print('Riemann-summor =', riemann_sums[-1])

# Example usage:
# riemann_sum_visualize("x**2", 0, 4, epsilon = 1/10)



def riemann_chose_n(expression, lower_bound, upper_bound, epsilon):
    f = lambda x: eval(expression)
    value = integrate.quad(f, lower_bound, upper_bound)
    
    n = 1
    prev_sum = 0
    current_sum = (f(lower_bound) + f(upper_bound)) * (upper_bound - lower_bound) / 2

    while abs(current_sum - prev_sum) > epsilon:
        n *= 2
        dx = (upper_bound - lower_bound) / n
        x_values = np.linspace(lower_bound, upper_bound, n+1)
        prev_sum = current_sum
        # current_sum = sum(f(x_values[i]) * dx for i in range(n))
        current_sum = np.sum(np.fromiter((f(x_values[i]) * dx for i in range(n)), dtype=float))

    print("Det exakta integrerade värdet är: ", value[0])
    print("Värdet av riemansummorna = ", current_sum, " då n = ", n)
    print("Då n = ", n, " (antalet rektanglar) approximerar Riemannsummorna integralen med epsilon < ", epsilon)


def riemann_vs_integral(expression, lower_bound, upper_bound, N=100, epsilon=None):
    """Denna funktion illustrerar kopplingen mellan Riemann-summor och integranden. Felmeddelande kommer då epsilon inte är angivet men graferna fungerar som det ska.
    
    Argument:
      expression: Den matematiska uttrycket som ska evalueras. Använd math.expression() i expression-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      lower_bound: Nedre gränsen av integralen.
      upper_bound: Övre gränsen av integralen.
      N: Antal uppdelningar, rektanglar, utfrån vilket area beräknas. Intervallen är lika delade i intervallet.
      epsilon: Ange värde av epsilon. Det här är alltså skillnaden från den matematiska beräknade bestämda integralen och Riemansummorna. Kan användas för att undersöka hur stort N (alltså indelaningar/rektanglar av intervallet) som behövs för att uppnå önskad precision.
      
      
    Returnerar:
      En plot av funktionen.
      En plot att de summerade Riemann-summorna.
      Om epsilon givet, hur stort N nödvändigt för given precision = epsilon
    """
    
    f = lambda x: eval(expression)
    value = integrate.quad(f, lower_bound, upper_bound)[0]
    x = sp.Symbol('x')
    primitiva_funktionen = sp.integrate(expression, x)
    primitive_lower =primitiva_funktionen.subs(x, lower_bound)
    primitive_upper = primitiva_funktionen.subs(x, upper_bound)
    primitive_with_substitution = primitiva_funktionen.subs(x, upper_bound) - primitiva_funktionen.subs(x, lower_bound)
    
    if epsilon is not None:
        value = round(value, math.ceil(np.log10(1/(epsilon)))) 
    else: 
        value = round(value, 5)
    
    def riemann_sum(a, b, n):
        dx = (b - a) / n
        x_values = np.linspace(a, b, n+1)
        return np.sum(np.fromiter((f(x_values[i]) * dx for i in range(n)), dtype=float))

    n_values = np.arange(1, N + 1)
    riemann_sums = [riemann_sum(lower_bound, upper_bound, n) for n in n_values]

    if epsilon is not None:
        rieman_last = round(riemann_sums[-1], math.ceil(np.log10(1/(epsilon)))) 
    else: 
        rieman_last = round(riemann_sums[-1], 5)
    
    
    # Create two subplots
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot the function
    x_values = np.arange(lower_bound, upper_bound, 0.01)
    y_values = [f(x) for x in x_values]
    ax1.plot(x_values, y_values)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title(r'Funktionen $f(x) = {}$'.format(expression))
    ax1.grid(True)

    # Plot the Riemann sums
    ax2.plot(n_values, riemann_sums, marker='o')
    ax2.set_xlabel('Antal rektanglar (n)')
    plt.axhline(y=value + epsilon, color="red", linestyle = 'dashed') if epsilon else None
    plt.axhline(y=value - epsilon, color="red", linestyle = 'dashed') if epsilon else None
    ax2.set_ylabel('Riemann Sum')
    ax2.set_title("Riemann-summa")
    ax2.grid(True)
    
    plt.show()
    
    n = 1
    prev_sum = 0
    current_sum = (f(lower_bound) + f(upper_bound)) * (upper_bound - lower_bound) / 2

    while abs(current_sum - prev_sum) > epsilon:
        n *= 2
        dx = (upper_bound - lower_bound) / n
        x_values = np.linspace(lower_bound, upper_bound, n+1)
        prev_sum = current_sum
        # current_sum = sum(f(x_values[i]) * dx for i in range(n))
        current_sum = np.sum(np.fromiter((f(x_values[i]) * dx for i in range(n)), dtype=float))
    
    print('Den primitiva funktionen: ', primitiva_funktionen)
    print('Den primitiva funktionen utvärderad F(b) - F(a):', primitive_upper, '-', primitive_lower, ' = ', primitive_with_substitution)
    print('\nDet exakta (approximerade) integrerade värdet är ≈', value)
    print('N, Antal indelningar = ', n_values[-1])
    print('Riemann-summor vid N = ', n_values[-1], ' ≈', rieman_last)
    print("\nDå N (antalet indelningar) = ", n, "  approximerar Riemannsummorna integralen med epsilon < ", epsilon)
    

