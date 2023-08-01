# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes

Det här är ett samlingsdokument för de alla olika funktioner vilka samlats under kursen MATA21. Funktioner med kort beskrivning listas nedan:
    
    # SERIER
    - series_terms:             Returnerar (printar) termerna för en serie/talföljd och för givna n
    - visualize_series:         Visualiserar en given serie/talföljd
    - find_n_series:            Beräknar för hur stora n för att uttrycket skall vara mindre än epsilon
    
    # FUNKTIONER
    - plot_function:            Plottar en given matematisk funktion
    - plot_piecewise_function:  Plottar en uppdelad (tvådelad) funktion 
    - evaluate_function:        Evaluerar ett matematisk uttryck för ett givet x-värde
    
    # DERIVATA OCH INTEGRALER
    - derivative_expression:    Återger derivatan för ett givet uttryck
    - integral_expression:      Återger den primitiva funktoinen för ett givet uttryck
    - function_and_derivative:  Plottar en given funktion och dess derivata. Återger även derivatan för uttrycket
    - riemann_sum_visualize:    Visuliserar Riemann-summorna grafiskt
    - riemann_chose_n:          Beräknar för hur stora N (antal indelningar) som Riemannsummorna beräknar area lika som integralen med givet epsilon    
    - riemann_vs_integral:      Visualiserar kopplingen mellan Riemann-summorna och Integralen samt beräknar hur stora N för att approximera integralen med fel < epsilon
    

"""

import numpy as np
import sympy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.integrate as integrate
import math

def series_terms(expression, n = 10, series_start=1):
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


def visualize_series(expression, n = 20, epsilon=None, series_start=1, manuell_linje=None):



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



def find_n_series(expression, epsilon, n = 0):
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








#### FUNKTIONER



def plot_function(expression, x_range=(-10, 10), num_points=1000, Grid=True, expression_2=None):
    x = np.linspace(x_range[0], x_range[1], num_points)
    try:
        y = eval(expression)
        plt.plot(x, y, label=f'Function: {expression}')
        plt.hlines(y=0, xmin=x_range[0], xmax=x_range[1], colors="black", linewidth=0.5)
        plt.vlines(x=0, ymin=min(y), ymax=max(y), colors="black", linewidth=0.5)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Graph of {expression}')

        if expression_2 is not None:
            y_2 = eval(expression_2)
            plt.plot(x, y_2, label=f'Function 2: {expression_2}')
            plt.vlines(x=0, ymin=min(y, y_2), ymax=max(y, y_2), colors="black", linewidth=0.5)

        if Grid:
            plt.grid(linewidth=0.3)
        else:
            plt.grid(False)

        # Place the legend outside the plot
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error:", e)


def plot_piecewise_function(expression_1, expression_2, bound_1, x_range=(-10, 10), num_points=1000, Grid=True):
    x = np.linspace(x_range[0], x_range[1], num_points)
    x_sym = sp.Symbol('x')

    # Define the piecewise function using SymPy's Piecewise
    f1 = sp.sympify(expression_1)
    f2 = sp.sympify(expression_2)
    piecewise_expr = sp.Piecewise((f1, x_sym < bound_1), (f2, x_sym >= bound_1))

    # Convert the piecewise expression to a NumPy function
    piecewise_func = sp.lambdify(x_sym, piecewise_expr, modules=["numpy"])

    # Evaluate the piecewise function for the given x values
    y = piecewise_func(x)

    plt.plot(x, y, label=f'Piecewise Function: $f(x) = {expression_1}$ för $x < {bound_1}$, $f(x) = {expression_2}$ för $x \geq {bound_1}$')
    plt.hlines(y=0, xmin=x_range[0], xmax=x_range[1], colors="black", linewidth=0.5)
    plt.vlines(x=0, ymin=min(y), ymax=max(y), colors="black", linewidth=0.5)
    plt.vlines(x=bound_1, ymin=min(y), ymax=max(y), colors="red", linestyle = 'dashed', linewidth=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Grafen för en uppdelad funktion')

    if Grid:
        plt.grid(linewidth=0.3)
    else:
        plt.grid(False)

    # Place the legend beneath the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()


    
        


def evaluate_function(funktion, x):
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
  print(evaluate_function(funktion, x))







#### DERIVATA OCH INTEGRALER


## DERIVATA

def derivative_expression(expression):
    x = sp.Symbol('x')
    derivative_func = sp.diff(expression, x)
    critical_points = sp.solve(derivative_func, x, complex=True)
    print('Derivatan av funktionen är:', derivative_func)
    print('Kritiska punkter:', critical_points)


def function_and_derivative(expression="x**2", x_val=None):
    x = sp.Symbol('x')
    expr = sp.sympify(expression)
    derivative_expr = sp.diff(expr, x)

    # Convert the SymPy expressions to Python functions using lambdify
    func = sp.lambdify(x, expr, 'numpy')
    derivative_func = sp.lambdify(x, derivative_expr, 'numpy')

    # Generate x values for the plot
    x_vals = np.linspace(-10, 10, 500)

    # Evaluate the functions at the x values
    y_vals = func(x_vals)
    dy_vals = derivative_func(x_vals)

    # Create the plot with subplots
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Plot the original function on the top
    axes[0].plot(x_vals, y_vals, label='Function')
    axes[0].set_title('Function: ' + str(expr))
    axes[0].legend()

    # Plot the derivative on the bottom
    axes[1].plot(x_vals, dy_vals, label='Derivative')
    axes[1].set_title('Derivative: ' + str(derivative_expr))
    axes[1].legend()

    if x_val is not None:
        try:
            value = derivative_func(x_val)
        except TypeError:
            print('Error: Unable to evaluate the derivative at the specified value. Please provide a valid value for x.')
        else:
            axes[1].axhline(value, color='red', linestyle='--', label='Derivative at x={}: {}'.format(x_val, value))
            axes[1].legend()

    plt.tight_layout()
    plt.show()



## INTEGRALER


def riemann_sum_visualize(expression, lower_bound, upper_bound, N=100, epsilon=None):
    f = lambda x: eval(expression)
    
    # Calculate the exact integral value using scipy's quad function
    value = integrate.quad(f, lower_bound, upper_bound)[0]
    
    # Define the Riemann sum function
    riemann_sum = lambda a, b, n: sum(f((a + (i / n) * (b - a))) * (b - a) / n for i in range(n))
    
    # Compute Riemann sums for different values of n
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
    plt.axhline(y=value + epsilon, color="red", linestyle='dashed') if epsilon else None
    plt.axhline(y=value - epsilon, color="red", linestyle='dashed') if epsilon else None
    ax2.set_ylabel('Riemann Sum')
    ax2.set_title("Riemann-summa")
    ax2.grid(True)

    plt.show()

    if epsilon is not None:
        n = 1
        prev_sum = 0
        current_sum = riemann_sum(lower_bound, upper_bound, n)

        while abs(current_sum - prev_sum) > epsilon:
            n *= 2
            x_values = np.linspace(lower_bound, upper_bound, n + 1)
            prev_sum = current_sum
            current_sum = riemann_sum(lower_bound, upper_bound, n)

        print("Det exakta integrerade värdet är: ", value)
        print("Värdet av Riemann-summan =", current_sum, "då n =", n)
        print("Då n =", n, "(antalet rektanglar) approximerar Riemann-summan integralen med epsilon <", epsilon)


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






def integral_expression(expression, lower_bound=None, upper_bound=None):
    x = sp.Symbol('x')
    primitiva_funktionen = sp.integrate(expression, x)
    
    if lower_bound is not None and upper_bound is not None:
        f = lambda x: eval(expression)
        try:
            value, _ = integrate.quad(f, lower_bound, upper_bound)
        except TypeError:
            print('Error: Unable to compute the definite integral. Please specify valid lower and upper bounds.')
        else:
            primitive_upper = primitiva_funktionen.subs(x, upper_bound)
            primitive_lower = primitiva_funktionen.subs(x, lower_bound)
            primitive_with_substitution = primitive_upper - primitive_lower

            print('Den primitiva funktionen: ', primitiva_funktionen)
            print('Den primitiva funktionen utvärderad F(b) - F(a):', primitive_upper, '-', primitive_lower,
                  ' = ', primitive_with_substitution)
            print('Definite integral från {} till {}: '.format(lower_bound, upper_bound), value)
    else:
        print('Den primitiva funktionen: ', primitiva_funktionen)



