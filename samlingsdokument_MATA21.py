# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:48:56 2023
@author: claes

Det här är ett samlingsdokument för de alla olika funktioner vilka samlats under kursen MATA21. Funktioner med kort beskrivning listas nedan:
    
    # SERIER
    - series_terms:                 Returnerar (printar) termerna för en serie/talföljd och för givna n
    - series_plot:                  Visualiserar en given serie/talföljd
    - series_find_n:                Beräknar för hur stora n för att uttrycket skall vara mindre än epsilon
    - series_comparison_plot:       Jämför två talföljder, detta är en typ av visualisering för jämförelsetestet
    
    # FUNKTIONER
    - function_Inverse:             Returnerar inversa funktionen av ett uttryck
    - function_plot:                Plottar en given matematisk funktion
    - function_plot_piecewise:      Plottar en uppdelad (tvådelad) funktion 
    - function_evaluate:            Evaluerar ett matematisk uttryck för ett givet x-värde
    - function_limit calculate:     Beräknar gränsvärdet för en given funktion
    - function_and_derivative:      Visualiserar funktionen och dess derivata i en plot, samt ger det deriverade uttrycket
    
    # DERIVATA OCH INTEGRALER
    - derivative:                   Returnerar derivatan av funktionen, gör en plot och kan hitta kritiska punkter
    - derivative_expression:        Returnerar derivatan för ett givet uttryck
    - derivative_plot:              Ritar funktionen, beräknar derivatan och plottar sedan båda.
    - derivative_function_plot:     -||-
    - mean_value_theorem:           Visualiserar Mean Value Theorem
    - intermediate_value_theorem:   Visualiserar Intermediate Value Theorem 
    - second_derivative_test:       Visualiserar second derivative test (test för extrempunkter)
    - teckentabell:                 Gör en teckentabell för ett givet uttryck.
    
    - plot_differential_eq:         Visualiserar en differentialekvation

    # Integraler
    - integral_expression:          Returnerar den primitiva funktoinen för ett givet uttryck
    - integral_and_function_plot:   Visualiserar funktionen och dess primitiva funktione i en plot, samt ger den primitiva funktionen.
    - riemann_sum_plot:             Visuliserar Riemann-summorna grafiskt
    - riemann_chose_n:              Beräknar för hur stora N (antal indelningar) som Riemannsummorna beräknar area lika som integralen med givet epsilon    
    - riemann_vs_integral:          Visualiserar kopplingen mellan Riemann-summorna och Integralen samt beräknar hur stora N för att approximera integralen med fel < epsilon
    
    
    - taylor_polynomial_symbolic:  Ger Taylor-utvecklingen av en funktion
    - taylor_polynomial_plot:      Visualiserar taylor polyno med given grad
    - taylor_polynomial_multiple_degrees:   Visualiserar flere grader av polynom
    
    - function_derivative_integral_plot: Plottar funktion, derivata, integral
    
    # Matematik
    - get_alternate_forms:          Skriver om matematisk uttryck
    - factorize_expression:         Faktoriserar ett matematiskt uttryck
"""

import numpy as np
import sympy as sp
from sympy import sin, cos, tan, cot, sec, csc, exp, log, sqrt
from sympy import Function, parse_expr
from sympy import print_latex

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.table import Table

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


def series_plot(expression, n = 20, epsilon=None, series_start=1, manuell_linje=None):
    """Denna funktion visualiserar en matematisk serie med ett givet uttryck, ett givet antal termer och ett givet epsilon.

    Argument:
      expression: Det matematiska uttrycket som ska visualiseras. Använd math.uttryck() i uttryck-argumentet när du använder exp, factorial eller liknande. Till exempel: 1/math.factorial(k)
      n: Antalet termer som ska visualiseras.
      epsilon: Värdet av epsilon. Om epsilon inte anges kommer det att ignoreras.
      series_start: Från vilket k-värde serien ska starta.

    Returnerar:
      En plot av serien.
    """
    f = sp.sympify(expression)
    
    x = range(series_start, n + 1)
    y = [eval(expression.replace("k", str(int(k)))) for k in x]
    cumulative_sum = np.cumsum(y)

    # Plot the terms of the series
    plt.figure(figsize=(10, 5))
    plt.figtext(0.5, 1, "Matematisk serie", va="bottom", ha="center", fontsize=20)
    plt.subplot(121)
    plt.plot(x, y, "ro")
    plt.axhline(y=epsilon, color="blue") if epsilon else None
    plt.xlabel("$k$")
    plt.ylabel("$a_k$")
    plt.title("Värdet på uttrycket")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the sum of the series
    plt.subplot(122)
    plt.plot(x, cumulative_sum, "ro")
    plt.axhline(y=manuell_linje, color="blue") if manuell_linje else None
    plt.xlabel("$k$")
    plt.ylabel("$S_n$")
    plt.title("Summa av serien")
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Add the expression to the middle of the plots at the bottom center
    plt.figtext(0.5, 0.02, f'${sp.latex(f)}$', va="top", ha="center", fontsize=14)

    plt.tight_layout()
    plt.show()



def series_comparison_plot(sequence1, sequence2, n_values = 20, dpi = 150):
    """
    Plottar värdena på två talföljder och deras partiella summor.

    Argument:
        sequence1 och sequence2: De två följderna som ska ritas. Dessa kan vara några giltiga Python-uttryck genom att "matematiskt uttryck".
        n_values: Antalet värden av följderna som ska ritas.
        dpi: Upplösningen av grafen.

    Returnerar:
        Ingen.
    """
    
    n = sp.Symbol('n')
    
    # Convert the sequences to SymPy expressions for LaTeX-like formatting
    sequence1_expr = sp.sympify(sequence1)
    sequence2_expr = sp.sympify(sequence2)
    sequence1_latex = sp.latex(sequence1_expr)
    sequence2_latex = sp.latex(sequence2_expr)


    # Define the sequence functions as Python lambda functions
    seq_func1 = lambda n: eval(sequence1)
    seq_func2 = lambda n: eval(sequence2)

    # Generate n values of the sequences
    sequence_values1 = [seq_func1(n_val) for n_val in range(1, n_values + 1)]
    sequence_values2 = [seq_func2(n_val) for n_val in range(1, n_values + 1)]

    # Calculate partial sums for both sequences
    partial_sums1 = np.cumsum(sequence_values1)
    partial_sums2 = np.cumsum(sequence_values2)

    # Calculate the limits of each sequence
    limit_sequence1 = sp.limit(sequence1_expr, n, sp.oo).evalf()
    limit_sequence2 = sp.limit(sequence2_expr, n, sp.oo).evalf()

    # limit_partial_sums1 = sp.limit(sp.Add(*[sp.Rational(val) for val in partial_sums1]), n, sp.oo).evalf()
    # limit_partial_sums2 = sp.limit(sp.Add(*[sp.Rational(val) for val in partial_sums2]), n, sp.oo).evalf()

     # Create a subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi = dpi)

    # Plot the sequence values in the first subplot
    ax1.plot(range(1, n_values + 1), sequence_values1, marker='o', linestyle='-', color='blue', label=fr'Sequence 1: ${sequence1_latex}$')
    ax1.plot(range(1, n_values + 1), sequence_values2, marker='o', linestyle='-', color='red', label=fr'Sequence 2: ${sequence2_latex}$')
    ax1.axhline(y=limit_sequence1, color='blue', linestyle='dashed', label=fr'Limit of Sequence 1: {limit_sequence1}')
    ax1.axhline(y=limit_sequence2, color='red', linestyle='dashed', label=fr'Limit of Sequence 2: {limit_sequence2}')
    ax1.set_xlabel('Antal termer (n)')
    ax1.set_ylabel('Värder på term')
    ax1.set_title('Talföljd')
    ax1.grid(True)

    # Plot the partial sums in the second subplot
    ax2.plot(range(1, n_values + 1), partial_sums1, marker='o', linestyle='-', color='blue', label='Partial Sums 1')
    ax2.plot(range(1, n_values + 1), partial_sums2, marker='o', linestyle='-', color='red', label='Partial Sums 2')
    ax2.set_xlabel('Antal termer (n)')
    ax2.set_ylabel('Partiell summa')
    ax2.set_title('Partiella summan')
    ax2.grid(True)

    # Combine the legend handles and labels from both subplots
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 # + handles2
    all_labels = labels1 # + labels2

    # Add a single legend for the entire figure
    fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2, fancybox=True, shadow=True)

    
    # Add a centered main title for the entire figure
    plt.suptitle("Jämförelse av två serier", fontsize=16)
    
    plt.tight_layout()
    plt.show()


def series_find_n(expression, epsilon, n = 0):
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

def function_Inverse(expression):
    """
        Hittar invers av en funktion.
        
        Argument:
            expression: Funktionen som ska inverteras. Detta kan vara ett giltigt Python-uttryck.
        
        Returnerar:
            Invers av funktionen.
    """
    
    x, y = sp.symbols('x y')
    func = sp.Eq(y, sp.sympify(expression))

    # Solve the function in x
    f_inv = sp.solve(func, x)

    # Convert the solution to be in terms of x
    f_invx = f_inv[0].subs(y, x)
    return f_invx


def function_plot(expression,expression_2=None, hlines = None, vlines = None, inverse=False, x_range=(-10, 10), y_range= (-10, 10), num_points=1000, Grid=True, dpi=None):
    
    """
    Ritar en funktion och dess inverse, om det är möjligt.
    
    Argument:
        expression: Den funktion som ska ritas. Detta kan vara ett giltigt Python-uttryck.
        expression_2: En sekundär funktion som ska ritas. Detta kan vara ett giltigt Python-uttryck.
        inverse: En boolesk variabel som anger om den inverse funktionen ska ritas.
        x_range: Ett intervall som anger de x-värden som ska användas för att rita funktionen.
        num_points: Antalet x-värden som ska användas för att rita funktionen.
        Grid: En boolesk variabel som anger om ett rutnät ska ritas.
        dpi: Upplösningen av figuren.
    
    Returns:
        None.
    """

    def Inverse_calculated(expression):
        x, y = sp.symbols('x y')
        func = sp.Eq(y, sp.sympify(expression))

        # Solve the function in x
        f_inv = sp.solve(func, x)

        # Convert the solution to be in terms of x
        f_invx = f_inv[0].subs(y, x)
        return f_invx
    
    
    if dpi is None: 
        plt.figure(dpi=150)
    else:
        plt.figure(dpi=dpi)

    fig = plt.gcf()  # Get the current figure

    x = np.linspace(x_range[0], x_range[1], num_points)

    # Convert the expression to a sympy expression
    x_sym = sp.Symbol('x')
    y_sym = sp.sympify(expression)

    func = sp.lambdify(x_sym, y_sym, 'numpy')
    y = func(x)

    ymin, ymax = np.min(y), np.max(y)

    plt.plot(x, y, label=f'Funktion: ${sp.latex(y_sym)}$')
    plt.axvline(0, c="black", linewidth = 1)
    plt.axhline(0, c="black", linewidth = 1)
    plt.xlabel(r'$x$')  # Use raw string with LaTeX formatting
    plt.ylabel(r'$y$')
    
    if expression_2 is not None:
        y_sym_2 = sp.sympify(expression_2)
        func_2 = sp.lambdify(x_sym, y_sym_2, 'numpy')
        y_2 = func_2(x)
        ymin = min(ymin, np.min(y_2))
        ymax = max(ymax, np.max(y_2))
        plt.plot(x, y_2, label=f'Funktion 2: ${sp.latex(y_sym_2)}$')

    if inverse:
        try:
            inverse_expr = Inverse_calculated(expression)
            inverse_func = sp.lambdify(x_sym, inverse_expr, 'numpy')
            y_inverse = inverse_func(x)
            
            # Plot the inverse function
            plt.plot(x, y_inverse, label=f'Inversa Funktion: ${sp.latex(inverse_expr)}$')
        except Exception as e:
            print("Error: Kunde ej hitta inversa funktionen.")
            print(e)

    if y_range is not None:
        plt.ylim(y_range)

    if hlines is not None:
        if isinstance(hlines, (int, float)):
            plt.axhline(y=hlines, color="gray", linestyle="--")
        elif isinstance(hlines, (list, tuple)):
            for hline in hlines:
                if isinstance(hline, (int, float)):
                    plt.axhline(y=hline, color="gray", linestyle="--")

    if vlines is not None:
        if isinstance(vlines, (int, float)):
            plt.axvline(x=vlines, color="gray", linestyle="--")
        elif isinstance(vlines, (list, tuple)):
            for vline in vlines:
                if isinstance(vline, (int, float)):
                    plt.axvline(x=vline, color="gray", linestyle="--")

    if expression_2 is not None:
        plt.title(f'Graf av Funktionerna: ${sp.latex(y_sym)}$, ${sp.latex(y_sym_2)}$')
    else:
        plt.title(f'Graf av Funktionen: ${sp.latex(y_sym)}$')

    if Grid:
        plt.grid(linewidth=0.3, color="black")
    else:
        plt.grid(False)

    # Place the legend outside the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fancybox=True, shadow=True)

    plt.tight_layout()
    
    if dpi is not None:
        fig.set_dpi(dpi)  # Set the figure's DPI
    
    plt.show()






def function_plot_piecewise(expression_1, expression_2, bound_1, x_range=(-10, 10), num_points=1000, Grid=True):
    
    """
    Ritar en uppdelad funktion.
    
    Argument:
        expression_1: Den första delen av den uppdelade funktionen. Detta kan vara ett giltigt Python-uttryck.
        expression_2: Den andra delen av den uppdelade funktionen. Detta kan vara ett giltigt Python-uttryck.
        bound_1: Det värde av x där funktionen byter gren.
        x_range: Ett intervall som anger de x-värden som ska användas för att rita funktionen.
        num_points: Antalet x-värden som ska användas för att rita funktionen.
        Grid: En boolesk variabel som anger om ett rutnät ska ritas.
    
    Returns:
        None.
    """
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
    plt.axvline(0, c="black", linewidth = 0.5)
    plt.axhline(0, c="black", linewidth = 0.5)    
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


    
        


def function_evaluate(funktion, x):
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



def function_limit_calculate(expression_str, variable, point):
    """
    Beräkar gränsvärdet för en funktion.
    
    Argument:
        expression_str: Den matematiska expressionen för funktionen.
        variable: Variabeln som gränsvärdet beräknas för.
        point: Det värde av variabeln som gränsvärdet beräknas mot.
    
    Returns:
        Det beräknade gränsvärdet.
    """
    
    x = sp.Symbol(variable)
    expression = sp.sympify(expression_str, locals={"sqrt": sqrt})
    limit_value = sp.limit(expression, x, point)
    print("Gränsvärdet av funktionen: ", limit_value)

def function_and_derivative_plot(expression="x**2", x_val=None):
    """
    Ritar en funktion och dess derivata.
    
    Argument:
        expression: Den matematiska expressionen för funktionen.
        x_val: Ett specifikt värde av x där derivatan ska beräknas.
    
    Returns:
        None.
    """
    
    x = sp.Symbol('x')
    expr = sp.sympify(expression)
    derivative_expr = sp.diff(expr, x)
    y_sym_diff = sp.sympify(derivative_expr)
    

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
    axes[0].plot(x_vals, y_vals)
    axes[0].set_title('Funktion: ${}$'.format(sp.latex(expr)))
    axes[0].legend()

    # Plot the derivative on the bottom
    axes[1].plot(x_vals, dy_vals)
    axes[1].set_title('Derivata: ${}$'.format(sp.latex(y_sym_diff)))  # LaTeX format for title    axes[1].legend()

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

# def calculate_limit(expression_str, variable, point):
#     x = sp.Symbol(variable)

#     # Define the functions you want to use in the expression
#     functions = {
#         "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot,
#         "sec": sp.sec, "csc": sp.csc, "exp": sp.exp, "log": sp.log,
#         "sqrt": sp.sqrt
#     }

#     # Replace the function names in the expression string
#     for func_name, func in functions.items():
#         expression_str = expression_str.replace(func_name, f"__{func_name}__")

#     # Parse the modified expression
#     expression = sp.sympify(expression_str)

#     # Replace the function placeholders back with the actual functions
#     for func_name, func in functions.items():
#         expression = expression.subs(f"__{func_name}__", func)

#     # Calculate the limit
#     limit_value = sp.limit(expression, x, point)
#     print("Gränsvärdet av funktionen: ", limit_value)


#### DERIVATA OCH INTEGRALER


## DERIVATA

def derivative_expression(expression):
    x = sp.Symbol('x')
    derivative_func = sp.diff(expression, x)
    critical_points = sp.solve(derivative_func, x, complex=True)
    print('Derivatan av funktionen är:', derivative_func)
    print('Kritiska punkter:', critical_points)


def derivative_symbolic(expression, latex=False, variable='x'):
    x = sp.Symbol(variable)
    expr = sp.sympify(expression)
    derivative_expr = sp.diff(expr, x, ln_notation=True)

    if latex:
        original_latex = sp.latex(expr)
        derivative_latex = sp.latex(derivative_expr, ln_notation=True)
        print(f"$\\frac{{d}}{{dx}} ({original_latex}) = {derivative_latex}$")

    return derivative_expr





def derivative_plot(expression, x_val=None, solve_for_x=False, points_around_x=100):
    """
    Ritar funktionen `expression` och dess derivata. 
    
    Argument:
        expression: Den matematiska expressionen för funktionen.
        x_val: Ett specifikt värde av x där derivatan ska beräknas.
        solve_for_x: Om True, söker funktionen efter kritiska punkter.
        points_around_x: Antalet x-värden som ska användas för att rita funktionen och derivatan.
    
    Returns:
        None.
    """
    
    x = sp.Symbol('x')
    derivative_func = sp.diff(expression, x)

    # Find critical points
    critical_points = []
    if solve_for_x:
        # Find critical points by solving the derivative equation
        critical_points = sp.solve(derivative_func, x, complex=True)

        # Filter out complex roots and convert to floats for later calculations
        critical_points = [float(cp) for cp in critical_points if cp.is_real]

    # Convert the expression to a sympy expression object
    expr_sympy = sp.sympify(expression)

    # Calculate the range of x_vals based on the critical points and x_val
    if x_val is not None and solve_for_x:
        max_x_abs = max(abs(x_val), max(abs(cp) for cp in critical_points))
        x_min = -(max_x_abs + 5)
        x_max = max_x_abs + 5
        x_vals = np.linspace(x_min, x_max, points_around_x)
    else:
        x_vals = np.linspace(-5, 5, 100)

    # Plot the function
    f = lambda x: eval(expression)
    y_vals = f(x_vals)

    plt.plot(x_vals, y_vals, label='Function {}'.format(expression))

    # Calculate derivative and tangent line
    if x_val is not None:
        try:
            value = derivative_func.subs(x, x_val)
        except TypeError:
            print('Error: Unable to evaluate the derivative at the specified value. Please provide a valid value for x.')
        else:
            print('Derivative of the function: ', derivative_func)
            print('Derivative at x = {}: '.format(x_val), value)

            # Plot the tangent line at x = x_val
            tangent_line = value * (x_vals - x_val) + f(x_val)
            plt.plot(x_vals, tangent_line, label='Tangent at x = {}'.format(x_val))

            # Print the y-value when d/dx = 0 and plot horizontal line
            if solve_for_x:
                if not critical_points:
                    print('Inga kritiska punkter (där derivatan är 0) för denna funktion.')
                else:
                    critical_values = [float(sp.N(expr_sympy.subs(x, cp))) for cp in critical_points]
                    for cp, cv in zip(critical_points, critical_values):
                        plt.axhline(y=cv, color='r', linestyle='--', label='Kritiska punkter at x = {}'.format(cp))
                        print('Kritisk punkt (där derivatan är 0) vid x = {}: y = {}'.format(cp, cv))

    plt.title('Derivatan')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.show()


def derivative_function_plot(expression="x**2", x_val=None, solve_for_x=False, x_range=(-5, 5), dpi=200):
    """
    Ritar en funktion och dess derivata bredvid varandra.

    Argument:
        expression: Den matematiska expressionen för funktionen.
        x_range: Ett intervall som anger start- och slutvärde för x-axeln i alla plotar.
        dpi: DPI (dots per inch) för figuren.

    Returns:
        None.
    """

    x = sp.Symbol('x')
    expr = sp.sympify(expression)
    derivative_expr = sp.diff(expr, x)
    y_sym_diff = sp.sympify(derivative_expr)

    # Convert the SymPy expressions to Python functions using lambdify
    func = sp.lambdify(x, expr, 'numpy')
    derivative_func = sp.lambdify(x, derivative_expr, 'numpy')

    # Generate x values for the plot
    x_vals = np.linspace(x_range[0], x_range[1], 500)

    # Evaluate the functions at the x values
    y_vals = func(x_vals)
    dy_vals = derivative_func(x_vals)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

    # Plot the original function on the left
    axes[0].plot(x_vals, y_vals, label='Funktion')
    axes[0].axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0
    axes[0].axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    axes[0].set_xlim(x_range)  # Set x-axis limits
    axes[0].set_title('Funktion: ${}$'.format(sp.latex(expr)))
    axes[0].legend()

    # Plot the derivative on the right
    axes[1].plot(x_vals, dy_vals, label='Derivata')
    axes[1].axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0
    axes[1].axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    axes[1].set_xlim(x_range)  # Set x-axis limits
    axes[1].set_title('Derivata: ${}$'.format(sp.latex(y_sym_diff)))
    axes[1].legend()

    plt.tight_layout()
    plt.show()






# def mean_value_theorem(expression, x_range=(-10, 10), num_points=1000):
#     x = np.linspace(x_range[0], x_range[1], num_points)

#     # Convert the expression to a sympy expression
#     x_sym = sp.Symbol('x')
#     function = sp.sympify(expression)

#     y = sp.lambdify(x_sym, function, 'numpy')(x)

#     a, b = x_range
#     mean_value = (function.subs(x_sym, b) - function.subs(x_sym, a)) / (b - a)

#     # Find points where the derivative is equal to the mean value
#     derivative = sp.diff(function, x_sym)
#     derivative_eq = sp.Eq(derivative, mean_value)
#     derivative_points = sp.solve(derivative_eq, x_sym)

#     if not isinstance(derivative_points, list):
#         derivative_points = [derivative_points]

#     derivative_points = [point.evalf() for point in derivative_points]

#     # Calculate the tangent lines at the found points
#     tangent_lines = [lambda x, point=point: function.subs(x_sym, point) + mean_value * (x - point) for point in derivative_points]

#     plt.plot(x, y, label='Function')
#     for tangent_line in tangent_lines:
#         plt.plot(x, tangent_line(x), linestyle='dashed')

#     plt.plot([a, b], [function.subs(x_sym, a), function.subs(x_sym, b)], label='Secant Line', linestyle='dotted')

#     plt.scatter([a, b], [function.subs(x_sym, a), function.subs(x_sym, b)], color='red', label='End Points')
#     plt.scatter(derivative_points, [function.subs(x_sym, point) for point in derivative_points], color='green', label='Point(s)')

#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Mean Value Theorem')
#     plt.grid(linewidth=0.3)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     # Add the text under the legend box with mean value rounded to two decimal places
#     rounded_mean_value = round(float(mean_value.evalf()), 2)
#     rounded_x_values = [round(float(point.evalf()), 2) for point in derivative_points]
#     x_values_str = ', '.join([str(point) for point in rounded_x_values])
#     plt.text(1.07, 0.45, f'Function: ${sp.latex(function)}$ \nAverage $\\frac{{d}}{{dx}}$: {rounded_mean_value} \nCount Point(s): {len(derivative_points)} \nx\'s: ({x_values_str})', transform=plt.gca().transAxes)

#     # Update the x-axis limits to the specified x_range
#     plt.xlim(x_range[0] *1.05, x_range[1] * 1.05)

#     plt.show()




def mean_value_theorem(expression, x_range=(-10, 10), num_points=1000, padding=0.5, dpi = 200):
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Convert the expression to a sympy expression
    x_sym = sp.Symbol('x')
    function = sp.sympify(expression)

    y = sp.lambdify(x_sym, function, 'numpy')(x)

    a, b = x_range
    mean_value = (function.subs(x_sym, b) - function.subs(x_sym, a)) / (b - a)

    # Find points where the derivative is equal to the mean value
    derivative = sp.diff(function, x_sym)
    derivative_eq = sp.Eq(derivative, mean_value)
    derivative_points = sp.solve(derivative_eq, x_sym)

    if not isinstance(derivative_points, list):
        derivative_points = [derivative_points]

    derivative_points = [point.evalf() for point in derivative_points]

    # Calculate the tangent lines at the found points
    tangent_lines = [lambda x, point=point: function.subs(x_sym, point) + mean_value * (x - point) for point in derivative_points]
    
    plt.figure(figsize = (12,8), dpi=dpi)
    plt.plot(x, y, label='Function')
    for tangent_line in tangent_lines:
        plt.plot(x, tangent_line(x), linestyle='dashed')
        tangent_x = derivative_points[0]
        tangent_y = tangent_line(derivative_points[0])
        plt.plot([tangent_x, tangent_x], [0, tangent_y], color='orange', linestyle='--', linewidth = 1.5)

    plt.axvline(0, c="black", linewidth=0.75)
    plt.axhline(0, c="black", linewidth=0.75)

    plt.plot([a, b], [function.subs(x_sym, a), function.subs(x_sym, b)], label='Secant Line', linestyle='dotted')

    plt.scatter([a, b], [function.subs(x_sym, a), function.subs(x_sym, b)], color='red', label='End Points')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mean Value Theorem')
    plt.grid(linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    rounded_mean_value = round(float(mean_value.evalf()), 2)
    rounded_x_values = [round(float(point.evalf()), 2) for point in derivative_points]
    x_values_str = ', '.join([str(point) for point in rounded_x_values])
    
    bbox_props = dict(boxstyle='square', edgecolor='grey', facecolor='white', alpha=0.5)

    plt.text(1.02, 0.6, f'Function: ${sp.latex(function)}$ \nAverage $\\frac{{d}}{{dx}}$: {rounded_mean_value} \nCount Point(s): {len(derivative_points)} \n \n c\'s: ({x_values_str})', transform=plt.gca().transAxes,bbox = bbox_props)

    x_min = x_range[0] - padding
    x_max = x_range[1] + padding
    plt.xlim(x_min, x_max)

    plt.show()

def intermediate_value_theorem(expression, x_range=(-10, 10), num_points=1000, padding=0.5, dpi=200):
    x = np.linspace(x_range[0], x_range[1], num_points)

    # Convert the expression to a sympy expression
    x_sym = sp.Symbol('x')
    function = sp.sympify(expression)

    y = sp.lambdify(x_sym, function, 'numpy')(x)

    a, b = x_range

    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.plot(x, y, label='Function')

    plt.axvline(0, c="black", linewidth=0.75)
    plt.axhline(0, c="black", linewidth=0.75)

    plt.scatter([a, b], [function.subs(x_sym, a), function.subs(x_sym, b)], color='red', label='End Points')

    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('Intermediate Value Theorem')
    plt.grid(linewidth=0.4)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    bbox_props = dict(boxstyle='square', edgecolor='grey', facecolor='white', alpha=0.5)

    plt.text(1.02, 0.6, f'Function: ${sp.latex(function)}$ \nInterval: [{a}, {b}]', transform=plt.gca().transAxes, bbox=bbox_props)

    x_min = x_range[0] - padding
    x_max = x_range[1] + padding
    plt.xlim(x_min, x_max)

    # Check if the Intermediate Value Theorem holds
    theorem_holds = False
    sign_at_a = np.sign(float(function.subs(x_sym, a)))
    for x_val in np.linspace(a, b, num_points):
        sign_at_x = np.sign(float(function.subs(x_sym, x_val)))
        if sign_at_x != sign_at_a:
            theorem_holds = True
            break
    
    if theorem_holds:
        plt.text(0.5, -0.15, 'Intermediate Value Theorem håller \n \n Antag att $f$ är kontinuerlig och ändlig på det stängda intervallet $[a,b]$. \n Om $f(a)$ och $f(b)$ har motsatta tecken ($\pm$), existerar det (åtminstone) en punkt $c \in (a,b)$ s.a. $f(c) = 0$.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='green')
    else:
        plt.text(0.5, -0.15, 'Intermediate Value Theorem håller EJ \n \n Antag att $f$ är kontinuerlig och ändlig på det stängda intervallet $[a,b]$. \n Om $f(a)$ och $f(b)$ har motsatta tecken ($\pm$), existerar det (åtminstone) en punkt $c \in (a,b)$ s.a. $f(c) = 0$.', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes, color='red')

    plt.show()
    

def teckentabell(expression, lower_bound, upper_bound, step=0.5):
    x = sp.Symbol('x')
    f = sp.sympify(expression)
    f_prime = sp.diff(f, x)
    f_bis = sp.diff(f_prime, x)
    
    x_values = []
    f_prime_values = []
    f_bis_values = []
    f_direction = []
    f_point = []
    
    current_x = lower_bound
    while current_x <= upper_bound:
        x_values.append(current_x)
        
        f_prime_value = f_prime.subs(x, current_x)
        f_prime_values.append(f_prime_value)
        
        f_bis_value = f_bis.subs(x, current_x)
        f_bis_values.append(f_bis_value)
        
        if f_prime_value > 0:
            f_direction.append('+')
        elif f_prime_value < 0:
            f_direction.append('-')
        else:
            f_direction.append('0')
        
        if f_prime_value == 0 and f_bis_value > 0:
            f_point.append('Min')
        elif f_prime_value  == 0 and f_bis_value < 0:
            f_point.append('Max')
        else:
            f_point.append('')
        
        current_x += step
    
    fig, ax = plt.subplots(figsize=(14, 3))  # Adjust figsize as needed
    
    ax.plot([lower_bound, upper_bound], [0, 0], color='black', linestyle='-', linewidth=2)
    
    # Plot only integer points on the real line
    integer_x_values = [x_val for x_val in x_values if abs(int(x_val) - x_val) < 1e-10]
    ax.scatter(integer_x_values, np.zeros(len(integer_x_values)), color='black', marker='|', label='Integer Points')
    ax.scatter(lower_bound, 0, color='green', marker='o', label='Lower Bound')
    ax.scatter(upper_bound, 0, color='blue', marker='o', label='Upper Bound')
    
    # Add text annotations for integer values
    for x_val in integer_x_values:
        ax.text(x_val, 0.02, f'{int(x_val)}', fontsize=10, ha='center')
    
    # Remove axis labels and ticks
    ax.axis('off')
    
    rounded_f_prime_values = [f'{value:.2f}' for value in f_prime_values]
    rounded_f_bis_values = [f'{value:.2f}' for value in f_bis_values]
    
    # Create a transposed table in LaTeX format
    table_data = [[f"${rounded_value}$" for rounded_value in rounded_f_prime_values]]
    table_data.append([f"${rounded_value}$" for rounded_value in rounded_f_bis_values])
    table_data.append([f"${direction}$" for direction in f_direction])
    table_data.append([f"{point}" for point in f_point])
    
    # Add a textbox to the left of the table
    textbox_content = "$f'(x)$ \n \n $f''(x)$ \n \n $f(x)$ \n \n Min/Max"
    ax.text(-0.05, -0.65, textbox_content, fontsize=19, ha='center', va='center', rotation=0, transform=ax.transAxes)
     
    table = ax.table(cellText=table_data, cellLoc='center', loc='bottom', edges='open')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 4.5)
    
    plt.suptitle(f"Teckentabell av ${sp.latex(f)}, \qquad x \in [{sp.latex(lower_bound)}, {sp.latex(upper_bound)}]$", fontsize=24)  # Adjust title size
    plt.subplots_adjust(top=0.8)  # Adjust top spacing
    
    # Change font size of x-axis tick labels
    ax.tick_params(axis='x', labelsize=18)
    
    plt.show()



def second_derivative_test(expression, x_range=(-5, 5), y_range=None, num_points=200):
    x = sp.symbols('x')
    
    funktion = sp.sympify(expression)
    first_derivative_sym = sp.diff(expression)
    
    # Parse the mathematical expression
    expr = sp.parse_expr(expression)
    
    # Calculate the original expression, first derivative
    func = lambda x_val: expr.subs(x, x_val)
    first_derivative = lambda x_val: sp.diff(expr, x).subs(x, x_val)
    
    # Find critical points where first derivative is zero
    critical_points = sp.solve(first_derivative_sym, x)
    
    # Calculate the second derivative expression
    second_derivative_sym = sp.diff(first_derivative_sym)
    second_derivative = lambda x_val: sp.diff(expr, x, 2).subs(x, x_val)
    
    # Generate x values
    x_values = np.linspace(x_range[0], x_range[1], num_points)
    
    # Calculate corresponding y values for each function
    y_func = np.array([func(x_val) for x_val in x_values])
    y_first_derivative = np.array([first_derivative(x_val) for x_val in x_values])
    y_second_derivative = np.array([second_derivative(x_val) for x_val in x_values])
    
    # Plot the functions
    plt.figure(figsize=(12, 6))
    func_line, = plt.plot(x_values, y_func, label=f'Funktion: ${sp.latex(funktion)}$')
    first_derivative_line, = plt.plot(x_values, y_first_derivative, label=f'Förstaderivatan ${sp.latex(first_derivative_sym)}$')
    second_derivative_line, = plt.plot(x_values, y_second_derivative, label=f'Andraderivatan: ${sp.latex(second_derivative_sym)}$')

    if y_range is not None:
        plt.ylim(y_range)
    
    # Plot the critical points on the graph
    if len(critical_points) > 0:
        for c in critical_points:
            c_second_derivative_val = func(c)
            plt.scatter(c, c_second_derivative_val, linewidth=4, color='green')
    
    plt.axvline(color="black", linewidth=0.8)
    plt.axhline(color="black", linewidth=0.8)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.suptitle('Second derivative test (test för extrempunkter)', size=16)
    
    # Create a separate legend box
    fig = plt.gcf()
    lines = [func_line, first_derivative_line, second_derivative_line]
    labels = [f'Funktion: ${sp.latex(funktion)}$', f'Förstaderivatan ${sp.latex(first_derivative_sym)}$', f'Andraderivatan: ${sp.latex(second_derivative_sym)}$']
    fig.legend(lines, labels, loc='upper left', bbox_to_anchor=(0.25, -0.02))
    
    # Create another legend for critical points
    if len(critical_points) > 0:
        handles = []
        labels = []
        for c in critical_points:
            c_second_derivative_val = second_derivative(c)
            handles.append(plt.Line2D([0], [0], marker='o', color='green', markersize=8, label=f'c = {c}, Andraderivatan = {c_second_derivative_val:.2f}'))
            labels.append(f'c = {c}, Andraderivatan = {c_second_derivative_val:.2f}')
        fig.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(0.65, -0.02))
    
    plt.grid()
    plt.tight_layout()
    plt.show()
    




def differential_equation_plot(differential_eq, initial_condition=None, x_range=(-5, 5), y_range=(-5, 5), mesh_width=0.5, num_points=1000, dpi = 200):
    """
    Ritar en graf som illustrerar en differentialekvation.
    
    Argument:
        differential_eq: Den matematiska expressionen för differentialekvationen.
        initial_condition: Det initiala värdet för y. Om inget initialvärde anges, används 0.
        x_range: Ett intervall som anger start- och slutvärde för x.
        y_range: Ett intervall som anger start- och slutvärde för y.
        mesh_width: Ett värde som används för att justera skalningen av graferna.
        num_points: Antalet x-värden som ska användas för att rita grafen.
    
    Returns:
        None.
    """
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    if initial_condition is None:
        y = [0]  # Use an arbitrary initial condition of 0
    else:
        y = [initial_condition]

    # Define the function for the direction field
    dydx = lambda x, y: eval(differential_eq)

    # Plot the direction field and solution curve using Euler's method
    plt.figure(figsize=(12,9), dpi = dpi)
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.axvline(0, c="black", linewidth=1)
    plt.axhline(0, c="black", linewidth=1)

    dir_field_x_template = np.linspace(-mesh_width / 3, mesh_width / 3, 100)

    for x_val in np.arange(x_range[0], x_range[1], mesh_width):
        for y_val in np.arange(y_range[0], y_range[1], mesh_width):
            if x_val != 0:  # Avoid division by zero at x = 0
                curr_slope = dydx(x_val, y_val)
                curr_intercept = y_val - curr_slope * x_val
                dir_field_xs = dir_field_x_template + x_val
                dir_field_ys = [curr_slope * dfx + curr_intercept for dfx in dir_field_xs]
                plt.plot(dir_field_xs, dir_field_ys, color="red")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Direction Field")

    h = x[1] - x[0]  # Calculate the step size 'h' after creating the x array
    for i in range(1, num_points):
        y.append(y[-1] + h * dydx(x[i - 1], y[i - 1]))  # Append new y value to the list
    plt.plot(x, y, label='Solution Curve', color='blue', linewidth = 2)

    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), ncol=2, fancybox=True, shadow=True)
    plt.show()





## INTEGRALER


def riemann_sum_plot(expression, lower_bound, upper_bound, N=100, epsilon=None):
    """
        Ritar en graf som illustrerar hur Riemann-summor kan användas för att approximera integraler.
        
        Argument:
            expression: Den matematiska expressionen för funktionen som ska integreras.
            lower_bound: Det nedre integralgränsen.
            upper_bound: Det övre integralgränsen.
            N: Antalet rektanglar som ska användas i Riemann-summan.
            epsilon: En epsilon-toleransgrad. Om epsilon inte är None, kommer funktionen att fortsätta att addera rektanglar till Riemann-summan tills approximationen är inom epsilon-toleransen.
        
        Returns:
            None.
        """
    
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
    """
    Beräknar det antal rektanglar som krävs för att approximera en integral med en given epsilon-tolerans.
    
    Argument:
        expression: Den matematiska expressionen för funktionen som ska integreras.
        lower_bound: Det nedre integralgränsen.
        upper_bound: Det övre integralgränsen.
        epsilon: En epsilon-toleransgrad.
    
    Returns:
        int: Antalet rektanglar som krävs för att approximera integralen med epsilon-tolerans.
    """
    
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



def riemann_vs_integral(expression, lower_bound, upper_bound, N=100, epsilon=None, dpi = 150):
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
    primitive_lower = primitiva_funktionen.subs(x, lower_bound)
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
    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)

    # Plot the function and fill the area under the curve in ax1
    x_values = np.arange(lower_bound, upper_bound, 0.01)
    y_values = [f(x) for x in x_values]
    ax1.fill_between(x_values, y_values, alpha=0.25, color='red')
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
    
    n = N
    prev_sum = 0
    current_sum = (f(lower_bound) + f(upper_bound)) * (upper_bound - lower_bound) / 2

    while epsilon is not None and abs(current_sum - prev_sum) > epsilon:
        n *= 2
        dx = (upper_bound - lower_bound) / n
        x_values = np.linspace(lower_bound, upper_bound, n+1)
        prev_sum = current_sum
        current_sum = np.sum(np.fromiter((f(x_values[i]) * dx for i in range(n)), dtype=float))
    
    print('Den primitiva funktionen: ', primitiva_funktionen)
    print('Den primitiva funktionen utvärderad F(b) - F(a):', primitive_upper, '-', primitive_lower, ' = ', primitive_with_substitution)
    print('\nDet exakta (approximerade) integrerade värdet är ≈', value)
    print('N, Antal indelningar = ', N)
    print('Riemann-summor vid N = ', N, ' ≈', rieman_last)
    if epsilon is not None:
        print("\nDå N (antalet indelningar) = ", n, "  approximerar Riemannsummorna integralen med epsilon < ", epsilon)




def integral_expression(expression, lower_bound=None, upper_bound=None):
    """
    Beräkar den primitiva funktionen av en given matematisk expression och, om övre och nedre integralgränser anges, den definite integralen.
    
    Argument:
        expression: Den matematiska expressionen för funktionen som ska integreras.
        lower_bound: Det nedre integralgränsen.
        upper_bound: Det övre integralgränsen.
    
    Returns:
        None.
    """
    
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



def integral_and_function_plot(expression="x**2", x_vals=None, lower=None, upper=None, dpi=200):
    """
    Ritar en funktion och dess integral.

    Argument:
        expression: Den matematiska expressionen för funktionen.
        x_val: Ett specifikt värde av x där integralen ska beräknas.
        lower: Undre gräns för integreringen.
        upper: Övre gräns för integreringen.

    Returns:
        None.
    """

    x = sp.Symbol('x')
    expr = sp.sympify(expression)
    integral_expr = sp.integrate(expr, x)
    y_sym_int = sp.sympify(integral_expr)

    # Convert the SymPy expressions to Python functions using lambdify
    func = sp.lambdify(x, expr, 'numpy')
    integral_func = sp.lambdify(x, integral_expr, 'numpy')

    # Calculate the primitive function
    primitive_expr = sp.integrate(expr, x)
    y_sym_primitive = sp.sympify(primitive_expr)
    y_sym_primitive_simplified = sp.sympify(primitive_expr).simplify()
    primitive_func = sp.lambdify(x, primitive_expr, 'numpy')

    # Generate x values for the plot
    if lower is not None and upper is not None:
        x_vals = np.linspace(lower, upper, 500)
    else:
        x_vals = np.linspace(-10, 10, 500)

    # Evaluate the functions at the x values
    y_vals = func(x_vals)
    integral_vals = integral_func(x_vals)
    primitive_vals = primitive_func(x_vals)

    # Create the plot with subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'height_ratios': [2, 3]}, dpi=dpi)

    # Plot the original function on the top
    axes[0].plot(x_vals, y_vals, label='Function')
    axes[0].set_title('Funktion: ${}$'.format(sp.latex(expr)))

    # Plot the integral on the bottom
    integral_legend = 'Integral'
    if lower is not None and upper is not None:
        # Fill the area between the curve and y=0 within the bounds with a red color
        axes[1].fill_between(x_vals, primitive_vals, color='red', alpha=0.25, where=(x_vals >= lower) & (x_vals <= upper))
    
        # Highlight the area under the curve between the primitive function and y=0 with a blue color
        axes[1].fill_between(x_vals, primitive_vals, color='blue', alpha=0.25, where=(x_vals >= lower) & (x_vals <= x_vals))  # Use x_vals instead of x_val
        integral_legend += f', Area: {sp.latex(y_sym_primitive_simplified.evalf(subs={x: upper}) - y_sym_primitive_simplified.evalf(subs={x: lower}))}'

    axes[1].plot(x_vals, integral_vals, label=integral_legend)
    axes[1].set_title('Primitiva funktionen: ${}$'.format(sp.latex(y_sym_int)))  # LaTeX format for title

    # Add legend to both subplots
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
    return y_sym_primitive_simplified

    


def taylor_polynomial_symbolic(expression, order=3, point=0):
    x = sp.symbols('x')
    taylor_series = sp.series(expression, x, point, order+1).removeO()
    return taylor_series


def taylor_polynomial_plot(function, x_value=0, degree=3, x_range=(-5, 5), num_points=1000, dpi=200):
    """
    Visualizes the Taylor polynomial of a given function around a specific point.

    Arguments:
        function: The mathematical expression of the function.
        x_value: The point around which the Taylor polynomial is centered.
        degree: The degree of the Taylor polynomial.
        x_range: The x-range for the plot.
        num_points: The number of points for plotting.

    Returns:
        None.
    """

    x = sp.Symbol('x')
    expr = sp.sympify(function)

    # Calculate the Taylor polynomial centered at x = x_value
    taylor_poly = sp.series(expr, x, x_value, degree + 1).removeO()

    # Convert the SymPy expressions to Python functions using lambdify
    func = sp.lambdify(x, expr, 'numpy')
    taylor_poly_func = sp.lambdify(x, taylor_poly, 'numpy')

    # Generate x values for the plot
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    # Evaluate the original function and the Taylor polynomial at the x values
    y_vals = func(x_vals)
    taylor_vals = taylor_poly_func(x_vals)

    # Create the plot
    plt.figure(figsize=(10, 6), dpi=dpi)
    plt.plot(x_vals, y_vals, label='$f(x)$')
    plt.plot(x_vals, taylor_vals, label=f'Taylor Polynom (${sp.latex(taylor_poly)}$)')
    plt.scatter(x_value, func(x_value), color='red', label=f'Centrering vid $x={x_value}$')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Taylor-Polynom av $f(x)={sp.latex(expr)}$ (Ordning {degree})')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True)
    plt.grid()
    plt.show()



def taylor_polynomial_multiple_degrees_plot(function, x_value, degrees = (1,3), x_range=(-5, 5), y_range=(-5, 5), num_points=1000, dpi=200):
    """
    Visualizes the Taylor polynomials of a given function around a specific point with different degrees.

    Arguments:
        function: The mathematical expression of the function.
        x_value: The point around which the Taylor polynomials are centered.
        degrees: A tuple of degrees for the Taylor polynomials.
        x_range: The x-range for the plot.
        y_range: The y-range for the plot.
        num_points: The number of points for plotting.

    Returns:
        None.
    """

    x = sp.Symbol('x')
    expr = sp.sympify(function)

    # Convert the SymPy expression to a Python function using lambdify
    func = sp.lambdify(x, expr, 'numpy')

    # Generate x values for the plot
    x_vals = np.linspace(x_range[0], x_range[1], num_points)

    # Create a single plot with multiple subplots for each degree of Taylor polynomial
    num_degrees = len(degrees)
    fig, axes = plt.subplots(1, num_degrees, figsize=(12, 5), dpi=dpi)

    for i, degree in enumerate(degrees):
        # Calculate the Taylor polynomial centered at x = x_value
        taylor_poly = sp.series(expr, x, x_value, degree + 1).removeO()

        # Convert the SymPy expression of the Taylor polynomial to a Python function using lambdify
        taylor_poly_func = sp.lambdify(x, taylor_poly, 'numpy')

        # Evaluate the original function and the Taylor polynomial at the x values
        y_vals = func(x_vals)
        taylor_vals = taylor_poly_func(x_vals)

        # Plot the original function and the Taylor polynomial
        ax = axes[i]
        ax.plot(x_vals, y_vals, label='Function')
        ax.plot(x_vals, taylor_vals, label=f'Taylor Polynomial (Degree {degree})')
        ax.scatter(x_value, func(x_value), color='red', label='Center Point')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Taylor Polynomial Approximation (Degree {degree})')
        ax.set_ylim(y_range)  # Set y-axis limits
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fancybox=True, shadow=True)
        ax.grid()

    plt.tight_layout()
    plt.show()




def function_derivative_integral_plot(expression="x**2", x_val=None, x_range=(-5, 5)):
    """
    Ritar en funktion, dess derivata och integral.

    Argument:
        expression: Den matematiska expressionen för funktionen.
        x_val: Ett specifikt värde av x där derivatan och integralen ska beräknas.
        x_range: Ett intervall som anger start- och slutvärde för x-axeln i alla plotar.

    Returns:
        None.
    """

    x = sp.Symbol('x')
    expr = sp.sympify(expression)
    derivative_expr = sp.diff(expr, x)
    integral_expr = sp.integrate(expr, x)
    y_sym_diff = sp.sympify(derivative_expr)
    y_sym_int = sp.sympify(integral_expr)

    # Convert the SymPy expressions to Python functions using lambdify
    func = sp.lambdify(x, expr, 'numpy')
    derivative_func = sp.lambdify(x, derivative_expr, 'numpy')
    integral_func = sp.lambdify(x, integral_expr, 'numpy')

    # Generate x values for the plot
    x_vals = np.linspace(x_range[0], x_range[1], 500)

    # Evaluate the functions at the x values
    y_vals = func(x_vals)
    dy_vals = derivative_func(x_vals)
    int_vals = integral_func(x_vals)

    # Create a larger plot on top and two smaller plots side by side below it
    fig = plt.figure(figsize=(12, 8), dpi=200)

    # Plot the original function on top
    ax_func = fig.add_subplot(3, 1, (1, 2))
    ax_func.plot(x_vals, y_vals, label='Funktion')
    ax_func.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    ax_func.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    ax_func.set_xlim(x_range)  # Set x-axis limits
    ax_func.set_title('Funktion: ${}$'.format(sp.latex(expr)))
    ax_func.legend()

    # Plot the derivative on the left in the middle
    ax_derivative = fig.add_subplot(3, 2, 5)
    ax_derivative.plot(x_vals, dy_vals, label='Derivata')
    ax_derivative.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0
    ax_derivative.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    ax_derivative.set_xlim(x_range)  # Set x-axis limits
    ax_derivative.set_title('Derivata: ${}$'.format(sp.latex(y_sym_diff)))
    ax_derivative.legend()

    # Plot the integral on the right in the middle
    ax_integral = fig.add_subplot(3, 2, 6)
    ax_integral.plot(x_vals, int_vals, label='Integral', color='orange')
    ax_integral.axvline(0, color='gray', linewidth=0.8, linestyle='--')  # Vertical line at x = 0
    ax_integral.axhline(0, color='gray', linewidth=0.8, linestyle='--')  # Horizontal line at y = 0
    ax_integral.set_xlim(x_range)  # Set x-axis limits
    ax_integral.set_title('Integral: ${}$'.format(sp.latex(y_sym_int)))
    ax_integral.legend()

    plt.tight_layout()
    plt.show()










# def differential_eq(x, y):
#     # Define the differential equation dy/dx = x^2 - y
#     return x*2 - y

# def direction_field(x_range=(-5, 5), y_range=(-5, 5), step=0.75):
#     x = np.arange(x_range[0], x_range[1] + step, step)
#     y = np.arange(y_range[0], y_range[1] + step, step)

#     X, Y = np.meshgrid(x, y)
#     dY = differential_eq(X, Y)

#     plt.figure(figsize=(8, 6))
#     plt.quiver(X, Y, np.ones_like(dY), dY, angles='xy', scale_units='xy', scale=1, color='b')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Direction Field of the Differential Equation dy/dx = x^2 - y')
#     plt.grid(True)
#     plt.show()

# def euler_method(f, x0, y0, x_range, num_points=1000):
#     x = np.linspace(x_range[0], x_range[1], num_points)
#     h = x[1] - x[0]
#     y = np.zeros(num_points)
#     y[0] = y0

#     for i in range(1, num_points):
#         y[i] = y[i-1] + h * f(x[i-1], y[i-1])

#     return x, y

# def plot_solution(x_range=(-5, 5), y0=0, num_points=1000):
#     plt.figure(figsize=(8, 6))
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.title('Solution to the Differential Equation dy/dx = x^2 - y')

#     x, y = euler_method(differential_eq, x_range[0], y0, x_range, num_points)
#     plt.plot(x, y, label=f'y(0) = {y0}')

#     plt.grid(linewidth=0.3)
#     plt.legend()
#     plt.show()




def get_alternate_forms(expression_str):
    """
    Returnerar en lista med alternativa former av en given funktion.
    
    Argument:
        expression_str: Den matematiska expressionen för funktionen.
    
    Returns:
        En lista med alternativa former av funktionen.
    """
    
    expr = sp.sympify(expression_str)

    alternate_forms = []

    # Alternate forms can be derived using SymPy's simplify function
    simplified_expr = sp.simplify(expr)
    alternate_forms.append(simplified_expr)

    # You can add more transformations to the alternate_forms list as needed
    # For example: alternate_forms.append(simplified_expr.expand())

    return alternate_forms

def factorize_expression(expression_str):
    """
    Factoriserar en given funktion.
    
    Argument:
        expression_str: Den matematiska expressionen för funktionen.
    
    Returns:
        Det faktoriserade uttrycket.
    """
    
    expr = sp.sympify(expression_str)

    # Factorize the expression using Sympy's factor function
    factored_expr = sp.factor(expr)

    return factored_expr

