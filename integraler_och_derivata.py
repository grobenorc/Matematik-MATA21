# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 12:55:47 2023

@author: claes
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sympy as sp
import math


def integral(expression, lower_bound=None, upper_bound=None):
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




# def derivative(expression, x_val=None, solve_for_x=False, points_around_x=100):
#     x = sp.Symbol('x')
#     derivative_func = sp.diff(expression, x)

#     # Find critical points
#     if solve_for_x:
#         critical_points = sp.solve(derivative_func, x)

#     if x_val is not None:
#         try:
#             value = derivative_func.subs(x, x_val)
#         except TypeError:
#             print('Error: Unable to evaluate the derivative at the specified value. Please provide a valid value for x.')
#         else:
#             print('Derivative of the function: ', derivative_func)
#             print('Derivative at x = {}: '.format(x_val), value)

#             if solve_for_x:
#                 if not critical_points:
#                     print('Inga kritiska punkter (där derivatan är 0) för denna funktion.')
#                 else:
#                     # Convert the expression to a sympy expression object
#                     expr_sympy = sp.sympify(expression)
#                     critical_points = [float(cp) for cp in critical_points]
#                     critical_values = [float(sp.N(expr_sympy.subs(x, cp))) for cp in critical_points]

#                     # Calculate the range of x_vals based on the critical points and x_val
#                     max_x_abs = max(abs(x_val), max(abs(cp) for cp in critical_points))
#                     x_min = -(max_x_abs + 5)
#                     x_max = max_x_abs + 5
#                     x_vals = np.linspace(x_min, x_max, points_around_x)
#             else:
#                 x_vals = np.linspace(x_val - 2, x_val + 2, points_around_x)
#     else:
#         x_vals = np.linspace(-5, 5, 100)

#     # Plot the function
#     f = lambda x: eval(expression)
#     y_vals = f(x_vals)

#     plt.plot(x_vals, y_vals, label='Function {}'.format(expression))

#     if x_val is not None and (not solve_for_x):
#         # Plot the tangent line at x = x_val
#         tangent_line = value * (x_vals - x_val) + f(x_val)
#         plt.plot(x_vals, tangent_line, label='Tangent at x = {}'.format(x_val))

#     if x_val is not None and solve_for_x:
#         # Plot the tangent line at x = x_val
#         tangent_line = value * (x_vals - x_val) + f(x_val)
#         plt.plot(x_vals, tangent_line, label='Tangent at x = {}'.format(x_val))

#         # Plot the horizontal lines at the critical points
#         for cp, cv in zip(critical_points, critical_values):
#             plt.axhline(y=cv, color='r', linestyle='--', label='Kritiska punkter at x = {}'.format(cp))
    
#         print('Kritiska punkter (där derivatan är 0):', critical_points)
        
#     plt.title('Derivatan')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid()
#     plt.show()

# # Test with function and specific x value
# derivative("x**2", x_val=3)
# derivative("x**2", solve_for_x=True)
# derivative("x**2", x_val=3, solve_for_x=True)


def derivative_expression(expression):
    x = sp.Symbol('x')
    derivative_func = sp.diff(expression, x)
    critical_points = sp.solve(derivative_func, x, complex=True)
    print('Derivatan av funktionen är:', derivative_func)
    print('Derivatan av funktionen är:', critical_points)



def derivative(expression, x_val=None, solve_for_x=False, points_around_x=100):
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


# def derivative(expression, x_val=None, solve_for_x=False, points_around_x=100):
#     x = sp.Symbol('x')
#     derivative_func = sp.diff(expression, x)

#     # Find critical points
#     critical_points = []
#     if solve_for_x:
#         # Find critical points by solving the derivative equation
#         derivative_func = sp.lambdify(x, derivative_func)
#         critical_points = sp.solve(derivative_func, x, complex=True)

#         # Filter out complex roots and convert to floats for later calculations
#         critical_points = [float(cp) for cp in critical_points if cp.is_real]

#         # Print the critical points
#         if critical_points:
#             for cp in critical_points:
#                 print('Kritisk punkt (där derivatan är 0) vid x = {}'.format(cp))

#         # Plot the critical points
#         for cp in critical_points:
#             plt.axhline(y=float(sp.N(expression.subs(x, cp))), color='r', linestyle='--', label='Kritiska punkter at x = {}'.format(cp))

#     # Plot the function
#     f = lambda x: eval(expression)
#     y_vals = f(np.linspace(-5, 5, points_around_x))

#     plt.plot(np.linspace(-5, 5, points_around_x), y_vals, label='Function {}'.format(expression))
    
#     # print('Derivative of the function: ', derivative_func)

#     # Calculate derivative and tangent line
#     if x_val is not None:
#         try:
#             value = derivative_func.subs(x, x_val)
#         except TypeError:
#             print('Error: Unable to evaluate the derivative at the specified value. Please provide a valid value for x.')
#         else:
#             print('Derivative of the function: ', derivative_func)
#             print('Derivative at x = {}: '.format(x_val), value)

#             # Plot the tangent line at x = x_val
#             tangent_line = value * (np.linspace(-5, 5, points_around_x) - x_val) + f(x_val)
#             plt.plot(np.linspace(-5, 5, points_around_x), tangent_line, label='Tangent at x = {}'.format(x_val))

#     plt.title('Derivatan')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid()
#     plt.show()


def plot_function_and_derivative(expression="x**2", x_val=None):
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

# Example usage:
plot_function_and_derivative(expression="x**2+2*x", x_val=3)








