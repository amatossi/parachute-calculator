import json
import os
import numpy as np
from scipy.interpolate import PchipInterpolator

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "shock_data.json")

def load_data():
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def get_pflanz_interpolator(data, case):
    """
    Returns a scipy PchipInterpolator for the given Pflanz case ('0.5', '1', '2').
    The PchipInterpolator is fit in log-log space based on the raw data.
    """
    knots = data["Pflanz_knots"].get(case)
    if not knots:
        raise ValueError(f"Unknown Pflanz case: {case}")
    
    # Original data extraction performed a base-10 log of both X and Y before PCHIP interpolation
    log_x = np.log10(knots["x"])
    log_y = np.log10(knots["y"])
    
    # Ensure log_x is strictly increasing to prevent PchipInterpolator errors
    sorted_indices = np.argsort(log_x)
    return PchipInterpolator(log_x[sorted_indices], log_y[sorted_indices])

def evaluate_pflanz(A_ballistic, case, data=None):
    """
    Evaluates the Pflanz method for a given ballistic parameter `A_ballistic`.
    Returns the opening force reduction factor X1.
    """
    if data is None:
        data = load_data()
        
    interpolator = get_pflanz_interpolator(data, case)
    
    # Evaluate in log10 space, then return 10^y
    log_x_val = np.log10(A_ballistic)
    
    # The domain is roughly 0.01 to 1000 for A. 
    # If outside, scipy's Pchip interpolator will extrapolate, which is consistent with the old repo,
    # though it's physically questionable outside 0.01 to 1000.
    log_y_val = interpolator(log_x_val)
    return 10**log_y_val

def evaluate_mit(Rm, n_gen_fill, data=None):
    """
    Evaluates the MIT opening shock coefficient `Ck` for a given Mass Ratio `Rm`
    and generalized fill constant `n_gen_fill`.
    """
    if data is None:
        data = load_data()
        
    if n_gen_fill >= 4:
        case = "upper"
    elif 1 <= n_gen_fill < 4:
        case = "lower"
    else:
        raise ValueError(f"n_gen_fill ({n_gen_fill}) < 1 is not represented in the MIT data.")
        
    coeffs = data["MIT_coeffs"].get(case)
    if not coeffs:
        raise ValueError(f"No coefficients found for MIT case: {case}")
        
    # Original data evaluated polynomial on log10 of mass ratio
    polynomial = np.poly1d(coeffs)
    return polynomial(np.log10(Rm))

def generate_pflanz_curve(case, data=None, num_points=200):
    """
    Generate curve data for plotting Pflanz
    Returns log-spaced X values and the corresponding evaluated Y values
    """
    # Base 10 log space from 0.01 to 1000
    x_vals = 10**np.linspace(np.log10(0.01), np.log10(1000), num_points)
    y_vals = np.array([evaluate_pflanz(x, case, data) for x in x_vals])
    return x_vals, y_vals

def generate_mit_curve(case, data=None, num_points=100):
    """
    Generate curve data for plotting MIT.
    case is 'upper' or 'lower'
    Returns log-spaced X values and corresponding evaluated Y values
    """
    if data is None:
        data = load_data()
    # Base 10 log space from 0.0001 to 10
    x_vals = 10**np.linspace(-4, 1, num_points)
    
    coeffs = data["MIT_coeffs"][case]
    polynomial = np.poly1d(coeffs)
    y_vals = polynomial(np.log10(x_vals))
    return x_vals, y_vals
