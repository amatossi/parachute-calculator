import numpy as np

def compute_nominal_diameter(S0):
    """
    Computes the nominal canopy diameter D0 from the surface area S0.
    D0 = sqrt(4 * S0 / pi)
    """
    return np.sqrt(4 * S0 / np.pi)

def compute_inflation_time(nfill, D0, v_ls):
    """
    Computes the parachute inflation time.
    tf = nfill * D0 / v_ls
    """
    return nfill * D0 / v_ls

def compute_ballistic_parameter(mass, S0, CD0, atm_density, v_ls, tf):
    """
    Computes the Ballistic Parameter (A).
    A_ballistic = 2 * mass / (S0 * CD0 * atm_density * v_ls * tf)
    """
    return 2 * mass / (S0 * CD0 * atm_density * v_ls * tf)

def compute_mass_ratio(atm_density, S0, CD0, mass):
    """
    Computes the Mass Ratio (Rm).
    Rm = atm_density * (S0 * CD0)**(3/2) / mass
    """
    return atm_density * (S0 * CD0)**(1.5) / mass

def compute_drag_integral(Rm):
    """
    Computes the drag integral based on the Mass Ratio (Rm).
    Following OSCalc Manual:
    If Rm > 0.1: 0.5
    If Rm > 0.01: 0.35
    Else: 0.2
    """
    if Rm > 0.1:
        return 0.5
    elif Rm > 0.01:
        return 0.35
    else:
        return 0.2

def compute_generalized_fill_constant(v_ls, tf, drag_integral, S0, CD0):
    """
    Computes the generalized fill constant (n_gen_fill).
    """
    return v_ls * tf * drag_integral / np.sqrt(S0 * CD0)

def compute_steady_state_force(atm_density, v_ls, S0, CD0):
    """
    Computes the nominal steady state force at line stretch.
    force_nominal = 0.5 * atm_density * v_ls^2 * S0 * CD0
    """
    return 0.5 * atm_density * (v_ls**2) * S0 * CD0

def compute_annular_geometry(D0, num_gores, vent_ratio=0.611, shape_ratio=0.392):
    """
    Computes annular parachute geometry dimensions based on nominal diameter D0.
    Default ratios are based on NWC reference annular geometry:
    D_o = 1.000
    D = 1.040
    D_c = 0.940
    L = 1.250
    h = 0.304
    b = 0.200
    
    Adjustable inputs:
    vent_ratio (Dv/Do): default 0.611
    shape_ratio ((a-b)/h): default 0.392
    """
    geom = {
        "D": 1.040 * D0,
        "D_c": 0.940 * D0,
        "L": 1.250 * D0,
        "h": 0.304 * D0,
        "b": 0.200 * D0,
    }
    
    # Adjustable geometries
    geom["D_v"] = vent_ratio * D0
    geom["a"] = geom["b"] + shape_ratio * geom["h"]
    
    # Derived apex height
    geom["h_x"] = (geom["a"] + geom["b"] + geom["h"]) / 2.0 - geom["b"]
    
    # Gore layout
    geom["C_v"] = (np.pi * geom["D_v"]) / num_gores
    geom["C_s"] = (np.pi * geom["D"]) / num_gores
    
    return geom
