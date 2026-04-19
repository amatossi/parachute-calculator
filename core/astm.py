import numpy as np

def compute_filling_distance(nfill, D0):
    """
    Computes the filling distance.
    s_f = n * D_p (where n is the fill constant and D_p is nominal diameter D0)
    """
    return nfill * D0

def compute_canopy_area(mass, target_vc, CD0, rho):
    """
    Computes the required canopy area S0 for a target steady-state descent rate.
    S0 = (2 * W) / (rho * C_D0 * v_c^2)
    """
    g = 9.81
    W_T = mass * g
    return (2 * W_T) / (rho * CD0 * target_vc**2)

def compute_sea_level_descent_rate(mass, S0, CD0, rho0=1.225):
    """
    Computes the rate of descent at sea level.
    vc0 = sqrt((2 * W_T) / (S0 * C_Do * rho_0))
    W_T = mass * g
    """
    g = 9.81
    W_T = mass * g
    return np.sqrt((2 * W_T) / (S0 * CD0 * rho0))

def compute_altitude_descent_rate(vc0, rho0, rho):
    """
    Computes descent rate at specific density.
    vc = vc0 * sqrt(rho0/rho)
    """
    return vc0 * np.sqrt(rho0 / rho)

def altitude_to_density(altitude_m):
    """
    Standard International Standard Atmosphere (ISA) model for troposphere (up to 11km).
    Computes density (kg/m^3) given altitude in meters.
    """
    rho_0 = 1.225 # kg/m^3 (Sea level standard density)
    T_0 = 288.15 # K (Sea level standard temperature)
    L = 0.0065 # K/m (Temperature lapse rate)
    g = 9.80665 # m/s^2 (Gravity)
    M = 0.0289644 # kg/mol (Molar mass of Earth's air)
    R = 8.3144598 # J/(mol·K) (Universal gas constant)
    
    # Cap altitude to 11000m for this simple troposphere model
    h = min(max(altitude_m, 0), 11000)
    
    temperature = T_0 - L * h
    density = rho_0 * (temperature / T_0)**((g * M) / (R * L) - 1)
    
    return density
