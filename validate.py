from core.calculations import *
from core.interpolation import evaluate_pflanz, evaluate_mit

# Original Inputs
atm_density = 1.18
mass = 25
v_ls = 24
g = 9.81
CD0 = 0.8
S0 = 12
D0 = 4 * np.sqrt(12 / np.pi)
Cx = 1.4
nfill = 11.7
pflanz_case = '0.5'

tf = compute_inflation_time(nfill, D0, v_ls)
A_ballistic = compute_ballistic_parameter(mass, S0, CD0, atm_density, v_ls, tf)
Rm = compute_mass_ratio(atm_density, S0, CD0, mass)
drag_integral = compute_drag_integral(Rm)
n_gen_fill = compute_generalized_fill_constant(v_ls, tf, drag_integral, S0, CD0)
force_nominal = compute_steady_state_force(atm_density, v_ls, S0, CD0)

pflanz_X1 = evaluate_pflanz(A_ballistic, pflanz_case)
pflanz_Ck = pflanz_X1 * Cx
pflanz_force = pflanz_Ck * force_nominal

mit_Ck = evaluate_mit(Rm, n_gen_fill)
mit_force = mit_Ck * force_nominal

print(f"Steady State Force = {force_nominal:.2f} N")
print(f"Pflanz X1 = {pflanz_X1:.4f}")
print(f"Pflanz Ck = {pflanz_Ck:.4f}")
print(f"Pflanz Force = {pflanz_force:.2f} N")
print(f"MIT Ck = {mit_Ck:.4f}")
print(f"MIT Force = {mit_force:.2f} N")
