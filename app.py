import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.interpolation import evaluate_pflanz, evaluate_mit, generate_pflanz_curve, generate_mit_curve
from core.calculations import (compute_nominal_diameter, compute_inflation_time, compute_ballistic_parameter,
                              compute_mass_ratio, compute_drag_integral, compute_generalized_fill_constant,
                              compute_steady_state_force)
from core.astm import compute_canopy_area, compute_filling_distance, compute_sea_level_descent_rate, compute_altitude_descent_rate, altitude_to_density
from core.simulation import run_descent_simulation

# Configure Streamlit page
st.set_page_config(page_title="Parachute Design & Analysis Tool", layout="wide")

st.title("🪂 Parachute Design & Analysis Tool")
st.markdown("Follow the tabs sequentially to size the canopy, evaluate opening shock, and simulate the descent.")

# ---- SIDEBAR: GLOBAL ENVIRONMENT INPUTS ----
st.sidebar.header("Global Parameters")
mass = st.sidebar.number_input("Total Mass (kg)", value=25.0, min_value=0.1, step=1.0)
CD0 = st.sidebar.number_input("Drag Coefficient CD0", value=0.8, min_value=0.01, step=0.1)

st.sidebar.markdown("---")
st.sidebar.subheader("Environmental Settings")
use_altitude = st.sidebar.checkbox("Compute Density from Altitude", value=False)
if use_altitude:
    altitude = st.sidebar.number_input("Altitude (m)", value=0.0, min_value=0.0, step=100.0)
    atm_density = altitude_to_density(altitude)
    st.sidebar.info(f"Computed Air Density: {atm_density:.4f} kg/m³")
else:
    atm_density = st.sidebar.number_input("Air Density (kg/m³)", value=1.225, min_value=0.01, step=0.05, 
                                          help="Default is 1.225 kg/m³ (ASTM Sea Level).")

# ---- TABS CONFIGURATION ----
tab1, tab2, tab3 = st.tabs(["1. Canopy Sizing", "2. Opening Shock", "3. Descent Simulation"])

# ==========================================
# TAB 1: CANOPY SIZING
# ==========================================
with tab1:
    st.subheader("Target Descent Velocity")
    st.markdown("Define the desired steady-state drop speed to calculate the required canopy geometry.")
    
    col_size1, col_size2 = st.columns(2)
    with col_size1:
        target_vc = st.number_input("Target Descent Velocity (m/s)", value=6.0, min_value=0.1, step=0.5)
        # Added new inputs for canopy sizing equations
        rho_0 = st.number_input("Sea-Level Air Density ρ₀ (kg/m³)", value=1.225, min_value=0.01, step=0.05)
    with col_size2:
        n_fill_const = st.number_input("Fill Constant (n)", value=11.7, min_value=0.1, step=0.5)
        D_P = st.number_input("Parachute Diameter at Full Inflation D_P (m)", value=5.0, min_value=0.1, step=0.1)
        
    # Calculate Canopy Geometry
    computed_S0 = compute_canopy_area(mass, target_vc, CD0, atm_density)
    computed_D0 = compute_nominal_diameter(computed_S0)
    
    st.markdown("---")
    col_eq1, col_eq2, col_eq3 = st.columns(3)
    
    with col_eq1:
        st.markdown("**1. Required Canopy Area**")
        st.latex(r"S_0 = \frac{2 \cdot W}{\rho \cdot C_{D0} \cdot v_c^2}")
        st.success(f"**Computed S0:** {computed_S0:.2f} m²")
        st.caption(f"Inputs: W = {mass*9.81:.1f} N, ρ = {atm_density:.4f} kg/m³, CD0 = {CD0}, vc = {target_vc} m/s")

    with col_eq2:
        st.markdown("**2. Nominal Canopy Diameter**")
        st.latex(r"D_0 = \sqrt{\frac{4 \cdot S_0}{\pi}}")
        st.success(f"**Computed D0:** {computed_D0:.2f} m")
        st.caption(f"Inputs: S0 = {computed_S0:.2f} m²")
        
    with col_eq3:
        st.markdown("**3. Impact Kinetic Energy**")
        st.latex(r"E = \frac{1}{2} \cdot m \cdot v_c^2")
        kinetic_energy = 0.5 * mass * target_vc**2
        st.success(f"**Computed Energy:** {kinetic_energy:.2f} J")
        st.caption(f"Inputs: m = {mass} kg, vc = {target_vc} m/s")

    st.markdown("---")
    
    # Added additional canopy sizing calculations
    col_eq4, col_eq5, col_eq6 = st.columns(3)
    
    with col_eq4:
        st.markdown("**4. Sea-Level Descent Rate**")
        st.latex(r"v_{c0} = \sqrt{\frac{2 \cdot W_T}{S_0 \cdot C_{D0} \cdot \rho_0}}")
        v_c0 = compute_sea_level_descent_rate(mass, computed_S0, CD0, rho_0)
        st.success(f"**Computed v_c0:** {v_c0:.2f} m/s")
        st.caption(f"Inputs: W_T = {mass*9.81:.1f} N, S_0 = {computed_S0:.2f} m², C_D0 = {CD0}, ρ_0 = {rho_0} kg/m³")

    with col_eq5:
        st.markdown("**5. Altitude-Corrected Descent Rate**")
        st.latex(r"v_c = v_{c0} \sqrt{\frac{\rho_0}{\rho}}")
        v_c = compute_altitude_descent_rate(v_c0, rho_0, atm_density)
        st.success(f"**Computed v_c:** {v_c:.2f} m/s")
        st.caption(f"Inputs: v_c0 = {v_c0:.2f} m/s, ρ_0 = {rho_0} kg/m³, ρ = {atm_density:.4f} kg/m³")

    with col_eq6:
        st.markdown("**6. Filling Distance**")
        st.latex(r"s_f = n \cdot D_P")
        s_f = compute_filling_distance(n_fill_const, D_P)
        st.success(f"**Computed s_f:** {s_f:.2f} m")
        st.caption(f"Inputs: n = {n_fill_const}, D_P = {D_P} m")

    st.markdown("---")
    st.subheader("Steady-State Descent Sizing Plots")
    col_plt1, col_plt2 = st.columns(2)
    
    with col_plt1:
        # S0 vs Descent Rate plot
        s0_range = np.linspace(max(0.1, computed_S0 * 0.2), computed_S0 * 3, 50)
        # We calculate direct descent rate for the current density
        vc_s0_range = [np.sqrt((2 * mass * 9.81) / (s * CD0 * atm_density)) for s in s0_range]
        
        fig_s0 = go.Figure()
        fig_s0.add_trace(go.Scatter(x=s0_range, y=vc_s0_range, mode='lines', name='Descent Rate'))
        fig_s0.add_trace(go.Scatter(x=[computed_S0], y=[target_vc], mode='markers', name='Design Point', marker=dict(size=12, color='red')))
        fig_s0.update_layout(title="Descent Rate vs Canopy Area (S0)", xaxis_title="Canopy Area S0 (m²)", yaxis_title="Descent Rate (m/s)")
        st.plotly_chart(fig_s0, use_container_width=True)

    with col_plt2:
        # Mass vs Descent Rate
        mass_range = np.linspace(max(0.1, mass * 0.2), mass * 3, 50)
        vc_mass_range = [np.sqrt((2 * m * 9.81) / (computed_S0 * CD0 * atm_density)) for m in mass_range]
        
        fig_mass = go.Figure()
        fig_mass.add_trace(go.Scatter(x=mass_range, y=vc_mass_range, mode='lines', name='Descent Rate'))
        fig_mass.add_trace(go.Scatter(x=[mass], y=[target_vc], mode='markers', name='Design Point', marker=dict(size=12, color='red')))
        fig_mass.update_layout(title="Descent Rate vs Total Mass", xaxis_title="Mass (kg)", yaxis_title="Descent Rate (m/s)")
        st.plotly_chart(fig_mass, use_container_width=True)

# ==========================================
# TAB 2: OPENING SHOCK
# ==========================================
with tab2:
    st.subheader("Opening Shock Evaluation")
    st.markdown("Evaluate Pflanz and MIT (OSCalc) methods using the sized canopy geometry.")
    
    col_os_in1, col_os_in2, col_os_in3 = st.columns(3)
    with col_os_in1:
        override_s0 = st.checkbox("Override Computed S0?", value=False)
        if override_s0:
            S0_eval = st.number_input("Override Canopy Area S0 (m²)", value=float(computed_S0), min_value=0.1, step=0.5)
        else:
            S0_eval = computed_S0
            st.info(f"Using Canopy Area (S0): {S0_eval:.2f} m²")
            
        D0_eval = compute_nominal_diameter(S0_eval)
        
    with col_os_in2:
        Cx_help = ("Empirical scaling factor converting steady drag force into peak opening load.\n\n"
                   "Typical ranges by canopy type:\n"
                   "- Flat circular: 1.0–1.2\n"
                   "- Hemispherical: 1.2–1.5\n"
                   "- Cruciform: 1.3–1.6\n"
                   "- Ribbon: 1.5–1.8\n"
                   "- Parafoil: 1.2–1.5")
        Cx = st.number_input("Opening Shock Coeff Cx", value=1.4, min_value=0.1, step=0.1, help=Cx_help)
        v_ls = st.number_input("Line Stretch Velocity (m/s)", value=24.0, min_value=0.1, step=1.0)
        
    with col_os_in3:
        nfill_help = ("Non-dimensional inflation time parameter. Higher n = slower inflation = lower opening shock.\n\n"
                      "Typical ranges by canopy type:\n"
                      "- Hemispherical: 10–16\n"
                      "- Cruciform: 12–18\n"
                      "- Ribbon / Ring-slot: 6–10\n"
                      "- Conical: 8–12\n"
                      "- Parafoil: 10–20")
        nfill = st.number_input("Fill Constant (n)", value=11.7, min_value=0.1, step=0.5, help=nfill_help)
            
        pflanz_help = ("0.5 = soft / long inflation\n\n"
                       "1.0 = nominal\n\n"
                       "2.0 = aggressive / short inflation")
        pflanz_case = st.selectbox("Pflanz Curve Case (n)", options=["0.5", "1", "2"], index=0, help=pflanz_help)

    # Perform Calculations
    tf = compute_inflation_time(nfill, D0_eval, v_ls)
    A_ballistic = compute_ballistic_parameter(mass, S0_eval, CD0, atm_density, v_ls, tf)
    Rm = compute_mass_ratio(atm_density, S0_eval, CD0, mass)
    drag_integral = compute_drag_integral(Rm)
    n_gen_fill = compute_generalized_fill_constant(v_ls, tf, drag_integral, S0_eval, CD0)
    force_nominal = compute_steady_state_force(atm_density, v_ls, S0_eval, CD0)

    pflanz_X1 = evaluate_pflanz(A_ballistic, pflanz_case)
    pflanz_Ck = pflanz_X1 * Cx
    pflanz_force = pflanz_Ck * force_nominal

    mit_Ck = evaluate_mit(Rm, n_gen_fill)
    mit_force = mit_Ck * force_nominal
    
    st.markdown("---")
    col_inter, col_pflanz, col_mit = st.columns(3)

    with col_inter:
        st.markdown("**Intermediate Values**")
        st.markdown(f"""
        - **Nominal Diameter (D0):** {D0_eval:.2f} m
        - **Inflation Time (tf):** {tf:.2f} s
        - **Steady State Force:** {force_nominal:.2f} N
        - **Ballistic Parameter (A):** {A_ballistic:.4f}
        - **Mass Ratio (Rm):** {Rm:.4f}
        - **Generalized Fill (n_gen):** {n_gen_fill:.4f}
        """)

    with col_pflanz:
        st.markdown("**Pflanz Output**")
        st.metric("Pflanz X1", f"{pflanz_X1:.4f}")
        st.metric("Pflanz Ck", f"{pflanz_Ck:.4f}")
        st.metric("Pflanz Force", f"{pflanz_force:.2f} N")

    with col_mit:
        st.markdown("**MIT / OSCalc Output**")
        st.metric("MIT Ck", f"{mit_Ck:.4f}")
        st.metric("MIT Force", f"{mit_force:.2f} N")
        st.metric("Generalized Fill Parameter", f"{n_gen_fill:.4f}")

    st.markdown("---")
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        x_pflanz_curve, y_pflanz_curve = generate_pflanz_curve(pflanz_case)
        fig_pflanz = go.Figure()
        fig_pflanz.add_trace(go.Scatter(x=x_pflanz_curve, y=y_pflanz_curve, mode='lines', name=f'Pflanz Curve (n={pflanz_case})', line=dict(color='red')))
        fig_pflanz.add_trace(go.Scatter(x=[A_ballistic], y=[pflanz_X1], mode='markers', name='Evaluation Point', marker=dict(color='blue', size=10)))
        fig_pflanz.add_shape(type="line", x0=A_ballistic, y0=0.01, x1=A_ballistic, y1=pflanz_X1, line=dict(color="blue", dash="dash"))
        fig_pflanz.add_shape(type="line", x0=0.01, y0=pflanz_X1, x1=A_ballistic, y1=pflanz_X1, line=dict(color="blue", dash="dash"))
        fig_pflanz.update_layout(title="Pflanz Interpolation", xaxis_title="Ballistic Coefficient A [-]", yaxis_title="Opening Force Reduction Factor - X1", xaxis_type="log", yaxis_type="log", xaxis_range=[-2, 3], yaxis_range=[-2, 0], showlegend=True)
        st.plotly_chart(fig_pflanz, use_container_width=True)

    with col_plot2:
        mit_case = "upper" if n_gen_fill >= 4 else "lower"
        x_mit_curve, y_mit_curve = generate_mit_curve(mit_case)
        fig_mit = go.Figure()
        fig_mit.add_trace(go.Scatter(x=x_mit_curve, y=y_mit_curve, mode='lines', name=f'MIT Curve ({mit_case.capitalize()})', line=dict(color='red')))
        fig_mit.add_trace(go.Scatter(x=[Rm], y=[mit_Ck], mode='markers', name='Evaluation Point', marker=dict(color='blue', size=10)))
        fig_mit.add_shape(type="line", x0=Rm, y0=0, x1=Rm, y1=mit_Ck, line=dict(color="blue", dash="dash"))
        fig_mit.add_shape(type="line", x0=0.0001, y0=mit_Ck, x1=Rm, y1=mit_Ck, line=dict(color="blue", dash="dash"))
        fig_mit.update_layout(title=f"MIT Interpolation (1 <= n_gen <= 4)" if mit_case == "lower" else "MIT Interpolation (n_gen >= 4)", xaxis_title="Mass Ratio - Rm", yaxis_title="Opening Shock Factor - Ck", xaxis_type="log", xaxis_range=[-4, 1], yaxis_range=[0, 2], showlegend=True)
        st.plotly_chart(fig_mit, use_container_width=True)

    st.markdown("---")
    st.subheader("Opening Shock Model Summary")
    st.markdown("""
- **Pflanz X1:** Dimensionless factor obtained from empirical Pflanz curves. It represents how inflation dynamics reduce or scale the peak opening load relative to steady-state drag.
- **Opening Shock Coefficient (Cx):** Empirical coefficient that accounts for canopy type and inflation behavior. It scales the load to reflect real parachute dynamics.
- **Pflanz Ck:** Final dimensionless load factor, computed as: `Ck = X1 × Cx`. This factor is used to estimate peak opening force.
- **Opening Shock Force:** Computed as: `F_open = Ck × F_steady`, where F_steady is the steady-state drag force.
- **MIT (OSCalc) Method:** Alternative semi-empirical method based on generalized inflation behavior. Typically more conservative than the Pflanz method and used for comparison.
- **MIT Inflation Regime:** The generalized fill parameter (n_gen_fill) determines how quickly the parachute inflates.
  - `n_gen_fill < 4` → short (fast) inflation
  - `n_gen_fill ≥ 4` → long (slow) inflation
  This classification affects which empirical curve the MIT method uses and influences the predicted opening load.
    """)

# ==========================================
# TAB 3: DESCENT SIMULATION
# ==========================================
with tab3:
    st.subheader("Transient Vertical Descent Simulation")
    st.markdown("Simulates the vertical drop physics after full inflation, using Euler integration:")
    st.latex(r"\frac{dv}{dt} = g - \frac{1}{2m} \cdot \rho \cdot v^2 \cdot C_D \cdot S")
    
    col_sim_in, col_sim_out = st.columns([1, 2])
    
    with col_sim_in:
        st.markdown("**Simulation Parameters**")
        override_s0_sim = st.checkbox("Override Computed S0? (Sim)", value=False)
        if override_s0_sim:
            S0_sim = st.number_input("Override Canopy Area S0 (m²)", value=float(computed_S0), min_value=0.1, step=0.5, key="s0_sim")
        else:
            S0_sim = computed_S0
            st.info(f"Using Canopy Area (S0): {S0_sim:.2f} m²")
            
        sim_h0 = st.number_input("Initial Height (m)", value=1000.0, min_value=0.0, step=50.0)
        sim_v0 = st.number_input("Initial Velocity (m/s) [Positive Downwards]", value=24.0, step=1.0)
        sim_dt = st.number_input("Time Step dt (s)", value=0.05, min_value=0.001, max_value=1.0, step=0.01)
        sim_time = st.number_input("Max Simulation Time (s)", value=300.0, min_value=1.0, step=10.0)
        
    # Run Simulation
    sim_data = run_descent_simulation(mass=mass, Cd=CD0, S=S0_sim, rho=atm_density, 
                                      h0=sim_h0, v0=sim_v0, dt=sim_dt, total_time=sim_time)
    
    with col_sim_out:
        st.markdown("**Simulation Outputs**")
        out1, out2, out3 = st.columns(3)
        out1.metric("Impact Velocity", f"{sim_data['velocity'][-1]:.2f} m/s")
        out2.metric("Terminal Velocity (Theoretical)", f"{sim_data['terminal_velocity']:.2f} m/s")
        # Target velocity is exactly what was put in Tab 1, or theoretical terminal velocity
        out3.metric("ASTM Steady-State Velocity", f"{sim_data['terminal_velocity']:.2f} m/s")
        
        st.markdown("*Notice how the simulation impact velocity converges towards the theoretical terminal velocity/ASTM descent rate.*")

    st.markdown("---")
    st.subheader("Time History Plot")
    
    plot_mode = st.radio("Plot Display Mode", ["Transient Phase Only", "Full Descent to Ground"], horizontal=True)
    
    plot_time = sim_data["time"]
    plot_height = sim_data["height"]
    plot_velocity = sim_data["velocity"]
    plot_acceleration = sim_data["acceleration"]
    
    if plot_mode == "Transient Phase Only":
        v_term = sim_data["terminal_velocity"]
        pct_error = np.abs(plot_velocity - v_term) / v_term
        steady_indices = np.where(pct_error < 0.05)[0]
        
        if len(steady_indices) > 0:
            cutoff_idx = steady_indices[0]
        else:
            cutoff_idx = np.argmin(pct_error)
            
        buffer_points = int(2.0 / sim_dt)
        cutoff_idx = min(cutoff_idx + buffer_points, len(plot_time))
            
        plot_time = plot_time[:cutoff_idx]
        plot_height = plot_height[:cutoff_idx]
        plot_velocity = plot_velocity[:cutoff_idx]
        plot_acceleration = plot_acceleration[:cutoff_idx]

    # Create Subplots
    fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sim.add_trace(go.Scatter(x=plot_time, y=plot_height, mode='lines', name='Height (m)', line=dict(color='blue')), secondary_y=False)
    fig_sim.add_trace(go.Scatter(x=plot_time, y=plot_velocity, mode='lines', name='Velocity (m/s)', line=dict(color='red')), secondary_y=True)
    fig_sim.add_trace(go.Scatter(x=plot_time, y=plot_acceleration, mode='lines', name='Acceleration (m/s²)', line=dict(color='green')), secondary_y=True)
    
    fig_sim.update_layout(title="Vertical Descent Kinematics", xaxis_title="Time (s)", hovermode="x unified")
    fig_sim.update_yaxes(title_text="Height (m)", secondary_y=False, color="blue")
    fig_sim.update_yaxes(title_text="Velocity (m/s) & Acceleration (m/s²)", secondary_y=True)
    st.plotly_chart(fig_sim, use_container_width=True)
