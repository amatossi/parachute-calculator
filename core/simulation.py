import numpy as np

def run_descent_simulation(mass, Cd, S, rho, h0, v0, dt, total_time):
    """
    Runs a transient vertical descent simulation using Euler integration.
    
    dv/dt = g - (0.5 * rho * v^2 * Cd * S) / m
    
    Returns a dictionary of time history arrays and final state.
    """
    g = 9.81
    
    num_steps = int(total_time / dt) + 1
    t_history = np.linspace(0, total_time, num_steps)
    
    v_history = np.zeros(num_steps)
    h_history = np.zeros(num_steps)
    a_history = np.zeros(num_steps)
    
    # Initial conditions
    v_history[0] = v0
    h_history[0] = h0
    
    for i in range(num_steps - 1):
        # Calculate current acceleration
        # Note: Velocity here represents downward speed (positive downwards)
        # Drag acts opposite to velocity.
        # Assuming v is always positive (falling down).
        drag_force = 0.5 * rho * v_history[i]**2 * Cd * S
        
        # We assume positive velocity is downward descent.
        # Gravity pulls down (positive), drag pulls up (negative).
        a = g - (drag_force / mass)
        
        a_history[i] = a
        
        # Euler step
        v_history[i+1] = v_history[i] + a * dt
        h_history[i+1] = h_history[i] - v_history[i] * dt  # height decreases as you descend
        
        # If height drops below zero, ground impact
        if h_history[i+1] < 0:
            h_history[i+1] = 0
            # We can zero velocity and acceleration after impact if desired, 
            # or just let it continue for math purposes. We'll zero it for realism.
            v_history[i+1] = 0
            # For plotting, the remaining array elements should be handled
            # We could truncate the arrays or keep them constant
            t_history = t_history[:i+2]
            v_history = v_history[:i+2]
            h_history = h_history[:i+2]
            a_history = a_history[:i+2]
            a_history[-1] = 0 # No acceleration on ground
            break

    # Calculate terminal velocity theoretically
    terminal_velocity = np.sqrt((2 * mass * g) / (rho * Cd * S))
    
    # Handle the final acceleration point if not impacted
    if len(a_history) == num_steps:
        drag_force = 0.5 * rho * v_history[-1]**2 * Cd * S
        a_history[-1] = g - (drag_force / mass)
        
    return {
        "time": t_history,
        "velocity": v_history,
        "height": h_history,
        "acceleration": a_history,
        "terminal_velocity": terminal_velocity
    }
