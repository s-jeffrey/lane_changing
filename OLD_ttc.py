import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def dynamics(xl, vl, x, v, params, ttc=None):
    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']
    dt = params['dt']

    # Leading dynamics
    xl_dot = vl
    vl_dot = 0.0

    # Ego dynamics
    x_dot = v
    delta_x = xl - x
    delta_v = vl - v
    v_optimal = np.clip(np.tanh(delta_x - 2) + np.tanh(2), 0, vmax_desired)
    # Use OVM to simulate ttc
    if ttc is None:
        v_dot = alpha * (v_optimal - v)
    else:
        # This should never happen
        if ttc == 0:
            print("Warning: ttc is zero")
            v_dot = alpha * (v_optimal - v)
        # Once ttc is simulated, plug it into the dynamic
        else:
            v_dot = alpha * (v_optimal - v) + beta * delta_v / ttc ** 2

    return xl_dot, vl_dot, x_dot, v_dot

def rk4(xl, vl, x, v, params, ttc=None):
    dt = params['dt']

    # K1
    k1_xl_dot, k1_vl_dot, k1_x_dot, k1_v_dot = dynamics(xl, vl, x, v, params, ttc)

    # K2
    k2_xl_dot, k2_vl_dot, k2_x_dot, k2_v_dot = dynamics(
        xl + 0.5 * dt * k1_xl_dot,
        vl + 0.5 * dt * k1_vl_dot,
        x + 0.5 * dt * k1_x_dot,
        v + 0.5 * dt * k1_v_dot,
        params, ttc
    )

    # K3
    k3_xl_dot, k3_vl_dot, k3_x_dot, k3_v_dot = dynamics(
        xl + 0.5 * dt * k2_xl_dot,
        vl + 0.5 * dt * k2_vl_dot,
        x + 0.5 * dt * k2_x_dot,
        v + 0.5 * dt * k2_v_dot,
        params, ttc
    )

    # K4
    k4_xl_dot, k4_vl_dot, k4_x_dot, k4_v_dot = dynamics(
        xl + dt * k3_xl_dot,
        vl + dt * k3_vl_dot,
        x + dt * k3_x_dot,
        v + dt * k3_v_dot,
        params, ttc
    )

    xl += (dt / 6) * (k1_xl_dot + 2 * k2_xl_dot + 2 * k3_xl_dot + k4_xl_dot)
    vl += (dt / 6) * (k1_vl_dot + 2 * k2_vl_dot + 2 * k3_vl_dot + k4_vl_dot)
    x += (dt / 6) * (k1_x_dot + 2 * k2_x_dot + 2 * k3_x_dot + k4_x_dot)
    v += (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)

    return xl, vl, x, v

def simulate(xl0, vl0, x0, v0, params, calculate_ttc=False):
    dt = params['dt']
    T = params['T']
    
    num_steps = int(T / dt)

    # Initial conditions
    xl = xl0
    vl = vl0
    x = x0
    v = v0

    # Create arrays to plot
    rel_x = []
    rel_v = []
    time_plot = []
    ttc_plot = []

    # Enter initial conditions to plot
    rel_x.append(xl - x)
    rel_v.append(vl - v)

    for i in range(num_steps):
        '''
        ttc is None means we need to calculate the ttc; e.g.,
        Use the OVM model and run the simulation to calculate ttc
        It does NOT mean ttc = 0
        '''
        ttc = None

        if not calculate_ttc:
            ttc = simulate(xl, vl, x, v, params, calculate_ttc=True) * dt # Calculates ttc in seconds
            # For plotting ttc vs. time
            ttc_plot.append(ttc)
            time_plot.append(i * dt)

        # If we are on the 'first' loop (where we have already calculated ttc by this point), pass ttc into rk4() and thus dynamics()
        # Otherwise if we are supposed to be calculating ttc (calculate_ttc = True), pass ttc = None into rk4() and dynamics()
        xl, vl, x, v = rk4(xl, vl, x, v, params, ttc if not calculate_ttc else None)

        if calculate_ttc:
            # If collision happens, report number of time steps as ttc
            if xl - x <= 0:
                return i + 1
            if vl - v > 0:
                return 100
        else:
            # For plotting rel_v vs. rel_x
            rel_x.append(xl - x)
            rel_v.append(vl - v)
        
    return np.array(rel_x), np.array(rel_v), np.array(time_plot), np.array(ttc_plot) if not calculate_ttc else exit(1)

if __name__ == "__main__":
    # Set initial conditions here
    xl0 = 1.15
    vl0 = 0.55
    x0 = 1.00
    v0 = 1.35

    params = {
        'alpha': 2.0,
        'beta': 3.0,
        'vmax_desired': 1.964,
        'dt': 0.01,
        'T': 20.0
    }

    # Simulation
    rel_x, rel_v, time_plot, ttc_plot = simulate(xl0, vl0, x0, v0, params, calculate_ttc=False)
    print(rel_x, rel_v)
    print(time_plot, ttc_plot)



    # Plot vl-v vs. xl-x
    plt.figure(figsize=(10, 7))
    plt.plot(rel_x, rel_v, label='trajectory', color='blue')
    plt.scatter(rel_x[0], rel_v[0], color='cyan', s=100, zorder=5, label='start')
    plt.scatter(rel_x[-1], rel_v[-1], color='orange', s=100, zorder=5, label='end')

    plt.title('Relative Velocity vs. Relative Position')
    plt.xlabel('Relative Position ($x_l - x$)')
    plt.ylabel('Relative Velocity ($v_l - v$)')
    plt.grid(True)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.savefig('OLD_ttc.png')

    # Plot ttc vs. time
    plt.figure(figsize=(10, 7))
    plt.plot(time_plot, ttc_plot, label='traj', color='blue')
    plt.title('TTC vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('TTC (s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('OLD_ttc_time.png')