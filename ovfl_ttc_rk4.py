import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def dynamics(xl, vl, x, v, params, ttc=None):
    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']
    dt = params['dt']

    # Leading
    xl_dot = vl
    vl_dot = 0.0

    # Ego
    x_dot = v
    delta_x = xl - x
    delta_v = vl - v

    v_optimal = np.clip(np.tanh(delta_x - 2) + np.tanh(2), 0, vmax_desired)

    if ttc is None:
        v_dot = alpha * (v_optimal - v)
    else:
        if ttc == 0:
            print("Warning: ttc is zero")
            v_dot = alpha * (v_optimal - v)
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

    # Initial
    xl = xl0
    vl = vl0
    x = x0
    v = v0

    rel_x = []
    rel_v = []

    rel_x.append(xl - x)
    rel_v.append(vl - v)

    for i in range(num_steps):
        ttc = None

        if not calculate_ttc:
            ttc = simulate(xl, vl, x, v, params, calculate_ttc=True) * dt # Do I use time here or time steps??

        xl, vl, x, v = rk4(xl, vl, x, v, params, ttc if not calculate_ttc else None)

        if calculate_ttc:
            if xl - x <= 0:
                return i + 1
            if vl - v > 0:
                return np.inf
        else:
            rel_x.append(xl - x)
            rel_v.append(vl - v)
        
    return np.array(rel_x), np.array(rel_v) if not calculate_ttc else np.inf

if __name__ == "__main__":
    xl0 = 1.5
    vl0 = 1.0
    x0 = 1.0
    v0 = 2.0

    params = {
        'alpha': 0.5,
        'beta': 0.5,
        'vmax_desired': 1.964,
        'dt': 0.01,
        'T': 20.0
    }

    # Simulation
    ttc_steps = simulate(xl0, vl0, x0, v0, params, calculate_ttc=True)
    ttc_time = ttc_steps * params['dt'] if ttc_steps != np.inf else np.inf
    rel_x, rel_v = simulate(xl0, vl0, x0, v0, params, calculate_ttc=False)



    # Plot
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
    plt.savefig('ovfl_ttc_rk4.png')