import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def dynamics(xl, vl, x, v, params):
    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']

    # Leading
    xl_dot = vl
    vl_dot = 0.0

    # Ego
    x_dot = v
    delta_x = xl - x
    delta_v = vl - v
    v_optimal = np.tanh(delta_x - 2) + np.tanh(2)
    v_optimal = np.clip(v_optimal, 0, vmax_desired)
    v_dot = alpha * (v_optimal - v) + beta * delta_v / delta_x ** 2

    return xl_dot, vl_dot, x_dot, v_dot

def rk4(xl0, vl0, x0, v0, params, dt, T):
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

    for _ in range(num_steps):
        # K1
        k1_xl_dot, k1_vl_dot, k1_x_dot, k1_v_dot = dynamics(xl, vl, x, v, params)

        # K2
        k2_xl_dot, k2_vl_dot, k2_x_dot, k2_v_dot = dynamics(
            xl + 0.5 * dt * k1_xl_dot,
            vl + 0.5 * dt * k1_vl_dot,
            x + 0.5 * dt * k1_x_dot,
            v + 0.5 * dt * k1_v_dot,
            params
        )

        # K3
        k3_xl_dot, k3_vl_dot, k3_x_dot, k3_v_dot = dynamics(
            xl + 0.5 * dt * k2_xl_dot,
            vl + 0.5 * dt * k2_vl_dot,
            x + 0.5 * dt * k2_x_dot,
            v + 0.5 * dt * k2_v_dot,
            params
        )

        # K4
        k4_xl_dot, k4_vl_dot, k4_x_dot, k4_v_dot = dynamics(
            xl + dt * k3_xl_dot,
            vl + dt * k3_vl_dot,
            x + dt * k3_x_dot,
            v + dt * k3_v_dot,
            params
        )

        xl += (dt / 6) * (k1_xl_dot + 2 * k2_xl_dot + 2 * k3_xl_dot + k4_xl_dot)
        vl += (dt / 6) * (k1_vl_dot + 2 * k2_vl_dot + 2 * k3_vl_dot + k4_vl_dot)
        x += (dt / 6) * (k1_x_dot + 2 * k2_x_dot + 2 * k3_x_dot + k4_x_dot)
        v += (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)

        rel_x.append(xl - x)
        rel_v.append(vl - v)

    return np.array(rel_x), np.array(rel_v)

if __name__ == "__main__":
    xl0 = 2.0
    vl0 = 1.0
    x0 = 1.0
    v0 = 1.5

    params = {
        'alpha': 0.5,
        'beta': 1.0,
        'vmax_desired': 1.964
    }

    dt = 0.01
    T = 20.0

    # Simulation
    rel_x, rel_v = rk4(xl0, vl0, x0, v0, params, dt, T)



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
    plt.savefig('ovfl_rk4.png')