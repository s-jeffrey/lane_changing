import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def dynamics(xl, vl, x, v, params):
    alpha = params['alpha']
    vmax_desired = params['vmax_desired']

    # Leading
    xl_dot = vl
    vl_dot = 0.0

    # Relative position and speed
    delta_x = xl - x
    delta_v = vl - v

    # Ego
    x_dot = v
    v_optimal = np.tanh(delta_x - 2) + np.tanh(2)
    # Set bounds on v_optimal
    v_optimal = np.clip(v_optimal, 0, vmax_desired)
    v_dot = alpha * (v_optimal - v)

    return xl_dot, vl_dot, x_dot, v_dot

def euler(xl0, vl0, x0, v0, params, dt, T):
    num_steps = int(T / dt)
    # time_points = np.linspace(0, T, num_steps)

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
        xl_dot, vl_dot, x_dot, v_dot = dynamics(xl, vl, x, v, params)

        xl += xl_dot * dt
        vl += vl_dot * dt
        x += x_dot * dt
        v += v_dot * dt

        rel_x.append(xl - x)
        rel_v.append(vl - v)

    return np.array(rel_x), np.array(rel_v)

if __name__ == "__main__":
    xl0 = 2.5
    vl0 = 1.0
    x0 = 1.0
    v0 = 1.9

    params = {
        'alpha': 0.5,
        'vmax_desired': 1.964
    }

    dt = 0.01
    T = 20.0

    # Simulation
    rel_x, rel_v = euler(xl0, vl0, x0, v0, params, dt, T)



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
    plt.savefig('ovm_euler.png')