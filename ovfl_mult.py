import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

def dynamics(positions, velocities, params, time):
    num_vehicles = len(positions)

    positions_dot = np.zeros(num_vehicles)
    velocities_dot = np.zeros(num_vehicles)

    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']

    # Leader
    positions_dot[0] = velocities[0]
    velocities_dot[0] = np.exp(-0.1 * time) * np.sin(time)

    # Followers
    for i in range(1, num_vehicles):
        x = positions[i]
        v = velocities[i]
        xl = positions[i - 1]
        vl = velocities[i - 1]

        positions_dot[i] = v
        delta_x = xl - x
        delta_v = vl - v
        v_optimal = np.clip(np.tanh(delta_x - 2) + np.tanh(2), 0, vmax_desired)
        velocities_dot[i] = alpha * (v_optimal - v) + beta * delta_v / delta_x ** 2

    return positions_dot, velocities_dot

def rk4(init_positions, init_velocities, params, dt, T):
    num_steps = int(T / dt)
    num_vehicles = len(init_positions)

    # Initial
    positions_curr = np.array(init_positions)
    velocities_curr = np.array(init_velocities)

    velocities_toplot = []
    velocities_toplot.append(velocities_curr)

    time_curr = 0.0

    for _ in range(num_steps):
        # K1
        k1_pos_dot, k1_vel_dot = dynamics(positions_curr, velocities_curr, params, time_curr)

        # K2
        k2_pos_dot, k2_vel_dot = dynamics(
            positions_curr + 0.5 * dt * k1_pos_dot,
            velocities_curr + 0.5 * dt * k1_vel_dot,
            params, time_curr + 0.5 * dt
        )

        # K3
        k3_pos_dot, k3_vel_dot = dynamics(
            positions_curr + 0.5 * dt * k2_pos_dot,
            velocities_curr + 0.5 * dt * k2_vel_dot,
            params, time_curr + 0.5 * dt
        )

        # K4
        k4_pos_dot, k4_vel_dot = dynamics(
            positions_curr + dt * k3_pos_dot,
            velocities_curr + dt * k3_vel_dot,
            params, time_curr + dt
        )

        positions_curr += (dt / 6) * (k1_pos_dot + 2 * k2_pos_dot + 2 * k3_pos_dot + k4_pos_dot)
        velocities_curr += (dt / 6) * (k1_vel_dot + 2 * k2_vel_dot + 2 * k3_vel_dot + k4_vel_dot)
        time_curr += dt

        velocities_toplot.append(velocities_curr.copy())

    return np.array(velocities_toplot)

if __name__ == "__main__":
    # Set total number of vehicles
    num_vehicles = 9

    # Initialize pos and vel arrays
    init_positions = np.zeros(num_vehicles)
    init_velocities = np.zeros(num_vehicles)
    
    # Initial conditions for Leader
    xl0 = 2.5
    vl0 = 1.0
    init_positions[0] = xl0
    init_velocities[0] = vl0

    # Initial conditions for Follower(s)
    x0 = 1.0
    v0 = 1.9

    init_headway = 1.5

    for i in range(1, num_vehicles):
        init_positions[i] = init_positions[i - 1] - init_headway
        init_velocities[i] = v0

    params = {
        'alpha': 2.0,
        'beta': 3.0,
        'vmax_desired': 1.964
    }

    dt = 0.01
    T = 50.0

    # Simulation
    velocities_toplot = rk4(init_positions, init_velocities, params, dt, T)
    time_toplot = np.linspace(0, T, len(velocities_toplot))



    # Plot Velocity vs Time
    plt.figure(figsize=(10, 7))
    
    # for i in range(num_vehicles):
    #     plt.plot(time, velocities_toplot[:, i], label=f'Vehicle {i} Velocity')

    # plt.plot(time_toplot, velocities_toplot[:, 0], label='Leader Velocity')
    for i in range(1, num_vehicles):
        relative_velocity = velocities_toplot[:, i - 1] - velocities_toplot[:, i]
        plt.plot(time_toplot, relative_velocity, label=f'Vehicle {i} relative to {i-1}')
    
    plt.title('Vehicle Velocities w/ Acceleration vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ovfl_mult.png')