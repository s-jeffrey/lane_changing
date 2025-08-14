import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# Dynamics for vehicles 0 and 1
def dynamics(x0, v0, x1, v1, params, time, ttc=None):
    # print(f"time: {time}")
    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']

    # Leading dynamics
    x0_dot = v0
    v0_dot = np.exp(-0.1 * time) * np.sin(time)

    # Ego dynamics
    x1_dot = v1
    delta_x = x0 - x1
    delta_v = v0 - v1
    v_optimal = np.clip(np.tanh(delta_x - 2) + np.tanh(2), 0, vmax_desired)
    if ttc is None:
        v1_dot = alpha * (v_optimal - v1)
    else:
        # This should never happen
        if ttc == 0:
            print("Warning: ttc is zero")
            v1_dot = alpha * (v_optimal - v1)
        # Once ttc is simulated, plug it into the dynamic
        else:
            v1_dot = alpha * (v_optimal - v1) + beta * delta_v / ttc ** 2

    return x0_dot, v0_dot, x1_dot, v1_dot

# Dynamics for vehicles 2 and beyond
def internal_dynamics(xl, vl, x, v, accel_leader, vel_leader, params, ttc=None):
    alpha = params['alpha']
    beta = params['beta']
    vmax_desired = params['vmax_desired']

    # Leading dynamics
    xl_dot = vel_leader
    vl_dot = accel_leader

    # Ego dynamics
    x_dot = v
    delta_x = xl - x
    delta_v = vel_leader - v
    v_optimal = np.clip(np.tanh(delta_x - 2) + np.tanh(2), 0, vmax_desired)
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

# RK4 for vehicles 0, 1
def rk4(xl, vl, x, v, params, time_for_v0_dot, ttc=None):
    dt = params['dt']

    # Keep track of the current follower's v_dot (accel) because it will be used in the calculations by its follower
    accel_follower = np.empty(4)
    vel_follower = np.empty(4)

    # K1
    k1_xl_dot, k1_vl_dot, k1_x_dot, k1_v_dot = dynamics(xl, vl, x, v, params, time_for_v0_dot, ttc)
    accel_follower[0] = k1_v_dot
    vel_follower[0] = k1_x_dot

    # K2
    k2_xl_dot, k2_vl_dot, k2_x_dot, k2_v_dot = dynamics(
        xl + 0.5 * dt * k1_xl_dot,
        vl + 0.5 * dt * k1_vl_dot,
        x + 0.5 * dt * k1_x_dot,
        v + 0.5 * dt * k1_v_dot,
        params, time_for_v0_dot + 0.5 * dt, ttc
    )
    accel_follower[1] = k2_v_dot
    vel_follower[1] = k2_x_dot

    # K3
    k3_xl_dot, k3_vl_dot, k3_x_dot, k3_v_dot = dynamics(
        xl + 0.5 * dt * k2_xl_dot,
        vl + 0.5 * dt * k2_vl_dot,
        x + 0.5 * dt * k2_x_dot,
        v + 0.5 * dt * k2_v_dot,
        params, time_for_v0_dot + 0.5 * dt, ttc
    )
    accel_follower[2] = k3_v_dot
    vel_follower[2] = k3_x_dot

    # K4
    k4_xl_dot, k4_vl_dot, k4_x_dot, k4_v_dot = dynamics(
        xl + dt * k3_xl_dot,
        vl + dt * k3_vl_dot,
        x + dt * k3_x_dot,
        v + dt * k3_v_dot,
        params, time_for_v0_dot + dt, ttc
    )
    accel_follower[3] = k4_v_dot
    vel_follower[3] = k4_x_dot

    xl += (dt / 6) * (k1_xl_dot + 2 * k2_xl_dot + 2 * k3_xl_dot + k4_xl_dot)
    vl += (dt / 6) * (k1_vl_dot + 2 * k2_vl_dot + 2 * k3_vl_dot + k4_vl_dot)
    x += (dt / 6) * (k1_x_dot + 2 * k2_x_dot + 2 * k3_x_dot + k4_x_dot)
    v += (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)

    return xl, vl, x, v, accel_follower, vel_follower

# RK4 for vehicles 2 and beyond
# Want to pass an the acceleration, velocity value for the current time step from some larger accel array
# for the immediate leading vehicle into this method and use those "vl_dot"s and "xl_dots"
# for the calculation of positions, velocities, accelerations of the following car.
# accel_leader, vel_leader are 4 element arrays that contains the accel, vel of the leader at each rk4 step
def internal_rk4(xl, vl, x, v, accel_leader, vel_leader, params, ttc=None):
    dt = params['dt']

    # Keep track of the current follower's v_dot (accel) because it will be used in the calculations by its follower
    accel_follower = np.empty(4)
    vel_follower = np.empty(4)

    # K1
    k1_xl_dot, k1_vl_dot, k1_x_dot, k1_v_dot = internal_dynamics(
        xl, vl, x, v,
        accel_leader[0], vel_leader[0],
        params, ttc)
    accel_follower[0] = k1_v_dot
    vel_follower[0] = k1_x_dot

    # K2
    k2_xl_dot, k2_vl_dot, k2_x_dot, k2_v_dot = internal_dynamics(
        xl + 0.5 * dt * k1_xl_dot,
        vl + 0.5 * dt * k1_vl_dot,
        x + 0.5 * dt * k1_x_dot,
        v + 0.5 * dt * k1_v_dot,
        accel_leader[1], vel_leader[1],
        params, ttc
    )
    accel_follower[1] = k2_v_dot
    vel_follower[1] = k2_x_dot

    # K3
    k3_xl_dot, k3_vl_dot, k3_x_dot, k3_v_dot = internal_dynamics(
        xl + 0.5 * dt * k2_xl_dot,
        vl + 0.5 * dt * k2_vl_dot,
        x + 0.5 * dt * k2_x_dot,
        v + 0.5 * dt * k2_v_dot,
        accel_leader[2], vel_leader[2],
        params, ttc
    )
    accel_follower[2] = k3_v_dot
    vel_follower[2] = k3_x_dot

    # K4
    k4_xl_dot, k4_vl_dot, k4_x_dot, k4_v_dot = internal_dynamics(
        xl + dt * k3_xl_dot,
        vl + dt * k3_vl_dot,
        x + dt * k3_x_dot,
        v + dt * k3_v_dot,
        accel_leader[3], vel_leader[3],
        params, ttc
    )
    accel_follower[3] = k4_v_dot
    vel_follower[3] = k4_x_dot

    xl += (dt / 6) * (k1_xl_dot + 2 * k2_xl_dot + 2 * k3_xl_dot + k4_xl_dot)
    vl += (dt / 6) * (k1_vl_dot + 2 * k2_vl_dot + 2 * k3_vl_dot + k4_vl_dot)
    x += (dt / 6) * (k1_x_dot + 2 * k2_x_dot + 2 * k3_x_dot + k4_x_dot)
    v += (dt / 6) * (k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot)

    return xl, vl, x, v, accel_follower, vel_follower

# ttc calculation for vehicle 1
def calculate_ttc(x0, v0, x1, v1, params):
    dt = params['dt']
    T = params['T']
    
    num_steps = int(T / dt)

    time_for_v0_dot = 0.0

    # Start from 1 since initial vehicles should not already be collided
    for i in range(1, num_steps):
        x0, v0, x1, v1, _, _ = rk4(x0, v0, x1, v1, params, time_for_v0_dot)

        if x0 - x1 <= 0:
            print(f"TTC is {i * dt} seconds")
            return i * dt
        if v0 - v1 >= 0:
            return 1
        
        time_for_v0_dot += dt
    
    # print("huh?")
    return 1

# ttc for vehicles 2 and beyond
def internal_calculate_ttc(xl, vl, x, v, accel_leader, vel_leader, params):
    dt = params['dt']
    T = params['T']
    
    num_steps = int(T / dt)

    # Start from 1 since initial vehicles should not already be collided
    for i in range(1, num_steps):
        xl, vl, x, v, _, _ = internal_rk4(xl, vl, x, v, accel_leader, vel_leader, params)

        if xl - x <= 0:
            print(f"TTC is {i * dt} seconds")
            return i * dt
        if vl - v >= 0:
            return 1
        # print("debug:")
        # print(f"x at time {i * dt}= {x}")
        # print(f"xl - x at time {i * dt}= {xl - x}")
        # print(f"vl - v at time {i * dt}= {vl - v}")
    # print("huh?")
    # print("debug")
    # print(f"vl at time {i * dt}= {vl}")
    # print(f"v at time {i * dt}= {v}")
    # print(f"xl - x at time {i * dt}= {xl - x}")
    # print(f"vl - v at time {i * dt}= {vl - v}")
    return 1

# First simulate vehicles 0, 1 separately. Then simulate the remaining followers.
def simulate(init_positions, init_velocities, params):
    dt = params['dt']
    T = params['T']
    
    # Vehicle 0 is the global leader
    num_vehicles = len(init_positions)
    num_steps = int(T / dt)

    # Creates a nested acceleration array containing four rk4 steps for each time step for each vehicle
    nested_accels = [[[0.0 for _ in range(4)] for _ in range(num_steps)] for _ in range(num_vehicles)]
    all_vehicle_accels = np.array(nested_accels)
    # Creates a nested velocity array containing four rk4 steps for each time step for each vehicle
    nested_vels = [[[0.0 for _ in range(4)] for _ in range(num_steps)] for _ in range(num_vehicles)]
    all_vehicle_vels = np.array(nested_vels)
    
    velocities_toplot = np.zeros((num_steps, num_vehicles))
    velocities_toplot[0] = init_velocities

    time_for_v0_dot = 0.0

    # Simulate vehicle 0, vehicle 1 pair
    x0 = init_positions[0]
    v0 = init_velocities[0]
    x1 = init_positions[1]
    v1 = init_velocities[1]

    for i in range(1, num_steps):
        # Calculates ttc in seconds
        ttc = calculate_ttc(x0, v0, x1, v1, params)
        # if ttc == 0:
        #     print(f"at {i * dt} seconds")
        
        # Make sure to update the same variable names x0, v0, x1, v1 here because the addition is done in rk4.
        # Probably bad design.
        x0, v0, x1, v1, accel_follower, vel_follower = rk4(x0, v0, x1, v1, params, time_for_v0_dot, ttc)
        all_vehicle_accels[1, i] = accel_follower
        all_vehicle_vels[1, i] = vel_follower
        # print(f"Vehicle 1 accels at {i * dt} sec: {all_vehicle_accels[1, i]}")
        # print(f"Vehicle 1 vels at {i * dt} sec: {all_vehicle_vels[1, i]}")

        time_for_v0_dot += dt

        velocities_toplot[i, 0] = v0
        velocities_toplot[i, 1] = v1

    # Simulate internal Leader, Follower pair(s)
    if num_vehicles > 2:
        # Start from Vehicle 2 (third vehicle counting global leader as Vehicle 0)
        for i in range(2, num_vehicles):

            xl = init_positions[i - 1]
            vl = init_velocities[i - 1]
            x = init_positions[i]
            v = init_velocities[i]

            for j in range(1, num_steps):
                # Contains four element array with accel at each rk4 step for the current time step and current leader
                curr_time_leader_accel = all_vehicle_accels[i - 1, j]
                curr_time_leader_vel = all_vehicle_vels[i - 1, j]

                ttc = internal_calculate_ttc(xl, vl, x, v, curr_time_leader_accel, curr_time_leader_vel, params)
                # if ttc == 0:
                #     print(f"at {j * dt} seconds")
                
                xl, vl, x, v, accel_follower, vel_follower = internal_rk4(xl, vl, x, v, curr_time_leader_accel, curr_time_leader_vel, params, ttc)
                all_vehicle_accels[i, j] = accel_follower
                all_vehicle_vels[i, j] = vel_follower
                # print(f"Vehicle {i} accels at {j * dt} sec: {all_vehicle_accels[i, j]}, {accel_follower}")
                # print(f"Vehicle {i} vels at {j * dt} sec: {all_vehicle_vels[i, j]}")
                # print(f"Why not {vel_follower}?")

                velocities_toplot[j, i] = v
    
    return velocities_toplot

if __name__ == "__main__":
    # Set initial conditions
    # (position, velocity)
    initial_pairs = [
        (1.30, 1.10), # Vehicle 0
        (1.00, 1.30), # Vehicle 1
        (0.70, 1.50), # Vehicle 2
        (0.40, 1.70), # Vehicle 3
        (0.10, 1.90), # Vehicle 4
    ]

    positions_list, velocities_list = zip(*initial_pairs)

    init_positions = np.array(positions_list)
    init_velocities = np.array(velocities_list)

    num_vehicles = len(init_positions)

    params = {
        'alpha': 2.0,
        'beta': 3.0,
        'vmax_desired': 1.964,
        'dt': 0.01,
        'T': 10.0
    }

    # Simulation
    velocities_toplot = simulate(init_positions, init_velocities, params)
    time_toplot = np.linspace(0, params['T'], len(velocities_toplot))








    # Plot Velocity vs Time
    plt.figure(figsize=(10, 7))
    for i in range(num_vehicles):
        plt.plot(time_toplot, velocities_toplot[:, i], label=f'Vehicle {i} Velocity')
    # for i in range(1, num_vehicles):
    #     relative_velocity = velocities_toplot[:, i - 1] - velocities_toplot[:, i]
    #     plt.plot(time_toplot, relative_velocity, label=f'Vehicle {i-1} relative to {i}')
    plt.title('Vehicle Velocities w/ Acceleration vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ttc_mult1.png')



    # Plot Velocity between Vehicles vs Time
    plt.figure(figsize=(10, 7))
    for i in range(1, num_vehicles):
        relative_velocity = velocities_toplot[:, i - 1] - velocities_toplot[:, i]
        plt.plot(time_toplot, relative_velocity, label=f'Vehicle {i-1} - Vehicle {i}')
    plt.title('Vehicle Velocities w/ Acceleration vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ttc_mult2.png')


    # # Plot vl-v vs. xl-x
    # plt.figure(figsize=(10, 7))
    # plt.plot(rel_x, rel_v, label='trajectory', color='blue')
    # plt.scatter(rel_x[0], rel_v[0], color='cyan', s=100, zorder=5, label='start')
    # plt.scatter(rel_x[-1], rel_v[-1], color='orange', s=100, zorder=5, label='end')

    # plt.title('Relative Velocity vs. Relative Position')
    # plt.xlabel('Relative Position ($x_l - x$)')
    # plt.ylabel('Relative Velocity ($v_l - v$)')
    # plt.grid(True)
    # plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    # plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('temp1.png')

    # # Plot ttc vs. time
    # plt.figure(figsize=(10, 7))
    # plt.plot(time_plot, ttc_plot, label='traj', color='blue')
    # plt.title('TTC vs. Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('TTC (s)')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('temp_time.png')