import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from src.state import HEV

def main():
    # Load data
    data_name = 'short_5min'  # Consider making this a command-line argument
    fake = True

    try:
        if fake:
            alpha = np.load(f'data/fake-slope_{data_name}.npy')
            v = np.load(f'data/fake-velocity_{data_name}.npy')
            a = np.load(f'data/fake-acceleration_{data_name}.npy')
            ts = np.load(f'data/fake-time_{data_name}.npy')
        else:
            alpha = np.load(f'data/slope_{data_name}.npy')
            v = np.load(f'data/velocity_{data_name}.npy')
            a = np.load(f'data/acceleration_{data_name}.npy')
            ts = np.load(f'data/time_{data_name}.npy')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)

    dt = np.diff(ts)  # Assuming constant time step
    alpha = alpha[1:]
    v = v[1:]
    a = a[1:]
    ts = ts[1:]

    # Initialize HEV class
    hybrid = HEV()

    # Get constants from the newer_model
    battery_capacity = hybrid.battery_capacity
    initial_charge = hybrid.initial_charge
    cost_fuel = hybrid.cost_fuel
    cost_energy = hybrid.cost_energy

    # Calculate torque requirement for each timestep
    T_req = hybrid.torque(a,v,alpha)
    w = hybrid.w(v)
    a_n = hybrid._a_n(v)

    # Calculate max torque for each timestep
    T_max_IC = hybrid.max_torque(a_n,"ICE")
    T_max_EV = hybrid.max_torque(a_n,"EV")
    T_max_Regen = hybrid.max_torque(a_n,"Regen")

    # Get power per torque for IC and EV
    IC_power_per_torque = hybrid.power_per_torque(w,"ICE")
    EV_power_per_torque = hybrid.power_per_torque(w,"EV")
    Regen_power_per_torque = hybrid.power_per_torque(w,"Regen")

    IC_torque_cost = cost_fuel * IC_power_per_torque
    EV_torque_cost = cost_energy * EV_power_per_torque
    Regen_torque_cost = -cost_energy * Regen_power_per_torque

    # Visualize torque requirement and angular velocity
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot torque and costs on the primary y-axis
    ax1.plot(ts, T_req, label='Torque Requirement')
    ax1.plot(ts, T_max_IC, label='Max IC Torque', linestyle='--')
    ax1.plot(ts, T_max_EV, label='Max EV Torque', linestyle=':')
    ax1.plot(ts, IC_torque_cost, label='IC Torque Cost', linestyle='-.')
    ax1.plot(ts, EV_torque_cost, label='EV Torque Cost', linestyle='-.')
    ax1.plot(ts, Regen_torque_cost, label='Regen Torque Cost', linestyle='-.')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Torque (Nm) / Cost')

    # Create a secondary y-axis for angular velocity
    ax2 = ax1.twinx()
    ax2.plot(ts, w, label='Angular Velocity', color='purple')
    ax2.set_ylabel('Angular Velocity (rad/s)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('Torque Requirement, Maximum Torques, Torque Costs, and Angular Velocity over Time')
    plt.grid(True)
    plt.show()
    
    ########## Linear Programming ##########
    intervals = len(ts)

    num_variables = 4 * intervals  # IC torque, EV torque, regen torque, and battery charge for each interval

    # Cost function
    f = np.concatenate((
        dt * IC_torque_cost,  # IC engine cost
        dt * EV_torque_cost,  # EV motor cost
        dt * Regen_torque_cost,  # Regen benefit
        np.zeros(intervals)  # No direct cost for battery charge
    ))

    # Constraints (A*x <= b)
    A = np.zeros((3*intervals + 1, num_variables))
    b = np.zeros(3*intervals + 1)

    # Battery charge constraints
    for i in range(intervals):
        if i == 0:
            A[i, 3*intervals+i] = 1
            A[i, intervals+i] = -dt[i] * EV_power_per_torque[i]
            A[i, 2*intervals+i] = dt[i] * Regen_power_per_torque[i]
            b[i] = initial_charge
        else:
            A[i, 3*intervals+i] = 1
            A[i, 3*intervals+i-1] = -1
            A[i, intervals+i] = -dt[i] * EV_power_per_torque[i]
            A[i, 2*intervals+i] = dt[i] * Regen_power_per_torque[i]
            b[i] = 0

    # Torque requirement constraint
    A_torque = np.concatenate((-np.eye(intervals), -np.eye(intervals), np.eye(intervals), np.zeros((intervals, intervals))), axis=1)
    A[intervals:2*intervals] = A_torque
    b[intervals:2*intervals] = T_req

    # Battery capacity constraints
    A[2*intervals:3*intervals, 3*intervals:] = np.eye(intervals)
    b[2*intervals:3*intervals] = battery_capacity

    # Final charge constraint
    A[-1, -1] = -1
    b[-1] = -initial_charge

    # Bounds
    lb = np.zeros(num_variables)
    ub = np.concatenate((T_max_IC, T_max_EV, T_max_Regen, np.ones(intervals) * battery_capacity))

    # Ensure lb and ub have the same shape
    if lb.shape != ub.shape:
        print(f"Shape mismatch: lb shape is {lb.shape}, ub shape is {ub.shape}")
        print(f"T_max_IC shape: {T_max_IC.shape}")
        print(f"T_max_EV shape: {T_max_EV.shape}")
        print(f"T_max_Regen shape: {T_max_Regen.shape}")
        sys.exit(1)

    # Solve linear programming problem
    try:
        res = linprog(f, A_ub=A, b_ub=b, bounds=list(zip(lb, ub)))
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(1)

    if res.success:
        # Process results
        IC_torque = res.x[:intervals]
        plt.plot(ts, IC_torque)
        plt.show()
        EV_torque = res.x[intervals:2*intervals]
        plt.plot(ts, EV_torque)
        plt.show()
        Regen_torque = res.x[2*intervals:3*intervals]
        plt.plot(ts, Regen_torque)
        plt.show()
        battery_charge = res.x[3*intervals:]

        # Calculate total torque
        total_torque = IC_torque + EV_torque - Regen_torque

        # Create visualizations
        plt.style.use('seaborn')
        fig, axs = plt.subplots(4, 1, figsize=(12, 20), sharex=True)
        fig.suptitle('Hybrid Electric Vehicle Optimization Results', fontsize=16)

        # Plot 1: Torque Distribution
        axs[0].plot(ts, T_req, label='Required Torque', color='black', linestyle='--')
        axs[0].plot(ts, IC_torque, label='IC Torque', color='red')
        axs[0].plot(ts, EV_torque, label='EV Torque', color='green')
        axs[0].plot(ts, -Regen_torque, label='Regen Torque', color='blue')
        axs[0].plot(ts, total_torque, label='Total Torque', color='purple')
        axs[0].set_ylabel('Torque (Nm)')
        axs[0].legend()
        axs[0].set_title('Torque Distribution')

        # Plot 2: Battery Charge
        axs[1].plot(ts, battery_charge / (3600*1000), label='Battery Charge')
        axs[1].set_ylabel('Battery Charge (kWh)')
        axs[1].legend()
        axs[1].set_title('Battery Charge')

        # Plot 3: Vehicle Velocity and Road Slope
        ax3_twin = axs[2].twinx()
        axs[2].plot(ts, v, label='Velocity', color='orange')
        ax3_twin.plot(ts, alpha, label='Road Slope', color='green', linestyle='--')
        axs[2].set_ylabel('Velocity (m/s)')
        ax3_twin.set_ylabel('Road Slope (rad)')
        axs[2].legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        axs[2].set_title('Vehicle Velocity and Road Slope')

        # Plot 4: Power Distribution
        IC_power = IC_torque * w * IC_power_per_torque
        EV_power = EV_torque * w * EV_power_per_torque
        Regen_power = Regen_torque * w * Regen_power_per_torque
        axs[3].plot(ts, IC_power, label='IC Power', color='red')
        axs[3].plot(ts, EV_power, label='EV Power', color='green')
        axs[3].plot(ts, Regen_power, label='Regen Power', color='blue')
        axs[3].set_ylabel('Power (W)')
        axs[3].legend()
        axs[3].set_title('Power Distribution')

        plt.tight_layout()
        plt.show()

        # Save plots
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{data_name}_results.png'))
        plt.close()

        print(f"Results saved in {output_dir}")
    else:
        print("Optimization failed:", res.message)

if __name__ == "__main__":
    main()
