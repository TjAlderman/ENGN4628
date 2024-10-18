import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from src.newer_model import HEV

def main():
    # Load data
    data_name = 'short_5min'  # Consider making this a command-line argument
    try:
        alpha = np.load(f'data/fake-slope_{data_name}.npy')
        v = np.load(f'data/fake-velocity_{data_name}.npy')
        a = np.load(f'data/fake-acceleration_{data_name}.npy')
        ts = np.load(f'data/fake-time_{data_name}.npy')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        sys.exit(1)

    # Initialize HEV class
    hybrid = HEV()

    # Calculate torque requirement for each timestep
    F_t = hybrid.force_balance(a=a, v=v, alpha=alpha)
    T_req = F_t / hybrid._a_n(v)  # Convert force to torque

    # Visualize torque requirement
    plt.figure(figsize=(12, 6))
    plt.plot(ts, T_req, label='Torque Requirement')
    plt.title('Torque Requirement over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque (Nm)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Set parameters
    intervals = len(ts)
    dt = np.diff(ts)[0]  # Assuming constant time step
    
    # Get vehicle parameters from the newer_model
    battery_capacity = hybrid.battery_capacity
    initial_charge = hybrid.initial_charge
    min_charge = hybrid.min_charge
    max_charge = hybrid.max_charge
    max_ic_torque = hybrid.max_ic_torque
    max_ev_torque = hybrid.max_ev_torque
    max_regen_torque = hybrid.max_regen_torque
    ic_efficiency = hybrid.ic_efficiency
    ev_efficiency = hybrid.ev_efficiency
    regen_efficiency = hybrid.regen_efficiency
    cost_fuel = hybrid.cost_fuel
    cost_battery = hybrid.cost_battery

    # Get power per torque for IC and EV
    ic_power_per_torque = hybrid.IC_power_per_torque(v)
    ev_power_per_torque_discharge, ev_power_per_torque_charge = hybrid.EV_power_per_torque(v)

    # Set up linear optimization
    num_variables = 3 * intervals  # IC torque, EV torque, regen torque for each interval

    # Cost function
    f = np.concatenate((
        dt * cost_fuel * ic_power_per_torque,  # IC engine cost
        dt * cost_battery * ev_power_per_torque_discharge,  # EV motor cost
        -dt * cost_battery * ev_power_per_torque_charge  # Regen benefit
    ))

    # Inequality constraints
    A_ub = np.zeros((3 * intervals, num_variables))
    b_ub = np.zeros(3 * intervals)

    # Torque requirement constraints
    for i in range(intervals):
        A_ub[i, i] = 1  # IC torque
        A_ub[i, i + intervals] = 1  # EV torque
        A_ub[i, i + 2 * intervals] = -1  # Regen torque
        b_ub[i] = T_req[i]

    # Battery charge constraints
    charge_matrix = np.tril(np.ones((intervals, intervals)))

    # Constraint 1: Upper bound on battery charge
    A_ub[intervals:2*intervals, intervals:2*intervals] = dt * np.diag(ev_power_per_torque_discharge) @ charge_matrix
    A_ub[intervals:2*intervals, 2*intervals:] = -dt * np.diag(ev_power_per_torque_charge) @ charge_matrix
    b_ub[intervals:2*intervals] = max_charge - initial_charge

    # Constraint 2: Lower bound on battery charge
    A_ub[2*intervals:3*intervals, intervals:2*intervals] = -dt * np.diag(ev_power_per_torque_discharge) @ charge_matrix
    A_ub[2*intervals:3*intervals, 2*intervals:] = dt * np.diag(ev_power_per_torque_charge) @ charge_matrix
    b_ub[2*intervals:3*intervals] = initial_charge - min_charge

    # Debugging constraints
    print("Shape of A_ub:", A_ub.shape)
    print("Shape of b_ub:", b_ub.shape)
    print("Number of variables:", num_variables)

    # Check for NaN or infinity values
    print("NaN in A_ub:", np.isnan(A_ub).any())
    print("Inf in A_ub:", np.isinf(A_ub).any())
    print("NaN in b_ub:", np.isnan(b_ub).any())
    print("Inf in b_ub:", np.isinf(b_ub).any())

    # Print some values for sanity check
    print("First few rows of A_ub:")
    print(A_ub[:5, :5])
    print("First few values of b_ub:")
    print(b_ub[:5])

    # Check the condition number of A_ub
    print("Condition number of A_ub:", np.linalg.cond(A_ub))

    # Visualize the constraint matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(A_ub, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Visualization of A_ub matrix")
    plt.show()

    # Upper and lower bounds
    bounds = [(0, max_ic_torque)] * intervals + [(0, max_ev_torque)] * intervals + [(0, max_regen_torque)] * intervals

    plt.imshow(A_ub)
    plt.colorbar()
    plt.show()

    # Solve linear programming problem
    try:
        res = linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(1)

    if res.success:
        # Process results
        ic_torque = res.x[:intervals]
        ev_torque = res.x[intervals:2*intervals]
        regen_torque = res.x[2*intervals:]

        # Calculate total torque and battery charge
        total_torque = ic_torque + ev_torque - regen_torque
        battery_charge = initial_charge + np.cumsum(dt * (ev_torque * ev_power_per_torque_discharge - regen_torque * ev_power_per_torque_charge))

        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        ax1.plot(ts, T_req, label='Required Torque', color='black', linestyle='--')
        ax1.plot(ts, ic_torque, label='IC Torque', color='red')
        ax1.plot(ts, ev_torque, label='EV Torque', color='green')
        ax1.plot(ts, -regen_torque, label='Regen Torque', color='blue')
        ax1.plot(ts, total_torque, label='Total Torque', color='purple')
        ax1.set_ylabel('Torque (Nm)')
        ax1.legend()
        ax1.set_title('Torque Distribution')

        ax2.plot(ts, battery_charge / 3600, label='Battery Charge')
        ax2.set_ylabel('Battery Charge (kWh)')
        ax2.legend()
        ax2.set_title('Battery Charge')

        ax3.plot(ts, v, label='Velocity')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.legend()
        ax3.set_title('Vehicle Velocity')

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
