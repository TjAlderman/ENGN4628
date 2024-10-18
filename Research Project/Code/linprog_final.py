import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from src.newer_model import HEV

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
    dt = np.diff(ts)  # Assuming constant time step
    
    # Get vehicle parameters from the newer_model
    battery_capacity = hybrid.battery_capacity
    initial_charge = hybrid.initial_charge
    max_IC_torque = hybrid.max_torque(v,"IC")
    max_EV_torque = hybrid.max_torque(v,"EV")
    max_Regen_torque = hybrid.max_torque(v,"Regen")
    cost_fuel = hybrid.cost_fuel
    cost_energy = hybrid.cost_battery

    # Get power per torque for IC and EV
    IC_power_per_torque = hybrid.IC_power_per_torque(v)
    EV_power_per_torque = hybrid.EV_power_per_torque(v)
    Regen_power_per_torque = hybrid.Regen_power_per_torque(v)

    IC_torque_cost = cost_fuel * IC_power_per_torque
    EV_torque_cost = cost_energy * EV_power_per_torque
    Regen_torque_cost = -cost_energy * Regen_power_per_torque

    # Set up linear optimization
    num_variables = 3 * intervals  # IC torque, EV torque, regen torque for each interval

    # Cost function
    f = np.concatenate((
        dt * IC_torque_cost,  # IC engine cost
        dt * EV_torque_cost,  # EV motor cost
        dt * Regen_torque_cost  # Regen benefit
    ))

    # Constraints (A*x <= b)
    A = np.zeros((3*intervals, num_variables))
    b = np.zeros(3*intervals)

    EV_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(EV_power_per_torque)
    Regen_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(Regen_power_per_torque)
    torque_matrix = np.eye(intervals)

    A_charge = np.concatenate((np.zeros((intervals, intervals)), -EV_charge_matrix, Regen_charge_matrix), axis=1)
    A_discharge = np.concatenate((np.zeros((intervals, intervals)), EV_charge_matrix, -Regen_charge_matrix), axis=1)
    A_torque = np.concatenate((torque_matrix, torque_matrix, -torque_matrix), axis=1)

    b_charge = np.ones(intervals) * (battery_capacity-initial_charge) # Max charge limit
    b_discharge = np.ones(intervals) * -initial_charge # Max discharge limit
    b_torque = T_req # Torque requirement

    A = np.concatenate((A_charge, A_discharge, A_torque), axis=1)
    b = np.concatenate((b_charge, b_discharge, b_torque), axis=0)

    # Bounds
    lb = np.zeros(num_variables)
    ub = np.concatenate((max_IC_torque,max_EV_torque,max_Regen_torque))

    # Solve linear programming problem
    try:
        res = linprog(f, A_ub=A, b_ub=b, bounds=(lb, ub), method='highs')
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(1)

    if res.success:
        # Process results
        IC_torque = res.x[:intervals]
        EV_torque = res.x[intervals:2*intervals]
        Regen_torque = res.x[2*intervals:]

        # Calculate total torque and battery charge
        total_torque = IC_torque + EV_torque - Regen_torque
        battery_charge = initial_charge + np.cumsum(dt * (EV_torque * EV_power_per_torque - Regen_torque * Regen_power_per_torque))

        # Plot results
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

        ax1.plot(ts, T_req, label='Required Torque', color='black', linestyle='--')
        ax1.plot(ts, IC_torque, label='IC Torque', color='red')
        ax1.plot(ts, EV_torque, label='EV Torque', color='green')
        ax1.plot(ts, -Regen_torque, label='Regen Torque', color='blue')
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