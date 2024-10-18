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

    # # Visualize torque requirement
    # plt.figure(figsize=(12, 6))
    # plt.plot(ts, T_req, label='Torque Requirement')
    # plt.plot(ts, T_max_IC, label='Max IC Torque', linestyle='--')
    # plt.plot(ts, T_max_EV, label='Max EV Torque', linestyle=':')
    # plt.plot(ts, IC_torque_cost, label='IC Torque Cost', linestyle='-.')
    # plt.plot(ts, EV_torque_cost, label='EV Torque Cost', linestyle='-.')
    # plt.plot(ts, Regen_torque_cost, label='Regen Torque Cost', linestyle='-.')
    # plt.title('Torque Requirement, Maximum Torques, and Torque Costs over Time')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Torque (Nm) / Cost')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    
    ########## Linear Programming ##########
    intervals = len(ts)

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

    EV_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(EV_power_per_torque)@np.diag(dt)
    Regen_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(Regen_power_per_torque)@np.diag(dt)
    torque_matrix = np.eye(intervals)

    A_charge = np.concatenate((np.zeros((intervals, intervals)), -EV_charge_matrix, Regen_charge_matrix), axis=1)
    A_discharge = np.concatenate((np.zeros((intervals, intervals)), EV_charge_matrix, -Regen_charge_matrix), axis=1)
    A_torque = np.concatenate((-torque_matrix, -torque_matrix, torque_matrix), axis=1)

    b_charge = np.ones(intervals) * (battery_capacity-initial_charge) # Max charge limit
    b_discharge = np.ones(intervals) * initial_charge # Max discharge limit
    b_discharge[-1] = 0  # Final charge must be at least as high as initial charge
    b_torque = T_req # Torque requirement

    A = np.concatenate((A_charge, A_discharge, A_torque), axis=0)
    b = np.concatenate((b_charge, b_discharge, b_torque), axis=-1)

    # # Constraints (A*x <= b)
    # A = np.zeros((intervals, num_variables))
    # b = np.zeros(intervals)

    # # Power constraint matrix
    # torque_matrix = np.eye(intervals)
    # A = np.concatenate((-torque_matrix, -torque_matrix, torque_matrix), axis=1)

    # # Power constraint vector (torque requirement)
    # b = T_req

    # Bounds
    lb = np.zeros(num_variables)
    ub = np.concatenate((T_max_IC, T_max_EV, T_max_Regen),axis=-1)

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
        EV_torque = res.x[intervals:2*intervals]
        Regen_torque = res.x[2*intervals:]

        # Calculate total torque and battery charge
        total_torque = IC_torque + EV_torque - Regen_torque
        battery_charge = initial_charge + np.cumsum(dt * (EV_torque * EV_power_per_torque - Regen_torque * Regen_power_per_torque))

        # Create a simple plot of the optimization outputs against time
        plt.figure(figsize=(12, 8))
        plt.plot(ts, IC_torque, label='IC Torque', color='red')
        plt.plot(ts, EV_torque, label='EV Torque', color='green')
        plt.plot(ts, -Regen_torque, label='Regen Torque', color='blue')
        plt.plot(ts, T_req, label='Required Torque', color='black', linestyle='--')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Optimization Outputs vs Time')
        plt.legend()
        plt.grid(True)
        plt.show()

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
