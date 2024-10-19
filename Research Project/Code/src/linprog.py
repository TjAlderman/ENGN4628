import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from state import HEV

class DataFrame:
    def __init__(self):
        pass

def state_dynamics(a:np.ndarray,v:np.ndarray,alpha:np.ndarray,t:np.ndarray,plot:bool=False)->DataFrame:
    df = DataFrame()
    df.dt = np.concatenate([[1e-7],np.diff(t)])  # Assuming constant time step

    # Initialize HEV class
    hybrid = HEV()

    # Get constants from the newer_model
    df.battery_capacity = hybrid.battery_capacity
    df.initial_charge = hybrid.initial_charge
    df.cost_fuel = hybrid.cost_fuel
    df.cost_energy = hybrid.cost_energy

    # Calculate torque requirement for each timestep
    df.T_req = hybrid.torque(a,v,alpha)
    w = hybrid.w(v)
    a_n = hybrid._a_n(v)

    # Calculate max torque for each timestep
    df.T_max_IC = hybrid.max_torque(a_n,"ICE")
    df.T_max_EV = hybrid.max_torque(a_n,"EV")
    df.T_max_Regen = hybrid.max_torque(a_n,"Regen")

    # Get power per torque for IC and EV
    df.IC_power_per_torque = hybrid.power_per_torque(w,"ICE")
    df.EV_power_per_torque = hybrid.power_per_torque(w,"EV")

    # Hacky
    df.IC_power_per_torque[df.IC_power_per_torque<0]=0
    df.EV_power_per_torque[df.IC_power_per_torque<0]=0
    assert df.IC_power_per_torque.min()>=0 and df.EV_power_per_torque.min()>=0, "Power per torque must be positive"

    df.IC_torque_cost = df.IC_power_per_torque
    df.EV_torque_cost = df.EV_power_per_torque
    # TJ: assume a flat efficiency curve for regenerative braking
    # Simplification but not able to find any good sources on efficiency
    # curves of regenerative breaks. Also just cbf.
    df.Regen_power_per_torque = np.ones_like(df.IC_power_per_torque)*40
    # TJ: I set regen torque cost to a very small negative value. This way, the optimiser
    # will always choose to assign free energy to regen (to minimise cost), but it will
    # never use the IC or EM to generate energy that's put into regen because it costs
    # more to generate the energy than it will save off regen
    df.Regen_torque_cost = -1e-7 * np.ones_like(df.IC_power_per_torque)

    if plot:
        # # Visualize torque requirement and costs
        plt.figure(figsize=(9, 8))
        # plt.plot(t, T_req, label='Torque Requirement')
        plt.plot(t, df.Regen_torque_cost, label='Regen Torque Cost', linestyle='-.')
        plt.plot(t, df.IC_torque_cost, label='IC Torque Cost', linestyle='-.')
        plt.plot(t, df.EV_torque_cost, label='EV Torque Cost')
        # plt.plot(t, hybrid.w(v=v), label='Angular velocity')
        plt.title('Torque Requirement, Maximum Torques, and Torque Costs over Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm) / Cost')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    return df
    
def linprog_optimiser(df:DataFrame,plot:bool=False):
    intervals = len(t)
    num_variables = 3 * intervals  # IC torque, EV torque, regen torque for each interval

    # Cost function
    f = np.concatenate((
        df.dt * df.IC_torque_cost,  # IC engine cost
        df.dt * df.EV_torque_cost,  # EV motor cost
        df.dt * df.Regen_torque_cost  # Regen benefit
    ))

    # Constraints (A*x <= b)
    A = np.zeros((3*intervals, num_variables))
    b = np.zeros(3*intervals)

    EV_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(df.EV_power_per_torque)@np.diag(df.dt)
    Regen_charge_matrix = np.tril(np.ones((intervals, intervals)))@np.diag(df.Regen_power_per_torque)@np.diag(df.dt)
    torque_matrix = np.eye(intervals)

    A_charge = np.concatenate((np.zeros((intervals, intervals)), -EV_charge_matrix, Regen_charge_matrix), axis=1)
    A_discharge = np.concatenate((np.zeros((intervals, intervals)), EV_charge_matrix, -Regen_charge_matrix), axis=1)
    A_torque = np.concatenate((-torque_matrix, -torque_matrix, torque_matrix), axis=1)

    b_charge = np.ones(intervals) * (df.battery_capacity-df.initial_charge) # Max charge limit
    b_discharge = np.ones(intervals) * df.initial_charge # Max discharge limit
    b_discharge[-1] = 0  # Final charge must be at least as high as initial charge
    b_torque = -df.T_req # Torque requirement

    A = np.concatenate((A_charge, A_discharge, A_torque), axis=0)
    b = np.concatenate((b_charge, b_discharge, b_torque), axis=-1)

    # Bounds
    lb = np.zeros(num_variables)
    ub = np.concatenate((df.T_max_IC, df.T_max_EV, df.T_max_Regen),axis=-1)

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
        battery_charge = df.initial_charge + np.cumsum(df.dt * (EV_torque * df.EV_power_per_torque - Regen_torque * df.Regen_power_per_torque))

        if plot:
            # Create a simple plot of the optimization outputs against time
            plt.figure(figsize=(6, 4))
            plt.plot(t, IC_torque, label='IC Torque', color='red')
            plt.plot(t, EV_torque, label='EV Torque', color='green')
            plt.plot(t, -Regen_torque, label='Regen Torque', color='blue')
            plt.plot(t, df.T_req, label='Required Torque', color='black', linestyle='--')
            
            plt.xlabel('Time (s)')
            plt.ylabel('Torque (Nm)')
            plt.title('Optimization Outputs vs Time')
            plt.legend()
            plt.grid(True)
            plt.show()

            # Plot results
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

            ax1.plot(t, df.T_req, label='Required Torque', color='black', linestyle='--')
            ax1.plot(t, IC_torque, label='IC Torque', color='red')
            ax1.plot(t, EV_torque, label='EV Torque', color='green')
            ax1.plot(t, -Regen_torque, label='Regen Torque', color='blue')
            ax1.plot(t, total_torque, label='Total Torque', color='purple')
            ax1.set_ylabel('Torque (Nm)')
            ax1.legend()
            ax1.set_title('Torque Distribution')

            ax2.plot(t, battery_charge / 3600, label='Battery Charge')
            ax2.set_ylabel('Battery Charge (kWh)')
            ax2.legend()
            ax2.set_title('Battery Charge')

            ax3.plot(t, v, label='Velocity')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Velocity (m/s)')
            ax3.legend()
            ax3.set_title('Vehicle Velocity')

            plt.tight_layout()
            plt.show()
    else:
        print("Optimization failed:", res.message)
        
        
def main(a:np.ndarray,v:np.ndarray,alpha:np.ndarray,t:np.ndarray):
    df=state_dynamics(a=a,v=v,alpha=alpha,t=t,plot=True)
    linprog_optimiser(df=df,plot=True)

if __name__ == "__main__":
    v = sys.argv[2] if len(sys.argv)>2 else np.load(f'/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-velocity.npy')
    a = sys.argv[3] if len(sys.argv)>3 else np.load(f'/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-acceleration.npy')
    alpha = sys.argv[1] if len(sys.argv)>1 else np.load(f'/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-slope.npy')
    t = sys.argv[4] if len(sys.argv)>4 else np.load(f'/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-time.npy')
    main(v=v,a=a,alpha=alpha,t=t)
