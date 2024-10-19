import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.linalg import block_diag
from state import HEV
from tqdm import tqdm


class DataFrame:
    def __init__(self):
        pass


def state_dynamics(
    a: np.ndarray, v: np.ndarray, alpha: np.ndarray, t: np.ndarray, plot: bool = False
) -> DataFrame:
    df = DataFrame()
    df.dt = np.concatenate([[1e-7], np.diff(t)])  # Assuming constant time step

    # Initialize HEV class
    hybrid = HEV()

    # Get constants from the newer_model
    df.battery_capacity = hybrid.battery_capacity
    df.initial_charge = hybrid.initial_charge

    # Calculate power requirement for each timestep
    df.P_req = hybrid.power(a, v, alpha)/1000 # W -> kW
    w = hybrid.w(v)
    a_n = hybrid._a_n(v)

    # Calculate max power for each timestep
    df.P_max_IC = hybrid.max_power(a_n, "ICE")
    df.P_max_EV = hybrid.max_power(a_n, "EV")
    df.P_max_REGEN = hybrid.max_power(a_n, "Regen")

    # Get efficiency curves for each timestep
    df.IC_efficiency = hybrid.efficiency(w, "ICE")
    df.EV_efficiency = hybrid.efficiency(w, "EV")
    
    # Map efficiency curve to arbitrary cost value that gives desired behaviour
    df.IC_efficiency_cost = 1/df.IC_efficiency
    df.EV_efficiency_cost = 1/df.EV_efficiency
    # TJ: assume a flat efficiency curve for regenerative braking
    # Simplification but not able to find any good sources on efficiency
    # curves of regenerative breaks. Also just cbf.
    df.REGEN_efficiency = np.ones_like(df.IC_efficiency) * 0.4
    # TJ: I set regen power cost to a very small negative value. This way, the optimiser
    # will always choose to assign free energy to regen (to minimise cost), but it will
    # never use the IC or EM to generate energy that's put into regen because it costs
    # more to generate the energy than it will save off regen
    df.REGEN_efficiency_cost = -1e-7 * np.ones_like(df.IC_efficiency)

    if plot:
        # # Visualize power requirement and costs
        plt.figure(figsize=(5, 4))
        # plt.plot(t, P_req, label='Power Requirement')
        plt.plot(t, df.REGEN_efficiency_cost, label="Regen Power Cost", linestyle="-.")
        plt.plot(t, df.IC_efficiency_cost, label="IC Power Cost", linestyle="-.")
        plt.plot(t, df.EV_efficiency_cost, label="EV Power Cost")
        # plt.plot(t, hybrid.w(v=v), label='Angular velocity')
        plt.title("Cost Variables")
        plt.xlabel("Time (s)")
        plt.ylabel("Arbitrary Cost")
        plt.legend()
        plt.grid(True)
        plt.show()

    return df


def linprog_optimiser(df: DataFrame, plot: bool = False):
    num_intervals = len(t)
    num_intervals = 500  # DEBUGGING
    num_variables = 6

    # Define the first two intervals by hand
    # Cost fn
    f = np.array(
        [
            [
                0,
                df.EV_efficiency_cost[0],
                df.REGEN_efficiency_cost[0],
                df.IC_efficiency_cost[0],
                -1e7,
                0,
                0,
                df.EV_efficiency_cost[1],
                df.REGEN_efficiency_cost[1],
                df.IC_efficiency_cost[1],
                -1e7,
                0,
            ]
        ]
    )
    # Inequality conditions
    A = np.array(
        [
            [-1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Initial charge - discharge + charge >= 0
            [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0, 0],
        ]
    )
    b = np.array([[0], [0]])
    # Equality conditions
    A_eq = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Initial charge
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # EV+REGEN+ICE+BRAKES=P
            [
                1,
                -1,
                -1,
                0,
                0,
                -1,
                0,
                0,
                0,
                0,
                0,
                0,
            ],  # initial charge - discharge + regen = final charge
            [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0],  # final charge = initial charge of next state
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # EV+REGEN+ICE+BRAKES=P
            [
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                -1,
                -1,
                0,
                0,
                -1,
            ],  # initial charge - discharge + regen = final charge
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )  # Used in loop to set initial charge
    b_eq = [df.initial_charge, df.P_req[0], 0, 0, df.P_req[1], 0, 0]

    # Bounds
    ub = [
        df.battery_capacity,
        df.P_max_EV[0], #EV
        0, # REGEN 
        df.P_max_IC[0], # ICE
        0, # Braking
        df.battery_capacity,
        df.battery_capacity,
        df.P_max_EV[1],
        0,
        df.P_max_IC[1],
        0,
        df.battery_capacity,
    ]
    lb = [0, 0, df.P_max_REGEN[0], 0, -1e7, 0, 0, 0, df.P_max_REGEN[1], 0, -1e7, 0]

    # Concatenate remaining intervals
    for i in tqdm(range(2, num_intervals), ncols=80):
        new_f = np.array([[0, df.EV_efficiency_cost[i], df.REGEN_efficiency_cost[i], df.IC_efficiency_cost[i], -1e7, 0]])
        f = np.concatenate([f, new_f], axis=1)

        new_A = np.array([[-1, 1, -1, 0, 0, 0]])
        A = block_diag(A, new_A)

        new_b = np.array([[0]])
        b = np.concatenate([b, new_b], axis=0)

        new_A_eq = np.array(
            [[0, 1, 1, 1, 1, 0], [1, -1, -1, 0, 0, -1]]  # EV+REGEN+ICE+BRAKES=P
        )  # Initial charge - discharge + regen = final charge
        A_eq = block_diag(A_eq, new_A_eq)
        A_eq[-3, -num_variables] = -1
        if not i == num_intervals - 1:
            conservation_bat_q = np.zeros((1, A_eq.shape[1]))
            conservation_bat_q[0, -1] = 1
            A_eq = np.concatenate([A_eq, conservation_bat_q], axis=0)
            new_b_eq = [df.P_req[i], 0, 0]

        else:
            new_b_eq = [df.P_req[i], 0]
        b_eq += new_b_eq

        new_ub = [df.battery_capacity, df.P_max_EV[i], 0, df.P_max_IC[i], 0, df.battery_capacity]
        new_lb = [0, 0, df.P_max_REGEN[i], 0, -1e7, 0]
        ub += new_ub
        lb += new_lb
    b_eq = np.array(b_eq)

    # Solve linear programming problem
    try:
        res = linprog(f, A_ub=A, b_ub=b, A_eq=A_eq, b_eq=b_eq, bounds=list(zip(lb, ub)))
    except Exception as e:
        print(f"Optimization failed: {e}")
        sys.exit(1)

    if res.success:
        # Process results
        reshaped_x = res.x.reshape(num_intervals, num_variables)
        initial_charge = reshaped_x[:, 0]  # initial charge for each interval
        EV_power = reshaped_x[:, 1]  # ice_power for each interval
        REGEN_power = reshaped_x[:, 2]  # ev_power for each interval
        IC_power = reshaped_x[:, 3]  # ev_power for each interval
        brake_power = reshaped_x[:, 4]  # final_charge for each interval
        final_charge = reshaped_x[:, 5]  # final_charge for each interval

        # Calculate total power and battery charge
        total_power = IC_power + EV_power + REGEN_power

        if plot:
            # Create a simple plot of the optimization outputs against time
            fig, axs = plt.subplots(2, 1, figsize=(6, 4))
            axs[0].plot(t[:num_intervals], IC_power, label="IC power", color="red")
            axs[0].plot(t[:num_intervals], EV_power, label="EV power", color="green")
            axs[0].plot(t[:num_intervals], REGEN_power, label="REGEN power", color="orange")
            axs[0].plot(t[:num_intervals], brake_power, label="Brake power", color="blue")
            # axs[0].plot(t[:num_intervals], total_power, label="Total power", color="blue")
            axs[0].plot(
                t[:num_intervals],
                df.P_req[:num_intervals],
                label="Required power",
                color="black",
                linestyle="--",
            )
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Power (kW)")
            axs[0].set_title("Optimization Outputs vs Time")
            axs[0].legend()
            axs[0].grid(True)

            axs[1].plot(
                t[:num_intervals], initial_charge, label="Initial charge", color="red"
            )
            axs[1].plot(
                t[:num_intervals], final_charge, label="Final charge", color="green"
            )
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Charge (kWh)")
            axs[1].set_title("Optimization Outputs vs Time")
            axs[1].legend()
            axs[1].grid(True)
            plt.show()
    else:
        print("Optimization failed:", res.message)


def main(a: np.ndarray, v: np.ndarray, alpha: np.ndarray, t: np.ndarray):
    df = state_dynamics(a=a, v=v, alpha=alpha, t=t, plot=True)
    linprog_optimiser(df=df, plot=True)


if __name__ == "__main__":
    v = (
        sys.argv[2]
        if len(sys.argv) > 2
        else np.load(
            f"/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-velocity.npy"
        )
    )
    a = (
        sys.argv[3]
        if len(sys.argv) > 3
        else np.load(
            f"/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-acceleration.npy"
        )
    )
    alpha = (
        sys.argv[1]
        if len(sys.argv) > 1
        else np.load(
            f"/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-slope.npy"
        )
    )
    t = (
        sys.argv[4]
        if len(sys.argv) > 4
        else np.load(
            f"/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/trip-time.npy"
        )
    )
    main(v=v, a=a, alpha=alpha, t=t)
