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
    df.cost_fuel = hybrid.cost_fuel
    df.cost_energy = hybrid.cost_energy

    # Calculate torque requirement for each timestep
    df.T_req = hybrid.torque(a, v, alpha)
    w = hybrid.w(v)
    a_n = hybrid._a_n(v)

    # Calculate max torque for each timestep
    df.T_max_IC = hybrid.max_torque(a_n, "ICE")
    df.T_max_EV = hybrid.max_torque(a_n, "EV")
    df.T_max_Regen = hybrid.max_torque(a_n, "Regen")

    # Get power per torque for IC and EV
    df.IC_power_per_torque = hybrid.power_per_torque(w, "ICE")
    df.EV_power_per_torque = hybrid.power_per_torque(w, "EV")

    # Hacky
    df.IC_power_per_torque[df.IC_power_per_torque < 0] = 0
    df.EV_power_per_torque[df.IC_power_per_torque < 0] = 0
    assert (
        df.IC_power_per_torque.min() >= 0 and df.EV_power_per_torque.min() >= 0
    ), "Power per torque must be positive"

    df.IC_torque_cost = df.IC_power_per_torque
    df.EV_torque_cost = df.EV_power_per_torque
    # TJ: assume a flat efficiency curve for regenerative braking
    # Simplification but not able to find any good sources on efficiency
    # curves of regenerative breaks. Also just cbf.
    df.Regen_power_per_torque = np.ones_like(df.IC_power_per_torque) * 40
    # TJ: I set regen torque cost to a very small negative value. This way, the optimiser
    # will always choose to assign free energy to regen (to minimise cost), but it will
    # never use the IC or EM to generate energy that's put into regen because it costs
    # more to generate the energy than it will save off regen
    df.Regen_torque_cost = -1e-7 * np.ones_like(df.IC_power_per_torque)

    if plot:
        # # Visualize torque requirement and costs
        plt.figure(figsize=(9, 8))
        # plt.plot(t, T_req, label='Torque Requirement')
        plt.plot(t, df.Regen_torque_cost, label="Regen Torque Cost", linestyle="-.")
        plt.plot(t, df.IC_torque_cost, label="IC Torque Cost", linestyle="-.")
        plt.plot(t, df.EV_torque_cost, label="EV Torque Cost")
        # plt.plot(t, hybrid.w(v=v), label='Angular velocity')
        plt.title("Torque Requirement, Maximum Torques, and Torque Costs over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm) / Cost")
        plt.legend()
        plt.grid(True)
        plt.show()

    return df


def linprog_optimiser(df: DataFrame, plot: bool = False):
    num_intervals = len(t)
    num_intervals = 50  # DEBUGGING
    num_variables = 4

    # Define the first two intervals by hand
    # Cost fn
    f = np.array(
        [
            [
                0,
                df.EV_torque_cost[0],
                df.IC_torque_cost[0],
                0,
                0,
                df.EV_torque_cost[1],
                df.IC_torque_cost[1],
                0,
            ]
        ]
    )
    # Inequality conditions
    A = np.array(
        [
            [-1, -1, 0, 0, 0, 0, 0, 0],  # Initial charge + discharge/charge >= 0
            [0, 0, 0, 0, -1, -1, 0, 0],
        ]
    )
    b = np.array([[0], [0]])
    # Equality conditions
    A_eq = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],  # Initial charge
            [0, 1, 1, 0, 0, 0, 0, 0],  # ICE+EV=P
            [
                1,
                -1,
                0,
                -1,
                0,
                0,
                0,
                0,
            ],  # initial charge + discharge/charge = final charge
            [0, 0, 0, 1, -1, 0, 0, 0],  # final charge = initial charge of next state
            [0, 0, 0, 0, 0, 1, 1, 0],  # ICE+EV=P
            [
                0,
                0,
                0,
                0,
                1,
                -1,
                0,
                -1,
            ],  # initial charge + discharge/charge = final charge
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )  # Used in loop to set initial charge
    b_eq = [df.initial_charge, df.T_req[0], 0, 0, df.T_req[1], 0, 0]

    # Bounds
    ub = [
        df.battery_capacity,
        200,
        170,
        df.battery_capacity,
        df.battery_capacity,
        200,
        170,
        df.battery_capacity,
    ]
    lb = [0, -40, 0, 0, 0, -40, 0, 0]

    # Concatenate remaining intervals
    for i in tqdm(range(2, num_intervals), ncols=80):
        new_f = np.array([[0, df.EV_torque_cost[i], df.IC_torque_cost[i], 0]])
        f = np.concatenate([f, new_f], axis=1)

        new_A = np.array([[-1, -1, 0, 0]])
        A = block_diag(A, new_A)

        new_b = np.array([[0]])
        b = np.concatenate([b, new_b], axis=0)

        new_A_eq = np.array(
            [[0, 1, 1, 0], [1, -1, 0, -1]]  # ICE+EV=P
        )  # Initial charge + discharge/charge = final charge
        A_eq = block_diag(A_eq, new_A_eq)
        A_eq[-3, -4] = -1
        if not i == num_intervals - 1:
            conservation_bat_q = np.zeros((1, A_eq.shape[1]))
            conservation_bat_q[0, -1] = 1
            A_eq = np.concatenate([A_eq, conservation_bat_q], axis=0)
            new_b_eq = [df.T_req[i], 0, 0]

        else:
            new_b_eq = [df.T_req[i], 0]
        b_eq += new_b_eq

        new_ub = [df.battery_capacity, 200, 170, df.battery_capacity]
        new_lb = [0, -40, 0, 0]
        ub += new_ub
        lb += new_lb
    b_eq = np.array(b_eq)

    np.savetxt("A_eq.csv", A_eq.astype("int8"), fmt="%d", delimiter=",")
    np.savetxt("B_eq.csv", b_eq.astype("float"), fmt="%d", delimiter=",")
    np.savetxt("lb.csv", lb, fmt="%d", delimiter=",")
    np.savetxt("ub.csv", ub, fmt="%d", delimiter=",")

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
        EV_torque = reshaped_x[:, 1]  # ice_torque for each interval
        IC_torque = reshaped_x[:, 2]  # ev_torque for each interval
        final_charge = reshaped_x[:, 3]  # final_charge for each interval

        # Calculate total torque and battery charge
        total_torque = IC_torque + EV_torque

        if plot:
            # Create a simple plot of the optimization outputs against time
            fig, axs = plt.subplots(2, 1, figsize=(6, 4))
            axs[0].plot(t[:num_intervals], IC_torque, label="IC Torque", color="red")
            axs[0].plot(t[:num_intervals], EV_torque, label="EV Torque", color="green")
            axs[0].plot(
                t[:num_intervals],
                df.T_req[:num_intervals],
                label="Required Torque",
                color="black",
                linestyle="--",
            )
            axs[0].set_xlabel("Time (s)")
            axs[0].set_ylabel("Torque (Nm)")
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
            axs[1].set_ylabel("Charge")
            axs[1].set_title("Optimization Outputs vs Time")
            axs[1].legend()
            axs[1].grid(True)
            plt.show()
    else:
        print("Optimization failed:", res.message)


def main(a: np.ndarray, v: np.ndarray, alpha: np.ndarray, t: np.ndarray):
    df = state_dynamics(a=a, v=v, alpha=alpha, t=t, plot=False)
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
