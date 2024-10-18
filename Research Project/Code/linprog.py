import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Import custom HEV class (assuming it's in a file named new_model.py)
from src.new_model import HEV

def main():
    # Load data
    data_name = 'short_5min'
    alpha = np.load(f'data/fake-slope_{data_name}.npy')
    v = np.load(f'data/fake-velocity_{data_name}.npy')
    a = np.load(f'data/fake-acceleration_{data_name}.npy')
    ts = np.load(f'data/fake-time_{data_name}.npy')

    # Initialize HEV class
    hybrid = HEV()

    # Generate power requirements
    F_t = hybrid.force_balance(a=a, v=v, alpha=alpha)
    Ps = hybrid.generate_power_req(v=v, F_t=F_t)
    Ps = Ps / 1000  # W to kW

    # Generate efficiency curves
    fuel_rate_IC, fuel_rate_EV = hybrid.generate_eff_curves(v=v, F_t=F_t, t=ts)
    IC_eff = 1/fuel_rate_IC
    EV_eff_D = 1/fuel_rate_EV # EV discharging efficiency
    EV_eff_C = 0.3 # EV charging efficiency
    
    # Create a visualization for the sample data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    # Plot velocity
    ax1.plot(ts, v, label='Velocity (m/s)', color='blue')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend(loc='upper left')
    ax1.set_title('Velocity over Time')

    # Plot acceleration
    ax2.plot(ts, a, label='Acceleration (m/s²)', color='red')
    ax2.set_ylabel('Acceleration (m/s²)')
    ax2.legend(loc='upper left')
    ax2.set_title('Acceleration over Time')

    # Plot slope (alpha)
    ax3.plot(ts, alpha, label='Slope (radians)', color='green')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Slope (radians)')
    ax3.legend(loc='upper left')
    ax3.set_title('Slope over Time')

    plt.tight_layout()
    plt.show()


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Plot power and velocity in the first subplot
    ax1.plot(ts, v, label='Velocity (m/s)', color='blue')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(ts, Ps, label='Power (kW)', color='red')
    ax1_twin.set_ylabel('Power (kW)')
    ax1_twin.legend(loc='upper right')

    ax1.set_title('Velocity and Power over Time')

    # Plot fuel rates in the second subplot
    ax2.plot(ts, IC_eff, label='ICE efficiency', color='green')
    ax2.plot(ts, EV_eff_D, label='EV efficiency', color='orange')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Efficiency')
    ax2.legend()
    ax2.set_title('Efficiency Curves over Time')

    plt.tight_layout()

    plt.show()
    # Set parameters
    EV_discharge_lim = 100  # kW
    EV_charge_lim = 50 # kW
    IC_power_lim = 150  # kW
    charge_max = 500 # kWh
    charge_0 = 150 # kWh

    cost_fuel = 1.56/710 # Cost of petrol ($/g)
    # cost_battery = 0.28 # Cost of electricity ($/kWh)
    cost_battery = 0.01 # cost of using battery ($/kWh)
    dt = np.diff(ts)
    dt = dt[0]
    # cost_IC = fuel_rate_IC * cost_fuel
    # cost_EV = fuel_rate_EV * cost_battery

    # Set up linear optimization
    intervals = np.size(ts, 0)

    # Cost function
    # f = np.concatenate((dt*cost_EV, dt*cost_EV, dt*cost_IC))
    f = np.concatenate((dt*cost_battery*np.ones(intervals), dt*cost_battery*np.ones(intervals), dt*cost_fuel*np.ones(intervals)))

    # Inequality conditions
    A = np.concatenate(
        (np.concatenate((dt*EV_eff_C*np.tril(np.ones([intervals,intervals])),-dt*np.tril(np.ones([intervals,intervals])),np.zeros([intervals,intervals])),1), # Charge balance
        np.concatenate((-dt*EV_eff_C*np.tril(np.ones([intervals,intervals])),dt*np.tril(np.ones([intervals,intervals])),np.zeros([intervals,intervals])),1), # Charge balance
        np.concatenate((-1*np.eye(intervals),EV_eff_D*np.eye(intervals),-IC_eff*np.eye(intervals)),1)) # Power balance
        ,0)

    b = np.concatenate((charge_0*3600*np.ones(intervals),(charge_max-charge_0)*3600*np.ones(intervals),-Ps),0)

    Aeq = None

    beq = None

    ub = np.concatenate((np.ones(intervals) * EV_discharge_lim
                     ,np.ones(intervals) * EV_charge_lim
                     , np.ones(intervals) * IC_power_lim)
                    ,0)

    lb = np.zeros(3*intervals)

    res = linprog(f, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=(np.transpose([lb, ub])))

    if res.success:
        # Process results
        results = np.reshape(res.x, (3, -1)).T
        EV = results[:, 0] - results[:, 1]
        IC = results[:, 2]

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(intervals)
        width = 0.8

        ax.bar(x, EV, width, label='EV', color='skyblue')
        ax.bar(x, IC, width, bottom=EV, label='IC', color='orange')
        ax.plot(x, Ps, color='green', linewidth=2, label='Power Required')

        ax.set_xlabel('Interval')
        ax.set_ylabel('Power')
        ax.set_title('Hybrid engine power output')
        ax.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("Optimization failed:", res.message)

if __name__ == "__main__":
    main()
