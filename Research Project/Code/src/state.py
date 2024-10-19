import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from curve_fit import fitted, polynomial_11th_order, exponential


class HEV:
    """
    Dynamic programming controller for a parallel HEV.

    Uses the split power ratio definition for the control variable.

    Output of system should be speed, acceleration, and fuel consumption rate.

    Objective is to minimise total fuel consumption, not instantaneous.

    Missions could be federal urban drive schedule (FUDS) or MVEG-95.

    [1] https://www.cs.cmu.edu/~cga/ok/Guzzella.pdf
    """

    def __init__(self):
        # Parameters of a 2003-2008 Toyota Corolla
        self.m = 1190  # Vehicle mass (kg)
        self.g = 9.81  # Gravity acceleration (m/s^2)
        self.C_r = 0.01  # Typical coefficient of rolling friction value
        self.C_d = 0.30  # Typical drag coefficient value
        self.r = 0.3  # Wheel radius (m)
        self.rho = 1.3  # Density of air (kg/m^3)
        self.A = 2.04  # Frontal area of the car (m^2)
        self.w_max = 439.82  # Engine speed that produces max torque (rad/s) - correspods to 4200 rpm

        # Powertrain parameters
        self.max_ic_power = 198  # Maximum IC engine power (kW)
        self.max_ev_power = 100  # Maximum electric motor power (kW)
        self.max_regen_power = -50  # Maximum regenerative braking power (kW)

        # Motor effiency params
        self.ev_efficiency_params = [
            1.0377836126858233,
            3.3725996647269003,
            -125.3377229830729,
            -31.97439754268986,
            1047.5495467468147,
            -4417.960397005899,
            11881.843447893687,
            -20654.122370089095,
            23269.366841768202,
            -16415.566485143005,
            6592.164239732642,
            -1150.2844062825682,
            0.8395480040347373,
        ]
        self.ev_efficiency_fn = polynomial_11th_order
        self.ice_efficiency_params = [
            4.295619416199058,
            5.592125774538136,
            -418.6144742131811,
            -24.841274370957077,
            3507.7342899247446,
            -15083.988274998506,
            40180.33730251919,
            -69179.7191035756,
            77292.42489485037,
            -54149.84038625687,
            21622.197497866884,
            -3755.3204098868027,
            0.005205429303159347 + 0.05,
        ]
        self.ice_efficiency_fn = polynomial_11th_order

        # Battery parameters
        # self.battery_capacity = 3600 * 1000 * 10  # Battery capacity (Ws)
        self.battery_capacity = 3600
        self.initial_charge = (
            0.4 * self.battery_capacity
        )  # Initial charge (60% of capacity)
        # self.min_charge = 0.2 * self.battery_capacity  # Minimum allowed charge (20% of capacity)
        # self.max_charge = 0.9 * self.battery_capacity  # Maximum allowed charge (90% of capacity)

        # Cost parameters
        self.cost_fuel = 1.5 / 1000  # Cost of fuel ($/g)
        self.cost_energy = 0.15 / (1000 * 3600)  # Cost of battery usage ($/Ws)

    @staticmethod
    def _sgn(x):
        x = x.copy()
        x[x > 0] = 1
        x[x < 0] = -1
        x[x == 0] = 0
        return x

    @staticmethod
    def _a_n(v):
        # self.r = # Wheel radius
        # self.n = # Gear ratio
        # self.a_n = self.n/self.r
        # return 40

        # Piecewise relationship (discontinuous, non-linear)
        a_n = np.zeros_like(v)
        v_kmh = v * 3.6
        a_n[v_kmh <= 20] = (
            40  # Typical value for wheel radius divided by gear ratio when in 1st gear
        )
        a_n[(v_kmh > 20) & (v <= 40)] = (
            25  # Typical value for wheel radius divided by gear ratio when in 2nd gear
        )
        a_n[(v_kmh > 40) & (v <= 60)] = (
            16  # Typical value for wheel radius divided by gear ratio when in 3rd gear
        )
        a_n[(v_kmh > 60) & (v <= 80)] = (
            12  # Typical value for wheel radius divided by gear ratio when in 4th gear
        )
        a_n[v_kmh > 80] = (
            10  # Typical value for wheel radius divided by gear ratio when in 5th gear
        )
        return a_n

    def force_balance(self, a, v, alpha):
        F_r = self.m * self.g * self.C_r * HEV._sgn(v.copy())
        F_a = 1 / 2 * self.rho * self.C_d * self.A * abs(v.copy()) * v.copy()
        F_g = self.m * self.g * np.sin(alpha.copy())
        # F_g = self.m*self.g*dh.copy() # This is incorrect. GPE (mgh) is ENERGY. We want force.
        F_d = F_r + F_a + F_g
        F = self.m * a.copy()
        F_t = F + F_d
        return F_t

    def torque(self, a, v, alpha):
        """Compute total torque required to meet thrust force based on current velocity."""
        # T = self.force_balance(a, v, alpha)/self.r
        T = self.force_balance(a, v, alpha) / self._a_n(v)
        # If the velocity is 0, but the car is on a slope, the required torque will be non-zero
        # This is not accurate, because in such scenarios the driver will use the brake - not the
        # engine - to overcome the disturbance force. Accordingly, the below condition is enforced.
        T[v == 0] = 0
        return T
    
    def power(self, a, v, alpha):
        T = self.torque(a=a,v=v,alpha=alpha)
        w = self._a_n(v)*v
        return T*w

    def max_torque(self, w, motor="ICE"):
        if motor == "ICE":
            return np.ones_like(w) * self.max_ic_torque
            # return np.minimum(self.max_ic_torque, self.max_ic_power/w)
        elif motor == "EV":
            return np.ones_like(w) * self.max_ev_torque
            # return np.minimum(self.max_ev_torque, self.max_ev_power/w)
        elif motor == "Regen":
            return np.ones_like(w) * self.max_regen_torque
            # return np.minimum(self.max_regen_torque, self.max_regen_power/w)
        else:
            raise Exception(f"Error! Unrecognised motor: {motor}")
        
    def max_power(self, w, motor="ICE"):
        if motor == "ICE":
            return np.ones_like(w) * self.max_ic_power
            # return np.minimum(self.max_ic_power, self.max_ic_power/w)
        elif motor == "EV":
            return np.ones_like(w) * self.max_ev_power
            # return np.minimum(self.max_ev_power, self.max_ev_power/w)
        elif motor == "Regen":
            return np.ones_like(w) * self.max_regen_power
            # return np.minimum(self.max_regen_power, self.max_regen_power/w)
        else:
            raise Exception(f"Error! Unrecognised motor: {motor}")

    def w(self, v):
        """Compute angular velocity based on velocity."""
        # w = v.copy()/self.r
        w = self._a_n(v) * v
        assert w.all() >= 0, "Angular velocity must be positive"
        return w

    def efficiency(
        self, w, motor="ICE"
    ):
        w_relative = w / self.w_max
        if motor == "EV":
            efficiency = fitted(
                w_relative, self.ev_efficiency_params, self.ev_efficiency_fn
            )
        elif motor == "ICE":
            efficiency = fitted(
                w_relative, self.ice_efficiency_params, self.ice_efficiency_fn
            )
        else:
            raise Exception(f"Error! Unrecognised motor: {motor}")

        if efficiency.min() <= 0 or efficiency.max() >= 1:
            raise Exception(
                "Error! Efficiency out of valid bounds. This is likely because you passed an unrealistic range of w..."
            )

        return efficiency
