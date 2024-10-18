import numpy as np

class HEV:
    def __init__(self):
        # Vehicle parameters
        self.m = 1500  # Vehicle mass (kg)
        self.g = 9.81  # Gravity acceleration (m/s^2)
        self.C_r = 0.01  # Rolling resistance coefficient
        self.C_d = 0.3  # Drag coefficient
        self.A = 2.5  # Frontal area (m^2)
        self.rho = 1.225  # Air density (kg/m^3)

        # Powertrain parameters
        self.max_ic_torque = 200  # Maximum IC engine torque (Nm)
        self.max_ev_torque = 250  # Maximum electric motor torque (Nm)
        self.max_regen_torque = 50  # Maximum regenerative braking torque (Nm)
        self.ic_efficiency = 0.35  # IC engine efficiency
        self.ev_efficiency = 0.9  # Electric motor efficiency
        self.regen_efficiency = 0.7  # Regenerative braking efficiency

        # Battery parameters
        self.battery_capacity = 3600 * 1000 * 10  # Battery capacity (Ws)
        self.initial_charge = 0.4 * self.battery_capacity  # Initial charge (60% of capacity)
        self.min_charge = 0.2 * self.battery_capacity  # Minimum allowed charge (20% of capacity)
        self.max_charge = 0.9 * self.battery_capacity  # Maximum allowed charge (90% of capacity)

        # Cost parameters
        self.cost_fuel = 1.5 / 1000  # Cost of fuel ($/g)
        self.cost_battery = 0.15 / 3600  # Cost of battery usage ($/Ws)

    def _a_n(self, v):
        # Gear ratio function (simplified)
        return 40 / (1 + 0.1 * v)

    def force_balance(self, a, v, alpha):
        F_r = self.m * self.g * self.C_r * np.sign(v)  # Rolling resistance
        F_a = 0.5 * self.rho * self.C_d * self.A * v**2 * np.sign(v)  # Aerodynamic drag
        F_g = self.m * self.g * np.sin(alpha)  # Gravitational force
        F_i = self.m * a  # Inertial force
        F_t = F_r + F_a + F_g + F_i  # Total force
        return F_t

    def generate_power_req(self, v, F_t):
        return F_t * v

    def generate_eff_curves(self, v, F_t, t):
        w = self._a_n(v) * v  # Angular velocity
        ic_eff = self.ic_efficiency * np.ones_like(w)  # Constant IC efficiency (simplified)
        ev_eff = self.ev_efficiency * np.ones_like(w)  # Constant EV efficiency (simplified)
        return ic_eff, ev_eff

    def EV_torque(self, v):
        w = self._a_n(v) * v  # Angular velocity
        max_torque = self.max_ev_torque * np.ones_like(w)  # Constant max torque (simplified)
        power_per_torque = np.where(v != 0, w, np.zeros_like(w))
        return max_torque, power_per_torque

    def IC_torque(self, v):
        w = self._a_n(v) * v  # Angular velocity
        max_torque = self.max_ic_torque * np.ones_like(w)  # Constant max torque (simplified)
        power_per_torque = np.where(v != 0, w, np.zeros_like(w))
        return max_torque, power_per_torque

    @staticmethod
    def decompose_torque(u, T_w):
        T_m = u * T_w  # EM torque contribution
        T_e = T_w - T_m  # ICE torque contribution
        return T_e, T_m

    @staticmethod
    def decompose_power(u, P_w):
        P_m = u * P_w  # EM power contribution
        P_e = P_w - P_m  # ICE power contribution
        return P_e, P_m

    def IC_power_per_torque(self, v):
        w = self._a_n(v) * v  # Angular velocity
        power_per_torque = w / self.ic_efficiency
        return power_per_torque

    def EV_power_per_torque(self, v):
        w = self._a_n(v) * v  # Angular velocity
        power_per_torque_discharge = w / self.ev_efficiency
        power_per_torque_charge = w * self.regen_efficiency
        return power_per_torque_discharge, power_per_torque_charge

    def regen_power_per_torque(self, v):
        w = self._a_n(v) * v  # Angular velocity
        power_per_torque = w * self.regen_efficiency
        return power_per_torque
