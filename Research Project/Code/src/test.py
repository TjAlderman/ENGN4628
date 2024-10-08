import math
import numpy as np

class HEV:
    def __init__(self):
        self.m = 1190  # Vehicle mass (kg)
        self.g = 9.81  # Gravity acceleration (m/s^2)
        self.C_r = 0.01  # Coefficient of rolling friction
        self.C_d = 0.30  # Drag coefficient
        self.rho = 1.3  # Air density (kg/m^3)
        self.A = 2.04  # Frontal area of the car (m^2)
        self.a_n = 10  # Gear ratio / wheel radius factor
        self.alpha = math.radians(30)  # Road slope
        self.battery_capacity = 5  # Battery capacity (kWh)
        self.fuel_rate = 0.1  # Fuel consumption rate (litres per second) as a basic approximation
    
    def sgn(self, x):
        return 1 if x > 0 else -1 if x < 0 else 0

    def cost(self, u, T_w, soc, v):
        # Cost function: Minimize fuel consumption and penalize battery depletion
        fuel_cost = u * self.fuel_rate  # Approximate fuel cost
        battery_penalty = (1 - u) * max(0, (1 - soc))  # Penalty for using electric power with low charge
        return fuel_cost + battery_penalty

    def state_transition(self, v, soc, u, T_w):
        # Update speed and state of charge based on control input
        F_t = self.a_n * T_w
        F_r = self.m * self.g * self.C_r * self.sgn(v)
        F_a = 0.5 * self.rho * self.C_d * self.A * abs(v) * v
        F_g = self.m * self.g * math.sin(self.alpha)
        F_d = F_r + F_a + F_g
        acceleration = (F_t - F_d) / self.m
        v_next = v + acceleration * self.dt
        
        # Update state of charge (SOC)
        soc_next = soc - (1 - u) * T_w * self.dt / self.battery_capacity  # Approximate battery drain
        
        # Limit SOC between 0 and 1
        soc_next = min(max(soc_next, 0), 1)
        
        return v_next, soc_next

    def dp_controller(self, horizon, initial_v, initial_soc, T_w, N=100):
        # Dynamic programming approach to find the optimal control policy
        
        self.dt = 0.1  # Time step (seconds)
        
        # Initialize state variables
        v = np.zeros((N, horizon + 1))  # Speed over the horizon
        soc = np.zeros((N, horizon + 1))  # State of charge (SOC) over the horizon
        J = np.zeros((N, horizon + 1))  # Cost-to-go function
        u_opt = np.zeros((N, horizon))  # Optimal control at each step
        
        # Set initial conditions
        v[:, 0] = initial_v
        soc[:, 0] = initial_soc
        
        # Discretize the control variable (power split ratio) into N possible values
        u_values = np.linspace(0, 1, N)
        
        # Iterate over the time horizon
        for t in range(horizon - 1, -1, -1):
            for i, u in enumerate(u_values):
                for j in range(N):
                    v_next, soc_next = self.state_transition(v[j, t], soc[j, t], u, T_w)
                    
                    # Calculate cost-to-go
                    current_cost = self.cost(u, T_w, soc[j, t], v[j, t])
                    total_cost = current_cost + J[j, t + 1]
                    
                    # Find the minimum cost and corresponding control
                    if total_cost < J[i, t]:
                        J[i, t] = total_cost
                        u_opt[i, t] = u
        
        return u_opt, J

# Example use case:
hev = HEV()
T_w = 200  # Example torque demand (Nm)
horizon = 50  # Time horizon for optimization
initial_v = 10  # Initial speed (m/s)
initial_soc = 0.8  # Initial state of charge (SOC)

u_opt, J = hev.dp_controller(horizon, initial_v, initial_soc, T_w)
print(u_opt, J)
