import math
import pandas as pd
from filter import rolling_avg, normalise
import matplotlib.pyplot as plt
from gps import calculate_velocity


class Controller:
    def __init__(self):
        """
        Initialize the controller with a reference to the HEV.
        The control variable u represents the split of torque between the electric motor (EM) and internal combustion engine (ICE).
        """
        self.hev = HEV()
        self.fuel_total = 0  # Total fuel consumption to be minimized

    def f(self, x, u):
        """Move to next state"""
        """This is wrong. We are not controlling velocity, we are controling the ratio of power"""
        # Decompose u into EM and ICE power
        # Use these powers to update the state (velocity)
        # Return the new state
        return x
    
    def cost(self, u, v):
        """
        Cost-to-go function (total fuel consumption over time).
        Input:
            u - control variable, the fraction of torque assigned to the electric motor (EM)
            v - current velocity
        Output:
            The instantaneous fuel consumption for the current state.
        """
        T_w = self.hev.force_balance(v)  # Get total required torque
        T_e, T_m = self.hev.decompose_torque(u, T_w)  # Split the torque between ICE and EM
        print(T_w, T_e, T_m)
        
        # Calculate fuel consumption based on ICE torque and angular speed
        w_e = self.hev._a_n(v) * v  # Get angular velocity of the ICE
        fuel_inst = self.hev.fuel_consumption(T_e, w_e)  # Get fuel consumption for ICE
        
        return fuel_inst
    
    def control(self, v, SoC):
        """
        Dynamic programming control to minimize fuel consumption and manage energy distribution between EM and ICE.
        Inputs:
            v - current velocity
            SoC - state of charge of the battery
        Output:
            u - control variable (EM torque ratio)
        """
        # Get the initial state

        # Determine the control input

        # Update the state


        # Heuristic: prioritize electric motor (u -> 1) when battery SoC is high, minimize ICE usage
        if SoC > 0.5:
            u = 1  # Full electric motor use when battery is sufficiently charged
        elif SoC < 0.1:
            u = 0  # Full ICE use when battery is low
        else:
            u = SoC  # Proportional use of both based on battery charge
        
        return u
    
    def run_simulation(self, velocity_profile):
        """
        Simulate the controller over a velocity profile.
        Input:
            velocity_profile - List of velocities for each time step.
        Output:
            Total fuel consumption over the simulation.
        """
        total_fuel = 0
        for v in velocity_profile:
            SoC = self.hev._SoC(self.hev.s)
            print(SoC)
            u = self.control(v, SoC)  # Compute control variable u based on current velocity and battery SoC
            fuel_inst = self.cost(u, v)  # Calculate instantaneous fuel consumption
            total_fuel += fuel_inst
        
        return total_fuel

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
        self.m = 1190 # Vehicle mass (kg)
        self.g = 9.81 # Gravity acceleration (m/s^2)
        self.C_r = 0.01 # Typical coefficient of rolling friction value
        self.C_d = 0.30 # Typical drag coefficient value
        self.rho = 1.3 # Density of air (kg/m^3)
        self.A = 2.04 # Frontal area of the car (m^2)
        self.T_max = 171 # Max torque of the motor (Nm)
        self.w_max = 439.82 # Engine speed that produces max torque (rad/s) - correspods to 4200 rpm
        self.alpha = math.radians(30) # Slope of road (rad)

        # Simulation parameters
        self.delta_t = 0.01 # Step duration (s)

        # Constraints
        self.s_0 = 9/3.6 # Maximum battery charge 9 MJ = 2.5kWh
        # self.T_e_max = # EM max torque
        # self.T_m_max = # ICE max torque

        # Initial state variables
        self.v = 0 # Initial velocity
        self.s = 9/3.6 # Initial battery charge

    @staticmethod
    def _sgn(x):
        if x>0:
            return 1
        elif x<0:
            return -1
        else:
            return 0
    
    def _a_n(self, v):
        # self.r = # Wheel radius
        # self.n = # Gear ratio
        # self.a_n = self.n/self.r
        return 40
        # v *= 3.6 # m/s to km/h
        # if v<=20:
        #     return 40 # Typical value for wheel radius divided by gear ratio when in 1st gear
        # elif v>20 and v<=40:
        #     return 25 # Typical value for wheel radius divided by gear ratio when in 2nd gear
        # elif v>40 and v<=60:
        #     return 16 # Typical value for wheel radius divided by gear ratio when in 3rd gear
        # elif v>60 and v<=80: 
        #     return 12 # Typical value for wheel radius divided by gear ratio when in 4th gear
        # else:
        #     return 10 # Typical value for wheel radius divided by gear ratio when in 5th gear
        
    def _SoC(self, s):
        return s/self.s_0

    def fuel_consumption(self, T_e, w_e):
        """
        Instantaneous fuel consumption for ICE to generate T_e torque at speed w_e
        https://www.researchgate.net/publication/42251332_Optimal_Energy_Management_in_Hybrid_Electric_Trucks_Using_Route_Information
        """
        P_e = T_e*w_e/1000 # kW
        return 8/140*P_e*self.delta_t # fuel consumption g/s

    def force_balance(self, v):
        w = self._a_n(v)*v # Angular velocity
        beta = 0.4
        T_w = self.T_max*(1-beta*(w/self.w_max-1)**2)
        F_t = self._a_n(v)*T_w
        F_r = self.m*self.g*self.C_r*HEV._sgn(v)
        F_a = 1/2*self.rho*self.C_d*self.A*abs(v)*v
        F_g = self.m*self.g*math.sin(self.alpha)
        F_d = F_r+F_a+F_g
        F = F_t-F_d 
        return F
    
    def generate_power_req(self, v):
        # Compute total torque required
        w = self._a_n(v)*v # Angular velocity
        beta = 0.4
        T_w = self.T_max*(1-beta*(w/self.w_max-1)**2)
        # Convert torque to power
        P_w = w*T_w
        return T_w

    @staticmethod
    def decompose_torque(u, T_w):
        T_m = u*T_w # EM torque contribution
        T_e  = T_w - T_m # ICE torque contribution
        return T_e, T_m
    
    @staticmethod
    def decompose_power(u,P_w):
        P_m = u*P_w # EM power contribution
        P_e  = P_w - P_m # ICE power contribution
        return P_e, P_m

if __name__=="__main__":
    # df = pd.read_csv("/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/bugden_3km_loop.csv")
    # x = df["time"]
    # y = df["ax"]+df["ay"]+df["az"]
    df = pd.read_csv("/Users/timothyalder/Documents/ANU/ENGN4628/Research Project/Code/data/gps_data_1728351441.888126.csv")
    x = df["timestamp"]
    x -= x[0]
    df = calculate_velocity(df)
    y = df["velocity_m_s"]
    y = rolling_avg(y,N=10)
    print(x,y)
    c = HEV()
    p = c.generate_power_req(y)
    plt.plot(x,y,label='Filtered Velocity data')
    plt.plot(x,p,label='Req. Power')
    plt.legend()
    plt.show()
    # total_fuel = c.run_simulation(velocity_profile=y)
    # print(total_fuel)