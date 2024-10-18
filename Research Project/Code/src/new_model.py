import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

    @staticmethod
    def _sgn(x):
        x = x.copy()
        x[x>0]=1
        x[x<0]=-1
        x[x==0]=0
        return x
    
    def _a_n(self, v):
        # self.r = # Wheel radius
        # self.n = # Gear ratio
        # self.a_n = self.n/self.r
        # return 40

        # Piecewise relationship (discontinuous, non-linear)
        a_n = np.zeros_like(v)
        v_kmh = v*3.6
        a_n[v_kmh<=20]=40 # Typical value for wheel radius divided by gear ratio when in 1st gear
        a_n[(v_kmh>20) & (v<=40)] = 25 # Typical value for wheel radius divided by gear ratio when in 2nd gear
        a_n[(v_kmh>40) & (v<=60)] = 16 # Typical value for wheel radius divided by gear ratio when in 3rd gear
        a_n[(v_kmh>60) & (v<=80)] = 12 # Typical value for wheel radius divided by gear ratio when in 4th gear
        a_n[v_kmh>80] = 10 # Typical value for wheel radius divided by gear ratio when in 5th gear
        return a_n
    
    def generate_power_req(self, v, F_t):
        # Compute total torque required
        w = self._a_n(v)*v # Angular velocity
        T = F_t/self._a_n(v)
        # Convert torque to power
        P = w*T
        return P
    
    def generate_eff_curves(self, v, F_t, t):
        w = self._a_n(v)*v # Angular velocity
        ice_eff = self.ice_efficiency(w)
        ev_eff = self.ev_efficiency(w)
        return ice_eff, ev_eff
        

    def force_balance(self, a, v, alpha):
        F_r = self.m*self.g*self.C_r*HEV._sgn(v)
        F_a = 1/2*self.rho*self.C_d*self.A*abs(v)*v
        F_g = self.m*self.g*np.sin(alpha)
        F_d = F_r+F_a+F_g
        F = self.m*a
        F_t = F+F_d
        return F_t
    
    # @staticmethod
    # def ice_efficiency(P, w, t):
    #     """Returns cost to generate required power based on current rotational velocity"""
    #     c = 1.56/710 # Cost of petrol per gram ($)
    #     b = 1/w # Slower rotational velocity shifts required fuel higher
    #     m_f = 7/120*P+b # Fuel flow rate (g/s)
    #     dt = np.diff(t)
    #     dt = np.append(dt,dt[-1])
    #     f = m_f*dt # Total consumed fuel (g)
    #     return f*c # Cost ($)

    @staticmethod
    def ice_efficiency(omega):
        """Returns cost to generate required power based on current rotational velocity"""
        #b = 1/(1+0.1*omega) # Slower rotational velocity shifts required fuel higher
        #f = 7/120+b # Fuel flow rate (g/s) / kW
        f = 2 * np.ones_like(omega)
        eff = 1/f
        return eff # Relative costs
        
    # @staticmethod
    # def ev_efficiency(P, w, t):
    #     """Returns cost to generate required power based on current rotational velocity"""
    #     m = w/w # Larger values of w give steeper gradient to electric motor efficiency curve
    #     return m*P # Relative cost

    @staticmethod
    def ev_efficiency(omega):
        """Returns cost to generate required power based on current rotational velocity"""
        max_eff = 0.85
        max_omega = 440 # rpm 
        x = omega/(max_omega/0.7)
        a = 2

        y = lambda a : np.exp(a * (x**6-0.7)) - a * x - 1
        y_scaled = lambda a : max_eff * (y(a) - np.min(y(a))) / (np.max(y(a)) - np.min(y(a)))

        eff = np.clip(y_scaled(a),0.1,1)
        return eff # Relative costs
    
    def EV_torque(self, v):
        """
        Calculate maximum torque and power per unit torque for the electric motor.
        
        Args:
        v (float): Vehicle velocity in m/s
        
        Returns:
        tuple: (max_torque, power_per_torque)
            max_torque (float): Maximum torque that can be provided by the electric motor in Nm
            power_per_torque (float): Power required per unit of torque in W/Nm
        """
        # Convert velocity to angular velocity (assuming direct drive)
        omega = v / (self.wheel_radius * 2 * np.pi) * 60  # Convert to RPM

        # Define EV motor characteristics (these should be adjusted based on actual motor specs)
        max_power = 100000  # Maximum power in Watts
        base_speed = 1500  # Base speed in RPM
        max_speed = 6000  # Maximum speed in RPM

        if omega <= base_speed:
            max_torque = max_power / (base_speed * 2 * np.pi / 60)
        else:
            max_torque = max_power / (omega * 2 * np.pi / 60)

        # Assuming a linear relationship between torque and power
        power_per_torque = omega * 2 * np.pi / 60  # W/Nm

        return max_torque, power_per_torque

    def IC_torque(self, v):
        """
        Calculate maximum torque and power per unit torque for the internal combustion engine.
        
        Args:
        v (float): Vehicle velocity in m/s
        
        Returns:
        tuple: (max_torque, power_per_torque)
            max_torque (float): Maximum torque that can be provided by the IC engine in Nm
            power_per_torque (float): Power required per unit of torque in W/Nm
        """
        # Use _a_n function to determine the gear ratio
        gear_ratio = self._a_n(v)
        
        # Convert velocity to angular velocity using the determined gear ratio
        omega = v / (self.wheel_radius * 2 * np.pi) * 60 * gear_ratio  # Convert to RPM

        # Define IC engine characteristics (these should be adjusted based on actual engine specs)
        max_power = 150000  # Maximum power in Watts
        torque_peak = 300  # Peak torque in Nm
        rpm_peak = 4000  # RPM at peak torque
        max_rpm = 6000  # Maximum engine RPM

        if omega <= rpm_peak:
            max_torque = torque_peak
        else:
            max_torque = max_power / (omega * 2 * np.pi / 60)

        # Assuming a linear relationship between torque and power
        power_per_torque = omega * 2 * np.pi / 60  # W/Nm

        return max_torque, power_per_torque

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
    alpha = np.load('Research Project/Code/data/fake-slope.npy')
    v = np.load('Research Project/Code/data/fake-velocity.npy') # VELOCITY GETS ALTERED BY FN
    a = np.load('Research Project/Code/data/fake-acceleration.npy')
    t = np.load('Research Project/Code/data/fake-time.npy')
    df = pd.read_csv("trip1.csv")
    v = df['v']
    a = df['a']
    t = df['t']
    alpha = df['slope']
    c = HEV()
    F_t = c.force_balance(a=a,v=v,alpha=alpha)
    P = c.generate_power_req(v=v,F_t=F_t)
    plt.plot(t,v,label='Velocity (m/s)')
    plt.plot(t,P/1000,label='Req. Power (kW)')
    plt.plot(t,F_t/1000,label='Thrust Force (kN)',linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.show()