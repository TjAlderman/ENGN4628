import math
import matplotlib.pyplot as plt
import numpy as np
from filter import normalise


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

    def force_balance(self, a, v, alpha):
        F_r = self.m*self.g*self.C_r*HEV._sgn(v)
        F_a = 1/2*self.rho*self.C_d*self.A*abs(v)*v
        F_g = self.m*self.g*np.sin(alpha)
        F_d = F_r+F_a+F_g
        F = self.m*a
        print(F)
        print(F_d)
        F_t = F+F_d
        return F_t
    
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
    alpha = np.load('Research Project/Code/data/fake-slope.npy')
    v = np.load('Research Project/Code/data/fake-velocity.npy') # VELOCITY GETS ALTERED BY FN
    a = np.load('Research Project/Code/data/fake-acceleration.npy')
    t = np.load('Research Project/Code/data/fake-time.npy')
    c = HEV()
    F_t = c.force_balance(a=a,v=v,alpha=alpha)
    P = c.generate_power_req(v=v,F_t=F_t)
    plt.plot(t,normalise(v),label='Velocity (m/s)')
    plt.plot(t,normalise(P),label='Req. Power (W)')
    plt.plot(t,normalise(F_t),label='Thrust Force (N)',linestyle='--')
    plt.xlabel('Time (seconds)')
    plt.legend()
    plt.show()