# Note: OpenAI 0-1 mini generated code
class HybridPowertrain:
    def __init__(self, ic_efficiency_curve, em_efficiency_curve, battery_capacity):
        """
        Initialize the hybrid powertrain model with non-linear efficiency curves.

        Args:
            ic_efficiency_curve (function): Function to calculate IC engine efficiency based on power and velocity.
            em_efficiency_curve (function): Function to calculate EM efficiency based on power and velocity.
            battery_capacity (float): Total battery capacity in kWh.
        """
        self.ic_efficiency_curve = ic_efficiency_curve
        self.em_efficiency_curve = em_efficiency_curve
        self.battery_capacity = battery_capacity
        self.battery_state = battery_capacity  # Initialize with full charge

    def compute_fuel_consumption(self, torque_ic, angular_velocity, velocity):
        """
        Calculate fuel consumption based on IC torque, angular velocity, and vehicle velocity with non-linear efficiency.

        Args:
            torque_ic (float): Torque provided by the internal combustion engine (Nm).
            angular_velocity (float): Angular velocity of the engine in rad/s.
            velocity (float): Vehicle speed in m/s.

        Returns:
            float: Fuel consumption in liters per hour.
        """
        power_ic = torque_ic * angular_velocity  # Power from IC engine in Watts
        efficiency_ic = self.ic_efficiency_curve(power_ic, velocity)  # Non-linear efficiency
        if efficiency_ic <= 0:
            efficiency_ic = 0.01  # Prevent division by zero or negative efficiency
        fuel_consumption = power_ic / (efficiency_ic * 3600)  # Convert W to kW and divide by efficiency
        return fuel_consumption

    def compute_electric_energy_consumption(self, torque_em, angular_velocity, time_step, velocity):
        """
        Calculate electric energy consumption based on EM torque, angular velocity, and vehicle velocity with non-linear efficiency.

        Args:
            torque_em (float): Torque provided by the electric motor (Nm).
            angular_velocity (float): Angular velocity of the motor in rad/s.
            time_step (float): Time step in hours.
            velocity (float): Vehicle speed in m/s.

        Returns:
            float: Energy consumption in kWh.
        """
        power_em = torque_em * angular_velocity  # Power from EM in Watts
        efficiency_em = self.em_efficiency_curve(power_em, velocity)  # Non-linear efficiency
        if efficiency_em <= 0:
            efficiency_em = 0.01  # Prevent division by zero or negative efficiency
        energy_consumption = (power_em / (efficiency_em * 1000)) * time_step  # Convert W to kW and multiply by time_step
        self.battery_state -= energy_consumption
        self.battery_state = max(self.battery_state, 0)  # Prevent negative state
        return energy_consumption


class ControlSystem:
    def __init__(self, powertrain, road_profile, velocity_profile, time_step, lambda_ecms):
        """
        Initialize the control system with ECMS parameters.

        Args:
            powertrain (HybridPowertrain): The hybrid powertrain model.
            road_profile (list of float): Road gradient angles for each timestep in degrees.
            velocity_profile (list of float): Vehicle speeds for each timestep in m/s.
            time_step (float): Duration of each timestep in hours.
            lambda_ecms (float): ECMS weighting factor for energy consumption.
        """
        self.powertrain = powertrain
        self.road_profile = road_profile
        self.velocity_profile = velocity_profile
        self.time_step = time_step
        self.lambda_ecms = lambda_ecms
        self.torque_split_history = []
        self.soc_history = [powertrain.battery_state]

    def optimize_torque_split(self, gradient, velocity, soc):
        """
        Optimize the torque split between IC engine and electric motor using ECMS.

        Args:
            gradient (float): Road gradient in degrees.
            velocity (float): Vehicle speed in m/s.
            soc (float): State of charge of the battery in kWh.

        Returns:
            tuple: (optimal_torque_ic, optimal_torque_em)
        """
        # Define possible torque allocations (discrete steps for simplicity)
        torque_options = [(torque, 120 - torque) for torque in range(0, 121, 10)]  # Total torque fixed at 120 Nm

        min_cost = float('inf')
        optimal_split = (0, 120)

        angular_velocity = velocity / 0.5  # Example conversion to rad/s

        for torque_ic, torque_em in torque_options:
            # Compute fuel consumption
            fuel = self.powertrain.compute_fuel_consumption(torque_ic, angular_velocity, velocity)

            # Compute energy consumption
            energy = self.powertrain.compute_electric_energy_consumption(torque_em, angular_velocity, self.time_step, velocity)

            # Equivalent consumption
            cost = fuel + self.lambda_ecms * energy

            if cost < min_cost and soc >= energy:
                min_cost = cost
                optimal_split = (torque_ic, torque_em)

        return optimal_split

    def run_optimization(self):
        """
        Run the torque split optimization using ECMS over the entire route.
        """
        total_fuel = 0
        total_energy = 0
        for i in range(len(self.road_profile)):
            gradient = self.road_profile[i]
            velocity = self.velocity_profile[i]
            soc = self.powertrain.battery_state

            torque_ic, torque_em = self.optimize_torque_split(gradient, velocity, soc)
            self.torque_split_history.append((torque_ic, torque_em))

            angular_velocity = velocity / 0.5  # Example conversion to rad/s
            fuel = self.powertrain.compute_fuel_consumption(torque_ic, angular_velocity, velocity)
            energy = self.powertrain.compute_electric_energy_consumption(torque_em, angular_velocity, self.time_step, velocity)

            total_fuel += fuel
            total_energy += energy
            self.soc_history.append(self.powertrain.battery_state)

            print(f"Step {i}: Fuel={fuel:.4f} L, Energy={energy:.4f} kWh, SoC={self.powertrain.battery_state:.2f} kWh")

        print(f"Total Fuel Consumption: {total_fuel:.4f} L")
        print(f"Total Electric Energy Consumed: {total_energy:.4f} kWh")


# Example non-linear efficiency curves using ECMS considerations
def ic_efficiency_curve(power, velocity):
    """
    Example non-linear efficiency curve for the internal combustion engine based on power and velocity.

    Args:
        power (float): Power output in Watts.
        velocity (float): Vehicle speed in m/s.

    Returns:
        float: Efficiency as a decimal.
    """
    # Example: Efficiency peaks at certain power and decreases otherwise
    peak_power = 50000  # 50 kW
    efficiency = 0.35 - 0.000002 * (power - peak_power) ** 2 + 0.00001 * velocity
    return max(efficiency, 0.1)  # Ensure efficiency doesn't go below 10%

def em_efficiency_curve(power, velocity):
    """
    Example non-linear efficiency curve for the electric motor based on power and velocity.

    Args:
        power (float): Power output in Watts.
        velocity (float): Vehicle speed in m/s.

    Returns:
        float: Efficiency as a decimal.
    """
    # Example: Efficiency varies with power and slightly with velocity
    if power < 10000:
        return 0.85
    elif power < 50000:
        return 0.9 - 0.000001 * (power - 30000) ** 2 + 0.000005 * velocity
    else:
        return 0.8  # Efficiency drops at very high power


# Example usage:
# road_gradients = [0, 5, 10, -5, -10, 0]
# vehicle_speeds = [20, 25, 30, 35, 25, 20]
# time_step_hours = 1/3600  # 1 second time step
# lambda_ecms = 0.05  # Example lambda value
# powertrain = HybridPowertrain(ic_efficiency_curve, em_efficiency_curve, battery_capacity=50)
# control_system = ControlSystem(powertrain, road_gradients, vehicle_speeds, time_step_hours, lambda_ecms)
# control_system.run_optimization()