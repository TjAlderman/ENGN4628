class Car:
    def __init__(
        self,
        ice_fuel_consumption: float,
        hybrid_fuel_consumption: float,
        regen_breaking_percentage: float = 0.25,
        solar_charge_percentage: float = 0.5,
    ):
        """
        Limitations
            * I don't think you actually need to charge a hybrid very often. I think regen_breaking should be ~1

        :param ice_fuel_consumption: L/100km
        :type ice_fuel_consumption: float
        :param hybrid_fuel_consumption: L/100km
        :type hybrid_fuel_consumption: float
        :param regen_breaking_percentage: % of energy recovered from regenerative breaking
        :type regen_breaking_percentage: float
        :param solar_charge_percentage: % of battery charged using solar
        :type solar_charge_percentage: float
        """
        self.ice_fc = ice_fuel_consumption
        self.hev_fc = hybrid_fuel_consumption
        pc = 1.57  # Cheapest petrol $/L from 13 Aug - 23 Sep 2024
        # https://www.accc.gov.au/consumers/petrol-and-fuel/petrol-price-cycles-in-major-cities#toc-petrol-prices-in-sydney
        ec = 0.3484  # Most expensive electricty $/kWh from 2020-2021 (SA)
        # https://www.aemc.gov.au/sites/default/files/2021-11/2021_residential_electricity_price_trends_report.pdf
        pd = 0.75  # Petrol density kg/L
        ped = 46.4  # Petrol energy density MJ/kg
        mj_to_kwh = 1 / 3.6
        # https://en.wikipedia.org/wiki/Gasoline
        pe = pd * ped * mj_to_kwh  # Petrol energy density kWh/L

        # Compute $/100km for ICE and HEV counterpart
        self.ice_cost_per_100km = self.ice_fc * pc
        self.hev_cost_per_100km = (
            self.hev_fc * pc
            + (1 - solar_charge_percentage)
            * (1 - regen_breaking_percentage)
            * (self.ice_fc - self.hev_fc)
            * pe
            * ec
        )
        # print(self.ice_cost_per_100km,self.hev_cost_per_100km)
