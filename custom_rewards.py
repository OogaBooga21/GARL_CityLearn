# custom_rewards.py
from citylearn.reward_function import RewardFunction
from typing import List, Mapping, Union

class GridConsumptionReward(RewardFunction):
    """
    A custom reward function that penalizes grid consumption.
    The reward for each building is the negative of its net electricity consumption.
    """
    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        """
        Calculates the reward for each building.
        """
        net_electricity_consumption = [o['net_electricity_consumption'] for o in observations]
        reward = [-v for v in net_electricity_consumption]

        if self.central_agent:
            return [sum(reward)]
        else:
            return reward

