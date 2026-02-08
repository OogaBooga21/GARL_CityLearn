
import numpy as np

class SimpleRBC:
    """
    A simple Rule-Based Controller that uses a standardized action space.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observations):
        """
        Returns a standardized 3-element action vector [cooling, dhw, electrical].
        The logic is to charge/discharge the battery based on the time of day.
        """
        hour = observations[2]
        
        # Standard action: [cooling, dhw, electrical]
        action = np.zeros(3)
        
        if 7 <= hour < 16:
            # Charge the battery
            action[2] = 1.0
        elif 16 <= hour < 20:
            # Discharge the battery
            action[2] = -1.0
        
        return action

