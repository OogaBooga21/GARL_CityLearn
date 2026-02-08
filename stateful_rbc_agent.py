import numpy as np

class StatefulRBC:
    """
    A stateful Rule-Based Controller that uses a standardized action space.
    It charges the battery until it is full, then discharges it until it is empty.
    """
    def __init__(self, action_space, active_observations):
        self.action_space = action_space
        self.charging_mode = True  # Start in charging mode
        self.soc_index = None
        if 'electrical_storage_soc' in active_observations:
            self.soc_index = active_observations.index('electrical_storage_soc')

    def predict(self, observations):
        """
        Returns a standardized 3-element action vector [cooling, dhw, electrical].
        The logic is to charge/discharge the battery based on its state of charge.
        """
        # Standard action: [cooling, dhw, electrical]
        action = np.zeros(3)

        if self.soc_index is not None:
            soc = observations[self.soc_index]
            if self.charging_mode:
                if soc < 0.99:
                    # Continue charging
                    action[2] = 1.0
                else:
                    # Switch to discharging mode
                    self.charging_mode = False
                    action[2] = -1.0
            else:  # Discharging mode
                if soc > 0.01:
                    # Continue discharging
                    action[2] = -1.0
                else:
                    # Switch to charging mode
                    self.charging_mode = True
                    action[2] = 1.0
        
        return action



