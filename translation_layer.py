import numpy as np

class TranslationLayer:
    """
    This class is responsible for translating standardized actions from the agents
    into the specific action format required by the CityLearn environment for each building.
    """
    def __init__(self, buildings):
        """
        Initializes the TranslationLayer with the metadata of the buildings' action spaces.
        """
        self.action_metadata = [b.action_metadata for b in buildings]

    def translate_actions(self, standard_actions):
        """
        Translates a list of standardized actions into environment-compatible actions.

        Args:
            standard_actions (list of np.array): A list where each element is a
                standardized 3-element action array from an agent.

        Returns:
            list of np.array: A list of action arrays that can be passed to env.step().
        """
        translated_actions = []
        for i, standard_action in enumerate(standard_actions):
            building_action = []
            metadata = self.action_metadata[i]
            
            if metadata['cooling_storage']:
                building_action.append(standard_action[0])
            if metadata['dhw_storage']:
                building_action.append(standard_action[1])
            if metadata['electrical_storage']:
                building_action.append(standard_action[2])
            
            translated_actions.append(np.array(building_action))
            
        return translated_actions
