import gymnasium as gym
import math

from constants import (
    MS_AGENT,
)

class LogScalingObservationWrapper():
    """ An observation wrapper that intercepts the observations build by the scheduling environment
    and applies log to the attributes that are positive.
    
    - Currently, we just apply log to individual observation components if they are positive.
    - Can do complex agent based or observation component specific logic
    """
    
    def observation(self, observations_dict):
        """ Uses agent specific transformers and updates the observations dict with transformed values.
        """
        # print("Type1: ", type(observations_dict), observations_dict)
        for agentName, obs in observations_dict.items():
            if agentName == MS_AGENT:
                observations_dict[agentName] = self._transform_msagent_obs(obs)
            else:
                observations_dict[agentName] = self._transform_osagent_obs(obs)
        # print("Type2: ", type(observations_dict), observations_dict)
        return observations_dict
    
    def _transform_msagent_obs(self, obs):
        for i, _obs in enumerate(obs):
            if _obs > 0:
                obs[i] = self._apply_log(_obs)
        return obs

    def _transform_osagent_obs(self, obs):
        for i, _obs in enumerate(obs):
            if _obs > 0:
                obs[i] = self._apply_log(_obs)
        return obs
    
    def _apply_log(self, obs):
        return math.log(obs)
