import gymnasium as gym
import math

from constants import (
    MS_AGENT,
)

class LogScalingRewardWrapper():
    """ An reward wrapper that intercepts the rewards build by the scheduling environment
    and applies log to the attributes that are positive.
    
    - Currently, we just apply log to rewards if they are positive.
    - Can do complex agent specific logic or linear scaling etc..
    """

    def reward(self, reward_dict):
        """ Uses agent specific transformers and updates the rewards dict with transformed values.
        """
        for agentName, rew in reward_dict.items():
            if agentName == MS_AGENT:
                reward_dict[agentName] = self._transform_msagent_rew(rew)
            else:
                reward_dict[agentName] = self._transform_osagent_rew(rew)

        return reward_dict
    
    def _transform_msagent_rew(self, rew):
        if rew != 0:
            rew = -self._apply_log(-rew)
        return rew

    def _transform_osagent_rew(self, rew):
        if rew != 0:
            rew = -self._apply_log(-rew)
        return rew
    
    def _apply_log(self, rew):
        return math.log(rew)
