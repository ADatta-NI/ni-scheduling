from ray.rllib.algorithms.algorithm import Algorithm
import ray
from environment import SchedulingEnv
from constants import (
    MS_POLICY, OS_POLICY
)





class PolicyEvaluation:
    """Defines a higher level abstraction for evaluating actions for all supported algorithms.
    """
    
    def __init__(self, env_config=None, checkpoint_path=None, random_policy=False) -> None:
        self.env_config = env_config
        self.checkpoint_path = checkpoint_path
        self.random_policy = random_policy

        # Loading the algorithm from checkpoint
        if self.checkpoint_path != None:
            self.algo = Algorithm.from_checkpoint(checkpoint_path)
            #print(self.algo.config)
        # Creating an instance of SchedulingEnv to refer action_space for random policy
        config = {
            "staticConfigurationFilePath": "data/static_configuration.json" if env_config == None else env_config['staticConfigurationFilePath'],
            "maxSteps": None if env_config == None else env_config['maxSteps'],
            "setupRendering": False if env_config == None else env_config['setupRendering'],
            "sampleRandomProblem": False if env_config == None else env_config['sampleRandomProblem']
        }
        self.env = SchedulingEnv(config=config)


    def compute_msagent_action(self, obs) -> list:
        """Feedforwards the ms-agent policy and returns the resulting action
        """
        if self.random_policy == True:
            return self.env.action_space_sample(['MSAgent'])['MSAgent']      
        else:
           return self.algo.compute_single_action(observation=obs, policy_id=MS_POLICY, explore=False)


    def compute_osagent_action(self, obs) -> list:
        """Feedforwards and os-agent policy and returns the resulting action
        """
        if self.random_policy == True:
            return self.env.action_space_sample(['OSAgent-T1'])['OSAgent-T1']
        
        return self.algo.compute_single_action(observation=obs, policy_id=OS_POLICY, explore=False)

    def _normalize_list(self, values):
        """Linearly normalizes a list of values
        """
        total = sum(values)
        if total == 0:
            return values
        return [value / total for value in values]
    