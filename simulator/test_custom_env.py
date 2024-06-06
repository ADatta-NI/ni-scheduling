from ray.rllib.utils import check_env
from environment import SchedulingEnv

config = {
        "staticConfigurationFilePath": "data/static_configuration.json",
        "maxSteps": 100,
        "setupRendering": False,
        "sampleRandomProblem": True,
    }

# Instantiate the scheudling environment
env = SchedulingEnv(config=config)

check_env(env)