from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
#from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig
#from ray.rllib.algorithms.td3 import TD3Config
#from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.core.rl_module import RLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.tune.registry import register_env
import json
import torch
import ray
from custom_models import FullyConnectedSoftmaxNetwork
from custom_action_dist import TorchDirichlet
from environment import SchedulingEnv
from constants import (
    MS_AGENT, OS_AGENT_PREFIX, MS_POLICY, OS_POLICY, CHECKPOINT_ROOT, RAY_RESULTS_ROOT
)
# from ray.tune.registry import register_env
# from log_scaling_obs_wrapper import LogScalingObservationWrapper
# from log_scaling_reward_wrapper import LogScalingRewardWrapper
from ray.rllib.models import ModelCatalog
from custom_models import FullyConnectedSoftmaxNetwork

# Environment Settings
render = False
maxSteps = None

ModelCatalog.register_custom_model("custom_softmax_model", FullyConnectedSoftmaxNetwork)

print("Model registered successfully")

ModelCatalog.register_custom_action_dist("dirichlet_dist", TorchDirichlet)

print("Distribution registered successfully")


## Define the custom model for softmax activation at the end
# Parses the scheduling problem in `config` json file and returns corresponding dict object.
def _get_static_config_data(config) -> dict:
    staticConfigurationFilePath = config.get("staticConfigurationFilePath")
    with open(staticConfigurationFilePath) as staticConfigurationFile:
        data = json.load(staticConfigurationFile)
    return data


# Env creator string

env_config = {
    "staticConfigurationFilePath": "data/static_configuration.json",
    "maxSteps": maxSteps,
    "setupRendering": render,
    "sampleRandomProblem": True,
}

register_env("scheduling_env", lambda _: SchedulingEnv(config=env_config))


# Policy Mapping Function
def policy_mapper(agent_id, episode, worker):
    if agent_id == MS_AGENT:
        return MS_POLICY
    else:
        return OS_POLICY


# Creates environment by wrapping with all necessary wrappers
# def env_creator(env_config):
#     return LogScalingObservationWrapper(LogScalingRewardWrapper(SchedulingEnv(config=env_config)))

# Register the custom wrapped environment
# register_env("WrappedSchedulingEnv", env_creator)


# Create SchedulingEnv instance for accessing observation and action space structures

env = SchedulingEnv(config=env_config)
data = _get_static_config_data(env_config)
randomTesterName = list(data['testers']['items'])[0]

# Policies dictionary with proper variations in PPOConfig
# pass the custom models and distributions to the agents
policies = {
    MS_POLICY: PolicySpec(
        policy_class=None,
        observation_space=env.observation_space[MS_AGENT],
        action_space=env.action_space[MS_AGENT],
        config=SACConfig.overrides(

            policy_model_config={
                "fcnet_hiddens": [64, 64],
                #"custom_model": "custom_softmax_model",
                #"custom_action_dist": "dirichlet_dist",
                "free_log_std": False,
            },
            q_model_config={
                "fcnet_hiddens": [64, 64]
            },

            explore=True),

    ),
    OS_POLICY: PolicySpec(
        policy_class=None,
        observation_space=env.observation_space[OS_AGENT_PREFIX + randomTesterName],
        action_space=env.action_space[OS_AGENT_PREFIX + randomTesterName],
        config=SACConfig.overrides(
            policy_model_config={
                "fcnet_hiddens": [64, 64],
                #"custom_model": "custom_softmax_model",
                #"custom_action_dist": "dirichlet_dist",
                "free_log_std": False,
            },
            q_model_config={
                "fcnet_hiddens": [64, 64]
            },
            explore=True),

    )
}
'''
model={
                    "fcnet_hiddens": [64,64],
                    "custom_model" : "custom_softmax_model",
                    "custom_action_dist": "dirichlet_dist",
                    "free_log_std" : False,
                    "use_lstm" : False,
                    "max_seq_len" : 50,
                    },
'''

# PPO Algorithm Config
multi_agent_config = SACConfig(
).resources(
    num_gpus=1,
    num_learner_workers=1,
    num_gpus_per_learner_worker=1,
    num_cpus_per_learner_worker=1,
    num_cpus_for_local_worker=1,

).framework(
    framework='torch',
).environment(
    env=SchedulingEnv,
    env_config=env_config,
    render_env=False,
).rollouts(
    num_rollout_workers=62,
    num_envs_per_worker=1,
    validate_workers_after_construction=True,
    # observation_filter='MeanStdFilter',
    preprocessor_pref=None,  # 'rllib' | 'deepmind' | None
).training(
    twin_q=True,
    gamma=0.99,
    tau=0.99,
    initial_alpha=1.0,
    train_batch_size=64,
).multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapper,
    policies_to_train=list(policies.keys()),
    count_steps_by='agent_steps',  # 'agent_steps' | 'env_steps'
)

algo = multi_agent_config.build()
ms_policy = algo.get_policy(policy_id=MS_POLICY)
dist = algo.get_policy(policy_id=MS_POLICY).dist_class
obs_dict, _ = env.reset()
ms_obs = obs_dict[MS_AGENT]
ms_obs = torch.Tensor(ms_obs)
ms_obs = ms_obs.t()
actions = ms_policy.compute_single_action(ms_obs)
print(actions)
num_of_iterations = 500

print("===========TRAINING===========")
for i in range(num_of_iterations):
    print(i)
    results = algo.train()

    if MS_POLICY in results["policy_reward_mean"] and OS_POLICY in results['policy_reward_mean']:
        print(
            f"Iteration={algo.iteration}: R1={results['policy_reward_mean'][MS_POLICY]} R2={results['policy_reward_mean'][OS_POLICY]}")

    if (algo.iteration) % 500 == 0:
        print("Completed 5 more iters")
        filename = algo.save(
            checkpoint_dir='/data/adatta14/PycharmProjects/ni-scheduling-project' + "checkpoint_sac_tardiness_" + str(
                algo.iteration) + "/")
        print(f"checkpoint saved after iteration: {algo.iteration} at {filename}")
print("=============DONE=============")
ray.shutdown()
