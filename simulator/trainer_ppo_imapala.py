from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
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
        config=PPOConfig.overrides(
            rollout_fragment_length = 16,
            model={
                "fcnet_hiddens": [64, 64],
                "custom_model": "custom_softmax_model",
                "custom_action_dist": "dirichlet_dist",
                "free_log_std": False,
            },
            explore=True),

    ),
    OS_POLICY: PolicySpec(
        policy_class=None,
        observation_space=env.observation_space[OS_AGENT_PREFIX + randomTesterName],
        action_space=env.action_space[OS_AGENT_PREFIX + randomTesterName],
        config=ImpalaConfig.overrides(
            rollout_fragment_length = 16,
            model={
                "fcnet_hiddens": [64, 64],
                "custom_model": "custom_softmax_model",
                "custom_action_dist": "dirichlet_dist",
                "free_log_std": False,
            },
            explore=True),

    )
}

# PPO Algorithm Config
ppo_agent_config = PPOConfig(
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
    num_rollout_workers=20,
    num_envs_per_worker=1,
    validate_workers_after_construction=True,
    # observation_filter='MeanStdFilter',
    preprocessor_pref=None,  # 'rllib' | 'deepmind' | None
).training(
    gamma=0.99,
    use_kl_loss=True,
    kl_coeff=0.01,
    sgd_minibatch_size=16,
    train_batch_size=64,
).multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapper,
    policies_to_train=[MS_POLICY],
    count_steps_by='agent_steps',  # 'agent_steps' | 'env_steps'
)

impala_agent_config = ImpalaConfig(
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
    num_rollout_workers=20,
    num_envs_per_worker=1,
    validate_workers_after_construction=True,
    # observation_filter='MeanStdFilter',
    preprocessor_pref=None,  # 'rllib' | 'deepmind' | None
).training(
    gamma=0.99,
    lr = 0.0001,
    minibatch_size='auto',
    train_batch_size=64,
    # model={
    #     "fcnet_hiddens": [256, 256, 256],
    # },
    grad_clip=10,
).multi_agent(
    policies=policies,
    policy_mapping_fn=policy_mapper,
    policies_to_train=[OS_POLICY],
    count_steps_by='agent_steps',  # 'agent_steps' | 'env_steps'
)

algo_ppo = ppo_agent_config.build()
algo_impala = impala_agent_config.build()
ms_policy = algo_ppo.get_policy(policy_id=MS_POLICY)

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
    results_ppo = algo_ppo.train()

    results_impala = algo_impala.train()

    if MS_POLICY in results_ppo["policy_reward_mean"] and OS_POLICY in results_impala['policy_reward_mean']:
        print(
            f"Iteration={algo_ppo.iteration}: R1={results_ppo['policy_reward_mean'][MS_POLICY]} R2={results_impala['policy_reward_mean'][OS_POLICY]}")

    if algo_ppo.iteration % 500 == 0:
        # print("Completed 5 more iters")
        filename_ppo = algo_ppo.save(
            checkpoint_dir='/data/adatta14/PycharmProjects/ni-scheduling-project' + "checkpoint_median_ppo" + str(
                algo_ppo.iteration) + "/")

        filename_sac = algo_impala.save(
            checkpoint_dir='/data/adatta14/PycharmProjects/ni-scheduling-project' + "checkpoint_median_impala" + str(
                algo_impala.iteration) + "/")
        # print(f"checkpoint saved after iteration: {algo_impala.iteration} at {filename}")
print("=============DONE=============")
ray.shutdown()
