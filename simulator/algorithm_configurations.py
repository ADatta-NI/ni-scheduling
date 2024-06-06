from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.policy.policy import PolicySpec
import json

from environment import SchedulingEnv
from constants import (
    MS_AGENT, OS_AGENT_PREFIX, MS_POLICY, OS_POLICY, CHECKPOINT_ROOT, RAY_RESULTS_ROOT
)
# from ray.tune.registry import register_env
# from log_scaling_obs_wrapper import LogScalingObservationWrapper
# from log_scaling_reward_wrapper import LogScalingRewardWrapper

from custom_models import FullyConnectedSoftmaxNetwork
# Environment Settings
render = False
maxSteps = None

## Define the custom model for softmax activation at the end 
# Parses the scheduling problem in `config` json file and returns corresponding dict object.
def _get_static_config_data(config) -> dict:
    staticConfigurationFilePath = config.get("staticConfigurationFilePath")
    with open(staticConfigurationFilePath) as staticConfigurationFile:
        data = json.load(staticConfigurationFile)
    return data


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
env_config = {
        "staticConfigurationFilePath": "data/static_configuration.json",
        "maxSteps": maxSteps,
        "setupRendering": render,
        "sampleRandomProblem": True,
    }
env = SchedulingEnv(config=env_config)
data = _get_static_config_data(env_config)
randomTesterName = list(data['testers']['items'])[0]


# Policies dictionary
policies = {
    MS_POLICY: PolicySpec(
        policy_class=None, 
        observation_space=env.observation_space[MS_AGENT], 
        action_space=env.action_space[MS_AGENT], 
        config={} # Overrides defined keys in the main Algorithm config
    ),
    OS_POLICY: PolicySpec(
        policy_class=None, 
        observation_space=env.observation_space[OS_AGENT_PREFIX + randomTesterName], 
        action_space=env.action_space[OS_AGENT_PREFIX + randomTesterName], 
        config={} # Overrides defined keys in the main Algorithm config
    )
}


# PPO Algorithm Config
ppo_config = PPOConfig(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
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
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None
            ).training(
                gamma=0.99,
                use_kl_loss= True,
                kl_coeff=0.2,
                use_critic = True,
                use_gae = True,
                kl_target = 1e-3,
                sgd_minibatch_size = 128,
                num_sgd_iter = 30,
                shuffle_sequences = True,
                vf_loss_coeff = 1.0,
                entropy_coeff = 0.01,
                entropy_coeff_schedule = None,
                clip_param = 0.3,
                vf_clip_param = 10.0,
                grad_clip = None,
                model={
                    "free_log_std" : True,
                    "use_lstm" : False,
                    "max_seq_len" : 50,
                    "custom_model" : "custom_softmax_model",
                    "fcnet_hiddens": [64,64],
                    },
                #exploration_config = {
                # The Exploration class to use. In the simplest case, this is the name
                # (str) of any class present in the `rllib.utils.exploration` package.
                # You can also provide the python class directly or the full location
                # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
                # EpsilonGreedy").
                #"type": "StochasticSampling",
                # Add constructor kwargs here (if any).
                #}
               
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'
            )


# PPO Algorithm Config
appo_config = APPOConfig(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
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
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None
            ).training(
                vtrace = True,
                gamma=0.99,
                use_kl_loss= True,
                kl_coeff=0.2,
                use_critic = True,
                use_gae = True,
                kl_target = 1e-3,
                #sgd_minibatch_size = 128,
                num_sgd_iter = 30,
                #shuffle_sequences = True,
                vf_loss_coeff = 1.0,
                entropy_coeff = 0.01,
                entropy_coeff_schedule = None,
                clip_param = 0.3,
                #vf_clip_param = 10.0,
                grad_clip = None,
                model={
                    "fcnet_hiddens": [64,64]},
                #exploration_config = {
                # The Exploration class to use. In the simplest case, this is the name
                # (str) of any class present in the `rllib.utils.exploration` package.
                # You can also provide the python class directly or the full location
                # of your class (e.g. "ray.rllib.utils.exploration.epsilon_greedy.
                # EpsilonGreedy").
                #"type": "StochasticSampling",
                # Add constructor kwargs here (if any).
                #}
               
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'
            )


# A3C Algorithm Config
a3c_config = A3CConfig(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
            ).framework(
                framework='torch',
            ).environment(
                env=SchedulingEnv,
                env_config=env_config,
                render_env=False,
            ).rollouts(
                num_rollout_workers=64,
                num_envs_per_worker=1,
                validate_workers_after_construction=True,
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None
            ).training(
                gamma=0.99,
                lr=0.00001,
                sample_async = False,
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'
            )


# SAC Algorithm Config
sac_config = SACConfig(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
            ).framework(
                framework='torch',   
            ).environment(
                env=SchedulingEnv,
                env_config=env_config,
                render_env=False,
            ).rollouts(
                num_rollout_workers=64,
                num_envs_per_worker=1,
                validate_workers_after_construction=True,
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None   
            ).training(
                gamma=0.99,   
                train_batch_size=2048
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'   
            )

ddpg_config = DDPGConfig(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
            ).framework(
                framework='torch',   
            ).environment(
                env=SchedulingEnv,
                env_config=env_config,
                render_env=False,
            ).rollouts(
                num_rollout_workers=64,
                num_envs_per_worker=1,
                validate_workers_after_construction=True,
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None   
            ).training(
                gamma=0.99,   
                train_batch_size=2048
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'   
            )

td3_config = TD3Config(
            ).resources(
                num_gpus=1,
                num_learner_workers=1,
                num_gpus_per_learner_worker=1,
            ).framework(
                framework='torch',   
            ).environment(
                env=SchedulingEnv,
                env_config=env_config,
                render_env=False,
            ).rollouts(
                num_rollout_workers=64,
                num_envs_per_worker=1,
                validate_workers_after_construction=True,
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None   
            ).training(
                gamma=0.99,   
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'   
            )
