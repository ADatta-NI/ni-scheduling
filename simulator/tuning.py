import random
import logging
from pprint import pformat
import argparse
import os
from os.path import expanduser
import json

import ray
from ray import air, tune
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.rllib.policy.policy import PolicySpec

from environment import SchedulingEnv
from callback import SchedulingCallback
from constants import (
    MS_AGENT, OS_AGENT_PREFIX, MS_POLICY, OS_POLICY, CHECKPOINT_ROOT, RAY_RESULTS_ROOT
)

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("tune_framework")


# Environment Settings
shouldRender = False
maxSteps = None

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


def run(smoke_test=False, storage_path: str = expanduser("~") + '/tune/'):

    stop = {"training_iteration": 1 if smoke_test else 10} 

    env_config = {
        "staticConfigurationFilePath": "data/static_configuration.json",
        "maxSteps": maxSteps,
        "setupRendering": shouldRender,
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

    config = DDPGConfig(
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
                num_rollout_workers=10,
                num_envs_per_worker=1,
                validate_workers_after_construction=True,
                preprocessor_pref=None, # 'rllib' | 'deepmind' | None   
            ).training(
                # gamma=tune.uniform(0.95, 1.0),   
                # train_batch_size=tune.grid_search([12a8, 256, 512])
                gamma=tune.grid_search([0.95,0.96,0.97,0.98,0.99])
            ).multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapper,
                policies_to_train=list(policies.keys()),
                count_steps_by='agent_steps', # 'agent_steps' | 'env_steps'   
            ).callbacks(
                SchedulingCallback
            )


    logger.info("Configuration: \n %s", pformat(config))

    scheduler = ASHAScheduler()

    return tune.Tuner(
        "DDPG",
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=1,
            progress_reporter=CLIReporter(
                metric_columns={
                    "episode_reward_mean": "reward_mean",
                    "policy_reward_mean/ms-policy": "mspolicy_reward_mean",
                    "info/learner/ms-policy/learner_stats/actor_loss": "mspolicy_actor_loss",
                    "info/learner/ms-policy/learner_stats/critic_loss": "mspolicy_critic_loss",
                    "policy_reward_mean/os-policy": "ospolicy_reward_mean",
                    "info/learner/os-policy/learner_stats/actor_loss": "ospolicy_actor_loss",
                    "info/learner/os-policy/learner_stats/critic_loss": "ospolicy_critic_loss",
                    "custom_metrics/static_configuration_1_mean_reward_mean": "sc_1_mean_reward",
                    "custom_metrics/static_configuration_1_tardiness_mean": "sc_1_mean_tardiness",
                },
                sort_by_metric=True,
                max_report_frequency=30,
            ),
            storage_path=storage_path,
        ),
        tune_config=tune.TuneConfig(
            num_samples=-1,
            metric="episode_reward_mean",
            mode="max",
            scheduler=scheduler
        ),
    ).fit()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5"

    parser = argparse.ArgumentParser(
        description="Tuning script parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args = parser.parse_args()

    if args.smoke_test:
        ray.init(num_cpus=2)
    else:
        ray.init()

    run(smoke_test=args.smoke_test)
    ray.shutdown()