import argparse
from typing import Optional
from typing import Sequence
import matplotlib.pyplot as plt
import statistics
import warnings
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import os
from custom_models import FullyConnectedSoftmaxNetwork
from custom_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog
from policy_evaluation import PolicyEvaluation
# from environment import SchedulingEnv
from scheduling_milp_plot import SchedulingEnv
from constants import (
    MS_AGENT
)
import json
import ray

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

ModelCatalog.register_custom_model("custom_softmax_model", FullyConnectedSoftmaxNetwork)

print("Model registered successfully")

ModelCatalog.register_custom_action_dist("dirichlet_dist", TorchDirichlet)

print("Distribution registered successfully")


def compute_overall_tardiness(jobs, products) -> float:
    """Computes the sum tardiness across all jobs.

    For computing tardiness of a job,
    - we look at last finished operation and subtract the due-date from it.
    - if completion time is earlier than due-date, tardiness is zero.
    """
    tardiness = 0
    for jobName, jobDetails in jobs.items():
        completion_time = jobDetails['completionTime']
        due_date = products[jobDetails['productName']]['duedate']
        # print(completion_time - due_date)
        tardiness += max(0, completion_time - due_date)
    return tardiness


def parse_arguments(argv: Optional[Sequence[str]] = None) -> dict:
    """Defines cmd args for this program and parses the provided arg values and returns them.
    """
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser()

    ## Arg: env static configuration filepath
    parser.add_argument(
        '-scf', '--static-config-filepath',
        nargs='+',  # Accept one or more arguments
        default=[
            '/home/adatta14/PycharmProjects/ni-scheduling-project/simulator/data/constrained_resources_try_newstatic_configuration_5.json',
            '/home/adatta14/PycharmProjects/ni-scheduling-project/simulator/data/constrained_resources_try_newstatic_configuration_6.json',
            '/home/adatta14/PycharmProjects/ni-scheduling-project/simulator/data/constrained_resources_try_newstatic_configuration_8.json',
            '/home/adatta14/PycharmProjects/ni-scheduling-project/simulator/data/constrained_resources_try_newstatic_configuration_9.json'],

        help='Specify the static configuration file path using which environment should be simulated.'
    )

    ## Arg: algorithm checkpoint filepath
    parser.add_argument(
        '-cf', '--checkpoint-filepath',
        help=f'Specify the checkpoint filepath of the algorithm which has to be evaluated.'
    )

    ## Random policy evaluation flag
    parser.add_argument(
        '-rp', '--random-policy',
        action='store_true',
        help='Specify whether a random policy should be used for evaluation.'
    )

    # Parse and return argument values
    return parser.parse_args(argv)


if __name__ == '__main__':
    # Parse CMD Arguments
    ray.init(num_cpus=64,
             num_gpus=4,
             ignore_reinit_error=True,
             _system_config={
                 "object_spilling_config": json.dumps(
                     {"type": "filesystem",
                      "params": {"directory_path": '/home/adatta14/PycharmProjects'}},
                 )
             },
             )
    args = parse_arguments()

    print(args)

    shouldRender = False
    maxSteps = None

    # Instantiate the scheduling environments
    completionTime_intercept = 0
    due_date_intercept = 0
    last_completionTime_intercept = 0
    last_due_date_intercept = 0
    tardiness_array = []
    count = 0
    for i, static_config_filepath in enumerate(args.static_config_filepath):
        count += 1
        ModelCatalog.register_custom_model("custom_softmax_model", FullyConnectedSoftmaxNetwork)

        print("Model registered successfully")

        ModelCatalog.register_custom_action_dist("dirichlet_dist", TorchDirichlet)

        print("Distribution registered successfully")
        config = {
            "staticConfigurationFilePath": static_config_filepath,
            "maxSteps": maxSteps,
            "setupRendering": shouldRender,
            "sampleRandomProblem": False,
            "followGreedyStrategy": False,
        }

        env = SchedulingEnv(config=config)

        # Instantiate the evaluator
        checkpoint_path = args.checkpoint_filepath
        random_policy = args.random_policy
        evaluator = PolicyEvaluation(env_config=config, checkpoint_path=checkpoint_path, random_policy=random_policy)
        # Get initial observation
        obs_dict, infos_dict = env.reset()
        # Agent-Environment interaction loop
        terminal_state_reached = False
        msagent_rewards = []
        osagent_rewards = {}
        operations = None
        while True:
            # Compute actions using evaluator
            actions_dict = {}
            for agentName, obs in obs_dict.items():
                if agentName == MS_AGENT:
                    a = evaluator.compute_msagent_action(obs)
                    print("MS_Agent action: ", a)
                    actions_dict[agentName] = a
                else:
                    a = evaluator.compute_osagent_action(obs)
                    # print("OS_Agent action: ", a)
                    actions_dict[agentName] = a

            # Step the environment
            obs_dict, rew_dict, terminateds_dict, truncateds_dict, infos_dict = env.step(actions_dict)

            # Render the environment
            if shouldRender:
                env.render()
                print("Rewards: ", rew_dict)
            if MS_AGENT in rew_dict:
                msagent_rewards.append(rew_dict[MS_AGENT])
            else:
                for osagent, reward in rew_dict.items():
                    if osagent not in osagent_rewards:
                        osagent_rewards[osagent] = [reward]
                    else:
                        osagent_rewards[osagent].append(reward)

            # Check termination
            if terminateds_dict["__all__"] == True or truncateds_dict["__all__"] == True:
                terminal_state_reached = terminateds_dict["__all__"]
                products = infos_dict['__common__']['products']
                jobs = infos_dict['__common__']['jobs']
                operations = infos_dict['__common__']['operations']
                testers = infos_dict['__common__']['testers']
                break

        if terminal_state_reached == True:
            print("Schedule Complete!!!")
            print("Mean MSAGENT reward: ", statistics.mean(msagent_rewards))
            for osagent in osagent_rewards.keys():
                print("Sum of OSAgent rewards for", osagent, "is: ", sum(osagent_rewards[osagent]))

            for opName, opDetails in operations.items():
                # print("opname",opName)
                # print(opDetails['completionTime'])
                # print(opDetails['duedate'])
                if opDetails['completionTime'] > opDetails['duedate']:
                    completionTime_intercept = max(completionTime_intercept, last_completionTime_intercept+opDetails['completionTime'])
                    due_date_intercept = max(due_date_intercept, last_due_date_intercept+opDetails['duedate'])
                    print(opName, last_completionTime_intercept+opDetails['completionTime'], last_due_date_intercept+opDetails['duedate'])

            last_completionTime_intercept = completionTime_intercept
            last_due_date_intercept = due_date_intercept
            print(last_completionTime_intercept)

            print("Tardiness: ", compute_overall_tardiness(jobs, products))
            tardiness_array.append(compute_overall_tardiness(jobs, products))
            print(tardiness_array)
        else:
            print("Truncated!!!")
        tardiness_intercept = compute_overall_tardiness(jobs, products)
        ray.shutdown()
