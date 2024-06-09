import argparse
from typing import Optional
from typing import Sequence
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import warnings
import ray
from ray.rllib.algorithms.algorithm import Algorithm
import os
from custom_models import FullyConnectedSoftmaxNetwork
from custom_action_dist import TorchDirichlet
from ray.rllib.models import ModelCatalog
from policy_evaluation import PolicyEvaluation
from environment import SchedulingEnv
import matplotlib.pyplot as plt
from statistics import mean, median, variance



#rom environment_corr import SchedulingEnv
from constants import (
    MS_AGENT
)
import json
import ray
from statistics import mean
# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def compute_overall_tardiness(jobs, products,staticConfigurationFilePath,plot) -> float:
    """Computes the sum tardiness across all jobs.

    For computing tardiness of a job,
    - we look at last finished operation and subtract the due-date from it.
    - if completion time is earlier than due-date, tardiness is zero.
    """
    tardiness = 0
    # tardiness_array = []
    percent_diff = []
    percent_tardiness = []
    for jobName, jobDetails in jobs.items():
        # print(jobDetails)
        # TODO
        # collect everything from here with the tardiness to analyse performance
        # study the datatype of the jobDetails and number of features here
        # correlation analysis of the jobDetails part 
        # learning trends of each agent w.r.t priority
        # priority as due date 
        completion_time = jobDetails['completionTime']
        due_date = products[jobDetails['productName']]['duedate']
        percent_diff.append(((completion_time - due_date)*100)/due_date)
        percent_tardiness.append(((max(0, completion_time - due_date))*100)/due_date)
        tardiness += max(0, completion_time - due_date)
        # tardiness_array.append(max(0, completion_time - due_date))
    mean_percent_diff = mean(percent_diff)
    median_percent_diff = median(percent_diff)
    variance_diff = variance(percent_diff)
    mean_tardiness_diff = mean(percent_tardiness)
    median_tardiness_diff = median(percent_tardiness)
    print('Difference percentage mean:', mean_percent_diff)
    print('Difference percentage median:', median_percent_diff)
    print('Difference percentage variance:', variance_diff)
    print('Tardiness percentage mean:', mean_tardiness_diff)
    print('Tardiness percentage median:', median_tardiness_diff)

    print(len(percent_tardiness))

    #print("difference array", percent_diff)

    if plot:
       plt.hist(percent_diff, bins = 'auto', color='cyan', edgecolor='black')
       plt.xlabel('Difference Percentages')
       plt.ylabel('job count')
       plt.savefig(staticConfigurationFilePath.split('.')[0]+'.png')
       plt.close()
    

    return tardiness

def parse_arguments(argv: Optional[Sequence[str]] = None) -> dict:
    """Defines cmd args for this program and parses the provided arg values and returns them.
    """
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser()

    ## Arg: env static configuration filepath
    parser.add_argument(
        '-scf', '--static-config-filepath',
        default= 'data/constrained_resources_mediumstatic_configuration_16.json',
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
             num_gpus = 4,
             ignore_reinit_error=True,
             _system_config={
                 "object_spilling_config": json.dumps(
                     {"type": "filesystem",
                      "params": {"directory_path": '/data/adatta14/PycharmProjects'}},
                 )
             },
             )
    args = parse_arguments()

    print(args)

    shouldRender = False
    maxSteps = None

    # Instantiate the scheduling environment
    config = {
        "staticConfigurationFilePath": args.static_config_filepath,
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
                #print("OS_Agent action: ", a)
                actions_dict[agentName] = a

        # Step the environment
        obs_dict, rew_dict, terminateds_dict, truncateds_dict, infos_dict = env.step(actions_dict)
        print(terminateds_dict)
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
            print("opname", opName)
            #print(opDetails['completionTime'])
            #print(opDetails['duedate'])
            #collect due date for priority and collect completion time for performance 
            if opDetails['completionTime'] > opDetails['duedate']:

                print(opName, opDetails['completionTime'], opDetails['duedate'])
        print("Cumulative Tardiness: ", compute_overall_tardiness(jobs, products, args.static_config_filepath,True))
        
    else:
        print("Truncated!!!")
    ray.shutdown()
