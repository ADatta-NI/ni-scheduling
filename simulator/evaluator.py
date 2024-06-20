import argparse
import os
import time
import csv
from argparse import Namespace
from typing import Optional, Tuple, Dict, Any
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
from policy_evaluation import PolicyEvaluation, PolicyScalableEvaluation
from environment import SchedulingEnv
import matplotlib.pyplot as plt
from statistics import mean, median, variance

# rom environment_corr import SchedulingEnv
from constants import (
    MS_AGENT
)
import json
import ray
from statistics import mean

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def create_feature_space(jobs, products, staticConfigurationFilePath):
    with open(staticConfigurationFilePath, 'r') as file:
        data = json.load(file)

    # Extract tardiness per product of given file
    tardiness_mean_dict, tardiness_median_dict = compute_overall_tardiness_per_product(jobs, products)

    # Extract the products, operations, and testers
    products = data['products']['items']
    operations = data['operations']['items']
    testers = data['testers']['items']

    # Prepare data for CSV
    csv_data = []
    for product_id, product_info in products.items():
        row = {
            'Arrival': product_info['arrival'],
            'Quantity': product_info['quantity'],
            'Due Date': product_info['duedate'],
            'Cyclomatic Complexity': product_info['complexity'],
            'Meshedness': product_info['meshedness'],
            'Node Edge Ratio': product_info['node_edge_ratio'],
            'Gamma Connectivity': product_info['gamma_connectivity'],
            'Edge Density': product_info['edge_density'],
            'Max weight': product_info['max_weight'],
            'Min weight': product_info['min_weight'],

        }

        # Calculate the total number of compatible configurations for all operations

        for operation_id in product_info['operations']:
            operation_info = operations[operation_id]
            compatible_configurations = set()
            for config in operation_info['compatibleConfigurations']:
                compatible_configurations.add(config)
                total_compatible_configurations = len(compatible_configurations)
        row['Total Compatible Configurations'] = total_compatible_configurations
        # print(compatible_configurations)

        # Calculate the total number of testers for each product
        total_testers = 0
        for tester_id, tester_info in testers.items():
            if any(config in tester_info['supportedConfigurations'] for config in compatible_configurations):
                total_testers += 1
        row['Total Testers'] = total_testers

        ## adding the final column
        row['Mean Tardiness'] = tardiness_mean_dict[product_id]
        row['Median Tardiness'] = tardiness_median_dict[product_id]

        csv_data.append(row)

    # Create directory
    directory = 'impala_analysis_files'
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    # Write data to CSV
    # Full file path
    file_path = staticConfigurationFilePath

    # Extract the base name (file name with extension)
    base_name = os.path.basename(file_path)

    # Extract the file name without the extension
    file_name = os.path.splitext(base_name)[0]

    print(file_name)

    with open('impala_analysis_files/' +
              file_name + '.csv',
              'w', newline='') as csvfile:
        fieldnames = list(csv_data[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

    print("CSV file has been created successfully.")


def compute_overall_tardiness_per_product(jobs, products) -> tuple[dict[Any, int], dict[Any, int]]:
    """Computes the sum tardiness across all products and return
    the tuple of tardiness

    For computing tardiness of a job,
    - we look at last finished operation and subtract the due-date from it.
    - if completion time is earlier than due-date, tardiness is zero.
    """
    tardiness_mean_dict = {}
    tardiness_median_dict = {}
    for product in products.keys():
        tardiness_list = []
        for jobName, jobDetails in jobs.items():
            if not (jobDetails['productName'] == product):
                continue
            else:
                completion_time = jobDetails['completionTime']
                due_date = products[jobDetails['productName']]['duedate']
                # print(f'{product}:', max(0, completion_time - due_date))
                tardiness_list.append(max(0, completion_time - due_date))
        mean_tardiness = mean(tardiness_list)
        median_tardiness = median(tardiness_list)
        tardiness_mean_dict[product] = mean_tardiness
        tardiness_median_dict[product] = median_tardiness
    return tardiness_mean_dict, tardiness_median_dict


def compute_overall_tardiness(jobs, products, staticConfigurationFilePath, plot) -> float:
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
        print(f"{jobDetails}")
        completion_time = jobDetails['completionTime']
        due_date = products[jobDetails['productName']]['duedate']
        percent_diff.append(((completion_time - due_date) * 100) / due_date)
        percent_tardiness.append(((max(0, completion_time - due_date)) * 100) / due_date)
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

    # print("difference array", percent_diff)
    print('products:', f"{products.keys()}")
    print('length of tardiness array', len(percent_tardiness))

    if plot:
        plt.hist(percent_diff, bins=10, color='lightgreen', edgecolor='black')
        plt.xlabel('Difference Percentages')
        plt.ylabel('job count')
        plt.savefig(staticConfigurationFilePath.split('.')[0] + '.png')
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
        default='/data/adatta14/PycharmProjects/ni-scheduling/simulator/data/eda_analysis20.json',
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
    ## Arg: algorithm
    parser.add_argument(
        '-a', '--algorithm', '--algo',
        default='impala',
        choices=('ppo', 'sac', 'impala'),
        help='Specify the algorithm using which the policies have to be learned. (Default: %(default)s)'
    )
    # Parse and return argument values
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments()

    print(args)

    shouldRender = False
    maxSteps = None

    # Instantiate the scheduling environment
    inference_config = {"num_workers": 2}
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
    algorithm = args.algorithm
    # evaluator = PolicyEvaluation(env_config=config, checkpoint_path=checkpoint_path, random_policy=random_policy)

    evaluator = PolicyScalableEvaluation(env_config=config,
                                         modify_config=inference_config,
                                         checkpoint_path=checkpoint_path,
                                         random_policy=random_policy)

    start_time = time.time()
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
            # print(opDetails['completionTime'])
            # print(opDetails['duedate'])
            # if opDetails['completionTime'] > opDetails['duedate']:

            # print(opName, opDetails['completionTime'], opDetails['duedate'])
        print("Cumulative Tardiness: ", compute_overall_tardiness(jobs, products, args.static_config_filepath, True))
        tardiness_dict = compute_overall_tardiness_per_product(jobs, products)
        print(f'{tardiness_dict}')
        # create_feature_space(jobs, products, args.static_config_filepath)
    else:
        print("Truncated!!!")
    ray.shutdown()
    elapsed_time = time.time() - start_time
    print('Time elapsed:', elapsed_time)
    print(inference_config)