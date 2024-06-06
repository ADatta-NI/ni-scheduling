import argparse
from typing import Optional
from typing import Sequence
import matplotlib.pyplot as plt
import statistics
import warnings
from ray.rllib.algorithms.algorithm import Algorithm

from policy_evaluation import PolicyEvaluation
from environment import SchedulingEnv
from constants import (
    MS_AGENT
)

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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
        default='data/static_configuration.json',
        help='Specify the static configuration file path using which environment should be simulated.'
    )

    ## Arg: msagent greedy strategy to be followed
    parser.add_argument(
        '-msgs', '--ms-greedy-strategy',
        choices=('least-num-ops', 'least-load'),
        help=f'Specify the greedy strategy to be used for MSAgent.'
    )

    ## Arg: osagent greedy strategy to be followed
    parser.add_argument(
        '-osgs', '--os-greedy-strategy',
        choices=('least-test-time', 'nearest-due-date'),
        help=f'Specify the greedy strategy to be used for OSAgent.'
    )

    # Parse and return argument values
    return parser.parse_args(argv)

if __name__ == '__main__':
    # Parse CMD Arguments
    args = parse_arguments()

    print(args)

    shouldRender = True
    maxSteps = None

    # Instantiate the scheduling environment
    config = {
        "staticConfigurationFilePath": args.static_config_filepath,
        "maxSteps": maxSteps,
        "setupRendering": shouldRender,
        "sampleRandomProblem": False,
        "followGreedyStrategy": True,
        "msGreedyStrategy": args.ms_greedy_strategy,
        "osGreedyStrategy": args.os_greedy_strategy,
    }
    env = SchedulingEnv(config=config)

    # Instantiate the evaluator
    evaluator = PolicyEvaluation(env_config=config, random_policy=True)

    # Get initial observation
    obs_dict, infos_dict = env.reset()

    # Agent-Environment interaction loop
    terminal_state_reached = False
    while True:
        # Compute actions using evaluator
        actions_dict = {}
        for agentName, obs in obs_dict.items():
            if agentName == MS_AGENT:
                a = evaluator.compute_msagent_action(obs)
                actions_dict[agentName] = a
            else:
                a = evaluator.compute_osagent_action(obs)
                actions_dict[agentName] = a

        # Step the environment
        obs_dict, rew_dict, terminateds_dict, truncateds_dict, infos_dict = env.step(actions_dict)

        # Render the environment
        if shouldRender:
            env.render()

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
        for opName, opDetails in operations.items():
            if opDetails['completionTime'] > opDetails['duedate']:
                print(opName, opDetails['completionTime'], opDetails['duedate'])
        print("Tardiness: ", compute_overall_tardiness(jobs, products))
    else:
        print("Truncated!!!")
