import os
import datetime
import argparse
from typing import Optional
from typing import Sequence
import shutil
import datetime
import ray
import custom_models 
import custom_action_dist 

from ray import tune
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.a3c import A3CConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.td3 import TD3Config
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
import warnings
from constants import (
    MS_POLICY, OS_POLICY, CHECKPOINT_ROOT, RAY_RESULTS_ROOT
)
from algorithm_configurations import(
    ppo_config, a3c_config, sac_config, ddpg_config, td3_config , appo_config
)
from custom_models import FullyConnectedSoftmaxNetwork
from custom_action_dist import TorchDirichlet
from send_email import send_email

# Filter unnecessary warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

    
def get_ppo_config() -> PPOConfig:
    return ppo_config

def get_appo_config() -> APPOConfig:
    return appo_config
    
def get_a3c_config() -> A3CConfig:
    return a3c_config

def get_sac_config() -> SACConfig:
    return sac_config
    
def get_ddpg_config() -> DDPGConfig:
    return ddpg_config

def get_td3_config() -> TD3Config:
    return td3_config

def get_algorithm_config(args) -> AlgorithmConfig:
    config: AlgorithmConfig = None
    match args.algorithm:
        case 'ppo':
            config = get_ppo_config()
        case 'appo':
            config = get_appo_config()
        case 'a3c':
            config = get_a3c_config()
        case 'sac':
            config = get_sac_config()
        case 'ddpg':
            config = get_ddpg_config()
        case 'td3':
            config = get_td3_config()
        case _:
            raise ValueError(f"Either {args.algorithm} is not supported or doesn't exist.")


    if args.models == 'mlp_soft': 
        ModelCatalog.register_custom_model("custom_softmax_model", FullyConnectedSoftmaxNetwork)
        config.model = {
                    "fcnet_hiddens": [256,256],
                    "custom_model" : "custom_softmax_model",
                    "free_log_std" : True,
                    "use_lstm" : False,
                    "max_seq_len" : 50,
                    
                    }
    elif args.models == 'mlp_soft_dirichlet': 
        ModelCatalog.register_custom_model("custom_softmax_model", FullyConnectedSoftmaxNetwork)
        ModelCatalog.register_custom_action_dist("dirichlet_dist", TorchDirichlet)
        config.model = {
                    "fcnet_hiddens": [256,256],
                    "custom_model" : "custom_softmax_model",
                    "custom_action_dist": "dirichlet_dist",
                    "free_log_std" : False,
                    "use_lstm" : False,
                    "max_seq_len" : 50,
                    
                    }
    ##without custom Model to avoid errors 
    elif args.models == 'mlp': 
        config.model = {
                    "fcnet_hiddens": [256,256],
                    "free_log_std" : False,
                    "use_lstm" : False,
                    "max_seq_len" : 50,
                    
                    }
    
    # Gamma
    if args.gamma != None:
        config.gamma = args.gamma

    if args.num_gpus != None:
        config.num_gpus = args.num_gpus
    
    # Learning Rate
    if args.learningrate != None:
        print("Prev lr: ", config.lr)
        config.lr = args.learningrate
        print("After lr: ", config.lr)

    # Gradient Clip Norm
    if args.gradclip != None:
        print("Prev grad_clip: ", config.grad_clip)
        config.grad_clip = args.gradclip
        print("After grad_clip: ", config.grad_clip)

    # Static Configuration
    # if args.staticconfigfile != None:
    #     config.env_config['staticConfigurationFilePath'] = args.staticconfigfile

    return config


def parse_arguments(argv: Optional[Sequence[str]] = None) -> dict:
    """Defines cmd args for this program and parses the provided arg values and returns them.
    """
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser()

    ## Arg: algorithm name
    parser.add_argument(
        '-a', '--algorithm', '--algo',
        default='ppo',
        choices=('ppo', 'appo', 'a3c', 'sac', 'ddpg', 'td3'),
        help='Specify the algorithm using which the policies have to be learned. (Default: %(default)s)'
    )

    parser.add_argument(
        '-m', '--models', '--mod',
        default='mlp_soft_dirichlet',
        choices=('mlp', 'mlp_soft', 'mlp_soft_dirichlet'),
        help='Specify the model using which the policies have to be learned. (Default: %(default)s)'
    )
    
    ## Arg: gamma value
    parser.add_argument(
        '-g', '--gamma',
        type=float,
        default = 0.99,
        help='Specify the discount factor to be used.'
    )

    parser.add_argument(
        '-num', '--num_gpus',
        type= int,
        default = 1,
        help='Specify the number of GPUs.'
    )

    ## Arg: learning rate value
    parser.add_argument(
        '-lr', '--learningrate',
        type=float,
        default = 1e-4,
        help='Specify the learning rate to be used.'
    )

    ## Arg: gradient clip norm value
    parser.add_argument(
        '-gc', '--gradclip',
        type=float,
        help='Specify the gradient clipping norm to be used.'
    )

    ## Arg: number of iterations
    parser.add_argument(
        '-i', '--iterations', '--iters',
        type=int,
        default = 1000,
        help='Specify the number of iterations to be trained.'
    )

    ## Arg: algorithm checkpoint filepath
    parser.add_argument(
        '-cf', '--checkpoint-filepath',
        help=f'Specify the checkpoint filepath of the algorithm to resume training.'
    )

    # Parse and return argument values
    return parser.parse_args(argv)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0"      # CUDA_VISIBLE_DEVICES=0 python3 myScript.py

    start_time = datetime.datetime.now()

    # Parse CMD Arguments
    args = parse_arguments()

    print(start_time, args)

    # Starting a new instance of Ray
    ray.shutdown()
    context = ray.init(include_dashboard=True)

    if args.checkpoint_filepath != None:
        # Load algo from checkpoint
        algo = Algorithm.from_checkpoint(args.checkpoint_filepath)
    else:
        # Get Algorithm config
        config = get_algorithm_config(args)      

        # Build the algorithm
        algo = config.build()

    print("Config: ", algo.get_config().to_dict())

    # print("Model Summary of ", MS_POLICY)
    # print(algo.get_policy(MS_POLICY).model.base_model.summary())

    # print("Model Summary of ", OS_POLICY)
    # print(algo.get_policy(OS_POLICY).model.base_model.summary())

    checkpoint_save_path = CHECKPOINT_ROOT + str(start_time) + "_" + "_".join(key + "-" + str(val) for key, val in vars(args).items()) + "/"

    # Clean up log directories
    # shutil.copytree(CHECKPOINT_ROOT, os.environ['HOME'] + '/checkpoints/' + str(args.algorithm) + "_" + str(args.gamma) + "_" + str(args.iterations) + str(datetime.datetime.now()) + "/")
    # shutil.rmtree(CHECKPOINT_ROOT, ignore_errors=True, onerror=None)
    # shutil.rmtree(RAY_RESULTS_ROOT, ignore_errors=True, onerror=None)

    try:
        # Train the model using algorithm
        print("===========TRAINING===========")
        num_of_iterations = args.iterations
        for _ in range(num_of_iterations):
            results = algo.train()

            if MS_POLICY in results["policy_reward_mean"] and OS_POLICY in results['policy_reward_mean']:
                print(f"Iteration={algo.iteration}: R1={results['policy_reward_mean'][MS_POLICY]} R2={results['policy_reward_mean'][OS_POLICY]}")

            if (algo.iteration) % 100 == 0:
                filename = algo.save(checkpoint_save_path + "checkpoint_" + str(algo.iteration) + "/") 
                print(f"checkpoint saved after iteration: {algo.iteration} at {filename}")
        print("=============DONE=============")
        end_time = datetime.datetime.now() 
        total_run_time = (end_time-start_time).total_seconds()
        print(end_time)
        print(total_run_time)
    except Exception as e:
        send_email(subject="SchedulingProject: RunFailure!", body=f"Started at: {start_time}, Failed with exception: \n{e}")
    else:
        send_email(subject="SchedulingProject: RunSuccess!", body=f"Started at: {start_time}, Ended at: {end_time}, Total Runtime: {total_run_time}")
    finally:
        ray.shutdown()
