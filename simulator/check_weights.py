from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.policy import Policy

from constants import (
    MS_POLICY, OS_POLICY
)
import obs_utils
import ray

for checkpoint_num in range(1530, 0, -10):
    checkpoint_path = "/home/byagant1/checkpoints/2023-10-22 23:45:34.821266_algorithm-ppo_gamma-None_learningrate-None_gradclip-None_iterations-2000_checkpoint_filepath-None/checkpoint_" + str(checkpoint_num)
    algo = Algorithm.from_checkpoint(checkpoint_path)

    ms_policy = algo.get_policy(MS_POLICY)
    ms_weights = ms_policy.get_weights()

    os_policy = algo.get_policy(OS_POLICY)
    os_weights = os_policy.get_weights()

    print("*************************************************************************")
    print("checkpoint_" + str(checkpoint_num))
    layer_names = ms_weights.keys()
    for layer_name in layer_names:
        print(layer_name, ms_weights[layer_name].shape)
        if layer_name.endswith("bias"):
            present = obs_utils.contains_invalid_numbers(ms_weights[layer_name])
            if present:
                print(True, ms_weights[layer_name])
        else:
            for w in ms_weights[layer_name]:
                present = obs_utils.contains_invalid_numbers(w) 
                if present:
                    print(True, w)
    print("*************************************************************************")
    ray.shutdown()





"""
odict_keys(['ms-policy/fc_1/kernel', 'ms-policy/fc_1/bias', 'ms-policy/fc_value_1/kernel', 'ms-policy/fc_value_1/bias', 'ms-policy/fc_2/kernel', 'ms-po
licy/fc_2/bias', 'ms-policy/fc_value_2/kernel', 'ms-policy/fc_value_2/bias', 'ms-policy/fc_out/kernel', 'ms-policy/fc_out/bias', 'ms-policy/value_out/k
ernel', 'ms-policy/value_out/bias'])
"""