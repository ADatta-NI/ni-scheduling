from os.path import expanduser
import re

obs_keys = [
    'num_of_ops_in_global_bag',
    'setup_plus_test_time_of_ops_in_global_bag_min',
    'setup_plus_test_time_of_ops_in_global_bag_max',
    'setup_plus_test_time_of_ops_in_global_bag_mean',
    'setup_plus_test_time_of_ops_in_global_bag_std',
    'num_of_compat_configs_for_ops_in_global_bag_min',
    'num_of_compat_configs_for_ops_in_global_bag_max',
    'num_of_compat_configs_for_ops_in_global_bag_mean',
    'num_of_compat_configs_for_ops_in_global_bag_std',
    'due_date_of_ops_in_global_bag_min',
    'due_date_of_ops_in_global_bag_max',
    'due_date_of_ops_in_global_bag_mean',
    'due_date_of_ops_in_global_bag_std',
    'due_date_of_ops_in_global_bag_skewness',
    'due_date_of_ops_in_global_bag_kurtosis',
    'remaining_test_time_of_jobs_corresponding_to_ops_in_global_bag_min',
    'remaining_test_time_of_jobs_corresponding_to_ops_in_global_bag_max',
    'remaining_test_time_of_jobs_corresponding_to_ops_in_global_bag_mean',
    'remaining_test_time_of_jobs_corresponding_to_ops_in_global_bag_std',
    'remaining_num_of_ops_for_jobs_corresponding_to_ops_in_global_bag_min',
    'remaining_num_of_ops_for_jobs_corresponding_to_ops_in_global_bag_max',
    'remaining_num_of_ops_for_jobs_corresponding_to_ops_in_global_bag_mean',
    'remaining_num_of_ops_for_jobs_corresponding_to_ops_in_global_bag_std',
    'avg_setup_plus_test_time_of_the_op_',
    'num_of_compat_configs_for_the_op',
    'due_date_of_the_job_corresponding_to_the_op',
    'remaining_test_time_of_job_corresponding_to_the_op',
    'remaining_num_of_ops_for_job_corresponding_to_the_op',
    'num_of_ops_in_local_bags_min',
    'num_of_ops_in_local_bags_max',
    'num_of_ops_in_local_bags_mean',
    'num_of_ops_in_local_bags_std',
    'total_setup_plus_test_time_of_ops_in_local_bags_min',
    'total_setup_plus_test_time_of_ops_in_local_bags_max',
    'total_setup_plus_test_time_of_ops_in_local_bags_mean',
    'total_setup_plus_test_time_of_ops_in_local_bags_std',
    'num_of_setups_in_local_bags_min',
    'num_of_setups_in_local_bags_max',
    'num_of_setups_in_local_bags_mean',
    'num_of_setups_in_local_bags_std',
    'due_date_skewness_in_local_bags_min',
    'due_date_skewness_in_local_bags_max',
    'due_date_skewness_in_local_bags_mean',
    'due_date_skewness_in_local_bags_std',
    'due_date_kurtosis_in_local_bags_min',
    'due_date_kurtosis_in_local_bags_max',
    'due_date_kurtosis_in_local_bags_mean',
    'due_date_kurtosis_in_local_bags_std',
    'ratio_of_setup_to_test_time_in_local_bags_min',
    'ratio_of_setup_to_test_time_in_local_bags_max',
    'ratio_of_setup_to_test_time_in_local_bags_mean',
    'ratio_of_setup_to_test_time_in_local_bags_std'
]

input_file_path = expanduser("~") + "/home/byagant1/tmux-user-logs/" + "tmux-0-6-0-20231008T191957.log"
output_obs_file_path = expanduser("~") + "/tmp/observations_with_filter.txt"


def extract_msagent_observations(inputfilename, outputfilename):
    p = re.compile("Observations:  {'MSAgent': .*(?=})}")
    with open(inputfilename, mode="r") as logFile:
        doc = logFile.read()
        matches = re.findall(p, doc)
    return matches

def get_list_of_floats_from_observation_log_str(string_list):
    string_list = string_list.removeprefix("Observations:  {'MSAgent': ")
    string_list = string_list.removesuffix("}")
    # Remove the brackets and split the string into individual elements
    string_list = string_list.strip("[]")
    elements = string_list.split(",")

    # Convert each element to a float and create a list
    try:
        float_list = [float(element) for element in elements]
    except ValueError:
        print("Invalid input string.")
    return float_list

def get_observations():
    matches = extract_msagent_observations(input_file_path, output_obs_file_path)
    observations = []
    for match in matches:
        obs = get_list_of_floats_from_observation_log_str(match)
        observations.append(obs)
    return observations
    
def get_stats(data, names):
    # Transpose the data to work with columns instead of rows
    columns = list(map(list, zip(*data)))

    # Calculate mean, variance, min, and max for each column
    column_stats = []

    name = 0
    for column in columns:
        mean = sum(column) / len(column)
        variance = sum((x - mean) ** 2 for x in column) / len(column)
        min_value = min(column)
        max_value = max(column)
        
        column_stats.append({
            'name': names[name],
            'mean': mean,
            'variance': variance,
            'min': min_value,
            'max': max_value
        })
        name += 1

    return column_stats


observations = get_observations()
obs_stats = get_stats(observations, obs_keys)

with open(output_obs_file_path, mode='wt') as out:
    for obs in obs_stats:
        out.write(str(obs) + '\n')

