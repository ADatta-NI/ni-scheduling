import numpy as np
import pandas as pd
import random 
from typing import Tuple
import math
import scipy

from constants import OperationStatus, JobStatus, SCHEDULING_ENV_RANDOM_SEED

"""Set of utility functions used by Scheduling Environment for performing tasks related to handling observations.
"""

def flatten(listOfLists) -> list:
    """Returns a flattened list by taking in list of lists
    """
    return [item for aList in listOfLists for item in aList]

def sample(config, posNeeded = True) -> float:
    """Returns a sampled value from provided distribution.

    - config is a dictionary containing information about distribution.
    - By default since most of the sampling requirements need positive values (eg. test times) in this project, this function assumes posNeeded to be true. 
    """
    match config['distribution']:
        case 'normal':
            rng = np.random.default_rng(seed=SCHEDULING_ENV_RANDOM_SEED)
            while True:
                val = rng.normal(loc=config['mean'], scale=config['std'])
                if posNeeded == True and val <= 0:
                    continue
                return val

def average(config) -> float:
    """Returns the average/mean of provided distribution.

    config is a dictionary containing information about distribution.
    """
    match config['distribution']:
        case 'normal':
            return config['mean']

def compute_opspecific_avg_setup_plus_test_time(data, jobs, operations, opName) -> float:
    """Returns the average setup plus test time of the given operation
    """
    jobName = operations[opName]['jobName']
    productName = jobs[jobName]['productName']
    opLogicalId = operations[opName]['logicalOperationId']
    opType = data['products']['items'][productName]['operations'][opLogicalId]
    
    avg_setup_plus_test_times = []
    for config in data['operations']['items'][opType]['compatibleConfigurations']:
        avg_setup = np.average(data['configurations']['items'][config]['setupTimes'])
        avg_test_time = average(data['operations']['items'][opType]['estimatedTestTime'][config])
        avg_setup_plus_test_times.append(avg_setup + avg_test_time)
    return np.average(avg_setup_plus_test_times)

def compute_opspecific_no_of_compatible_configurations(data, jobs, operations, opName) -> int:
    """Returns the number of compatible configurations for the given operation
    """
    jobName = operations[opName]['jobName']
    productName = jobs[jobName]['productName']
    opLogicalId = operations[opName]['logicalOperationId']
    opType = data['products']['items'][productName]['operations'][opLogicalId]
    return len(data['operations']['items'][opType]['compatibleConfigurations'])
    

def compute_opspecific_due_date_of_job(data, jobs, operations, opName) -> int:
    """Returns the due date of job corresponding to given operation
    """
    jobName = operations[opName]['jobName']
    productName = jobs[jobName]['productName']
    return data['products']['items'][productName]['duedate']


def compute_opspecific_remaining_test_time_of_job(jobs, operations, opName, currentTime) -> float:
    """Returns the remaining test time of the job corresponding to given operation
    """
    remaining_job_test_time = 0
    jobName = operations[opName]['jobName']
    for op in jobs[jobName]['operations']:
        remaining_job_test_time += compute_estimated_remaining_test_time_of_operation(operations, op, currentTime)
    return remaining_job_test_time

def compute_opspecific_no_of_remaining_ops_of_job(jobs, operations, opName) -> int:
    """Returns the remaining number of operations of job corresponding to given operation
    """
    jobName = operations[opName]['jobName']
    return compute_remaining_number_of_ops_for_job(jobs, operations, jobName)

def compute_avg_operation_test_time(data, opType) -> float:
    """Returns estimated test time of the given operation
    """
    test_times = []
    for config in data['operations']['items'][opType]['compatibleConfigurations']:
        test_time = average(data['operations']['items'][opType]['estimatedTestTime'][config])
        test_times.append(test_time)
    return np.average(test_times)

def compute_avg_operation_setup_time(data, configurations, opType) -> float:
    """Returns average setup time for the given operation
    """
    setup_times = []
    for config in data['operations']['items'][opType]['compatibleConfigurations']:
        setup_times.append(configurations[config]['averageSetupTime'])
    return np.average(setup_times)

def compute_estimated_operation_test_time_under_configuration(data, opType, config) -> float:
    """Returns the estimated test time under a given configuration for given operation
    """
    print(sample(data['operations']['items'][opType]['estimatedTestTime'][config]))
    return sample(data['operations']['items'][opType]['estimatedTestTime'][config])

def compute_estimated_remaining_test_time_of_operation(operations, opName, currentTime):
    """Returns the remaining test time of a given operation
    """
    match operations[opName]['status']:
        case OperationStatus.IN_PROGRESS:
            return operations[opName]['startTime'] + operations[opName]['setupTime'] + operations[opName]['estimatedTestTime'] - currentTime
        case OperationStatus.COMPLETED:
            return 0
        case _:
            return operations[opName]['averageTestTime']

##############################################################################################################################

# def get_descriptive_stats(values, req_stats, higher_moments_needed = False) -> list:
#     """Returns a list of descriptive statistics for the values list as per the keys in req_stats

#     higher_moment_needed parameter should be passed as True, if req_stats contains 'skewness' or 'kurtosis'
#     """
#     if len(values) == 0:
#         return [0] * len(req_stats)

#     values = pd.Series(values)

#     statistics = values.describe().to_dict()

#     if len(values) == 1:
#         statistics['std'] = 0

#     if higher_moments_needed == True:
#         if len(values) > 5:
#             statistics['skewness'] = values.skew()
#             statistics['kurtosis'] = values.kurtosis()
#         else:
#             statistics['skewness'] = 0
#             statistics['kurtosis'] = 0


#     for stat in req_stats:
#         if math.isnan(statistics[stat]) or statistics[stat] == float('inf') or statistics[stat] == float('-inf'):
#             print("CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") 
#             print(values.to_list())
#             print(stat, statistics[stat])
    
#     return [statistics[stat] for stat in req_stats]

def get_descriptive_stats(values, req_stats) -> list:
    """Returns a list of descriptive statistics for the values list as per the keys in req_stats
    """
    if len(values) == 0:
        return [0] * len(req_stats)

    result = []
    for stat in req_stats:
        if stat == "min":
            result.append(float(np.min(values)))
        elif stat == "max":
            result.append(float(np.max(values)))
        elif stat == "mean":
            result.append(float(np.mean(values)))
        elif stat == "std":
            if len(values) == 1:
                result.append(0)
            else:
                result.append(float(np.std(values)))
        elif stat == "skewness":
            if len(values) > 5:
                skew = scipy.stats.skew(values)
                result.append(0 if np.isnan(skew) else skew)
            else:
                result.append(0)
        elif stat == "kurtosis":
            if len(values) > 5:
                kurtosis = scipy.stats.kurtosis(values)
                result.append(0 if np.isnan(kurtosis) else kurtosis)
            else:
                result.append(0)

    return result

def compute_global_number_of_operations(globalBag) -> list:
    """Returns the number of operations in the globalBag
    """
    return [globalBag.size()]

def compute_bag_setup_plus_test_times(operations, bag) -> list:
    """Returns the stats about setup + test times of operations in the provided bag
    """
    setup_plus_test_times = []
    for op in bag:
        avg_test_time = operations[op]['averageTestTime']
        avg_setup_time = operations[op]['averageSetupTime']
        setup_plus_test_times.append(avg_setup_time + avg_test_time)
    return setup_plus_test_times

def compute_global_setup_plus_test_times(operations, globalBag) -> list:
    """Returns the stats about setup + test times of operations in the global bag
    """
    setup_plus_test_times = compute_bag_setup_plus_test_times(operations, globalBag)
    return get_descriptive_stats(setup_plus_test_times, ['min', 'max', 'mean', 'std'])

def compute_global_number_of_compatible_configurations(data, operations, globalBag) -> list:
    """Returns the stats about number of compatible configurations of operations in the global bag.
    """
    num_of_compat_configs = []
    for op in globalBag:
        num_of_compat_configs.append(len(data['operations']['items'][operations[op]['opType']]['compatibleConfigurations']))
    return get_descriptive_stats(num_of_compat_configs, ['min', 'max', 'mean', 'std'])

def compute_global_due_dates(operations, globalBag) -> list:
    """Returns the stats about due dates of operations in the global bag.
    """
    due_dates = collect_due_dates_of_bag(operations, globalBag)
    return get_descriptive_stats(due_dates, ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis'])

def collect_due_dates_of_bag(operations, bag):
    """Returns the due dates of operations in a bag
    """
    due_dates = []
    for op in bag:
        due_dates.append(operations[op]['duedate'])
    return due_dates

def compute_remaining_setup_plus_test_time_of_op(operations, opName, currentTime):
    """Returns remaining setup plus test time of the given operation
    """
    match operations[opName]['status']:
        case OperationStatus.NOT_STARTED | OperationStatus.IN_GLOBAL_BAG | OperationStatus.IN_LOCAL_BAG | OperationStatus.FAILED:
            return operations[opName]['averageSetupTime'] + operations[opName]['averageTestTime']
        case OperationStatus.IN_PROGRESS:
            return operations[opName]['completionTime'] - currentTime
        case OperationStatus.COMPLETED:
            return 0

def compute_remaining_setup_plus_test_time_of_job(jobs, operations, jobName, currentTime):
    """Returns remaining setup plus test time of the given job
    """
    remaining_setup_plus_test_time = 0
    for op in jobs[jobName]['operations']:
        remaining_setup_plus_test_time += compute_remaining_setup_plus_test_time_of_op(operations, op, currentTime)
    return remaining_setup_plus_test_time

def compute_remaining_test_times_of_jobs_corresponding_to_ops_in_bag(jobs, operations, bag, current_time) -> list:
    """Returns the remaining test time values of jobs corresponding to operations in given bag"""
    remaining_test_times = []
    for jobName, jobDetails in jobs.items():
        jobDetails['estimatedRemainingTime'] = compute_remaining_setup_plus_test_time_of_job(jobs, operations, jobName, current_time)
    for op in bag:
        jobName = operations[op]['jobName']
        remaining_test_times.append(jobs[jobName]['estimatedRemainingTime'])
    return remaining_test_times

def compute_global_remaining_test_time_of_jobs(data, jobs, operations, globalBag, current_time) -> list:
    """Returns stats about remaining test time of jobs corresponding to operations in the global bag 
    """
    remaining_test_times = compute_remaining_test_times_of_jobs_corresponding_to_ops_in_bag(jobs, operations, globalBag, current_time)
    return get_descriptive_stats(remaining_test_times, ['min', 'max', 'mean', 'std'])

def compute_remaining_number_of_ops_for_job(jobs, operations, jobName) -> int:
    """Returns remaining number of operations for a given job
    """
    remaining_no_of_ops = 0
    for op in jobs[jobName]['operations']:
        if operations[op]['status'] != OperationStatus.COMPLETED:
            remaining_no_of_ops += 1
    return remaining_no_of_ops

def compute_remaining_number_of_ops_for_jobs_in_bag(jobs, operations, bag) -> list:
    """Returns the stats about remaining number of operations for jobs corresponding to operations in the bag
    """
    remaining_no_of_ops = []
    for op in bag:
        jobName = operations[op]['jobName']
        remaining_no_of_ops.append(compute_remaining_number_of_ops_for_job(jobs, operations, jobName))
    return remaining_no_of_ops

def compute_global_remaining_number_of_operations_for_jobs(jobs, operations, globalBag) -> list:
    """Returns the stats about remaining number of operations for jobs corresponding to operations in the global bag
    """
    remaining_no_of_ops = compute_remaining_number_of_ops_for_jobs_in_bag(jobs, operations, globalBag)
    return get_descriptive_stats(remaining_no_of_ops, ['min', 'max', 'mean', 'std'])

##############################################################################################################################

def compute_local_number_of_operations(localBags) -> list:
    """Returns the stats about number of operations in local bags
    """
    no_of_ops = []
    for bag in localBags.values():
        no_of_ops.append(bag.size())
    return get_descriptive_stats(no_of_ops, ['min', 'max', 'mean', 'std'])

def get_total_setup_and_test_time_of_op_on_a_tester(data, operations, testers, configurations, opName, testerName) -> float:
    """Returns setup plus test time of an operation on a tester.
    """
    best_config = compute_best_config_for_operation_on_a_tester(data, operations, opName, testerName)
    setup_time = np.average(data['configurations']['items'][best_config]['setupTimes'])
    test_time = average(data['operations']['items'][operations[opName]['opType']]['estimatedTestTime'][best_config])
    return setup_time + test_time

def get_total_setup_and_total_test_time_of_local_bag(data, operations, configurations, bag, testerName) -> Tuple[float, float]:
    """Returns the total setup, total test time of a bag

    For calculating the total setup and test times in a local bag, we assume that 
    - we process every operation under it's most optimal configuration (optimal -> least test time)
    - we switch configuration when we complete all the operations which are supposed to be tested under that configuration.
    - Instead of estimating the ordering among available configs, we use avg setup times of each configuration as it's setup time
    """
    total_test_time = 0
    configs = set()
    for op in bag:
        best_config = compute_best_config_for_operation_on_a_tester(data, operations, op, testerName)
        configs.add(best_config)
        total_test_time += average(data['operations']['items'][operations[op]['opType']]['estimatedTestTime'][best_config])
    total_setup = sum([configurations[config]['averageSetupTime'] for config in configs])
    return total_setup, total_test_time

def compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, bag, testerName) -> float:
    """Returns the total setup plus test time of a bag.
    """
    total_setup, total_test_time = get_total_setup_and_total_test_time_of_local_bag(data, operations, configurations, bag, testerName)
    return total_setup + total_test_time 

    # total_setup_plus_test_time = 0
    # for op in bag:
    #     avg_test_time = operations[op]['averageTestTime']
    #     avg_setup_time = operations[op]['averageSetupTime']
    #     total_setup_plus_test_time += (avg_setup_time + avg_test_time)
    # return total_setup_plus_test_time

def compute_local_total_setup_plus_test_times(data, operations, configurations, localBags) -> list:
    """Returns the stats about setup + test times of operations in the local bag
    """
    total_setup_plus_test_times = []
    for testerName, bag in localBags.items():
        total_setup_plus_test_times.append(compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, bag, testerName))
    return get_descriptive_stats(total_setup_plus_test_times, ['min', 'max', 'mean', 'std'])


def compute_number_of_setups_of_bag(data, operations, bag, testerName) -> int:
    """Returns the minimum number of setups needed to complete all the operations in a bag when those operations are processed on their best available config 
    """
    configs = set()
    for op in bag:
        best_config = compute_best_config_for_operation_on_a_tester(data, operations, op, testerName)
        configs.add(best_config)
    return len(configs)

def compute_local_number_of_setups(data, operations, localBags) -> list:
    """Returns the stats about estimated number of setups across all local queues

    For calculating the number of setups in a local bag, we assume that 
    - we process every operation under it's most optimal configuration (optimal -> least test time)
    - we switch configuration when we complete all the operations which are supposed to be tested under that configuration.
    """
    num_of_setups = []
    for testerName, bag in localBags.items():
        num_of_setups.append(compute_number_of_setups_of_bag(data, operations, bag, testerName))
    return get_descriptive_stats(num_of_setups, ['min', 'max', 'mean', 'std'])

def compute_duedate_skewness_of_bag(operations, bag) -> float:
    """Returns the skewness statistic of duedate distribution corresponding to all the operations in a bag
    """
    due_dates = collect_due_dates_of_bag(operations, bag)
    return get_descriptive_stats(due_dates, ['skewness'])[0]

def compute_local_duedate_skewness(operations, localBags) -> list:
    """Returns the stats about skewness of due dates across all local bags 
    """
    skewness_values = []
    for bag in localBags.values():
        skewness_values.append(compute_duedate_skewness_of_bag(operations, bag))
    return get_descriptive_stats(skewness_values, ['min', 'max', 'mean', 'std'])

def compute_duedate_kurtosis_of_bag(operations, bag):
    """Returns the kurtosis statistic of duedate distribution corresponding to all the operations in a bag
    """
    due_dates = collect_due_dates_of_bag(operations, bag)
    return get_descriptive_stats(due_dates, ['kurtosis'])[0]

def compute_local_duedate_kurtosis(operations, localBags) -> list:
    """Returns the stats about kurtosis of due dates across all local bags 
    """
    kurtosis_values = []
    for bag in localBags.values():
        kurtosis_values.append(compute_duedate_kurtosis_of_bag(operations, bag))
    return get_descriptive_stats(kurtosis_values, ['min', 'max', 'mean', 'std'])

def compute_local_ratio_of_setup_to_test_time(data, operations, configurations, localBags) -> list:
    """Returns the stats about ratio of total setup to total test times across all bags

    For calculating the total setup and test times in a local bag, we assume that 
    - we process every operation under it's most optimal configuration (optimal -> least test time)
    - we switch configuration when we complete all the operations which are supposed to be tested under that configuration.
    - Instead of estimating the ordering among available configs, we use avg setup times of each configuration as it's setup time
    """
    ratios = []
    for testerName, bag in localBags.items():
        total_setup, total_test_time = get_total_setup_and_total_test_time_of_local_bag(data, operations, configurations, bag, testerName)
        ratios.append(0 if total_test_time == 0 else total_setup/total_test_time)
    return get_descriptive_stats(ratios, ['min', 'max', 'mean', 'std'])

def compute_best_config_for_operation_on_a_tester(data, operations, operationName, testerName) -> str:
    """Returns the minimum test time config for a given operation. If there are multiple, returns one by choosing randomly.
    """
    opType = operations[operationName]['opType']
    compatible_configs = set(data['operations']['items'][opType]['estimatedTestTime'])
    tester_configs = set(data['testers']['items'][testerName]['supportedConfigurations'])
    configs = compatible_configs.intersection(tester_configs)
    return min(configs, key=lambda x: average(data['operations']['items'][opType]['estimatedTestTime'][x]))
    # return random.choice([key for (key, value) in configs.items() if average(value) == average(min_test_time_config)])

##############################################################################################################################

def compute_local_bag_num_of_ops(localBag) -> list:
    """Returns the number of operations in the given local bag
    """
    return [localBag.size()]

def compute_local_bag_total_setup_plus_test_time(data, operations, configurations, localBag, testerName) -> list:
    """Returns the total setup plus test time of the local bag
    """
    return [compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, localBag, testerName)]

def compute_local_bag_due_date(operations, localBag) -> list:
    """Returns the stats about due dates in a local bag
    """
    due_dates = collect_due_dates_of_bag(operations, localBag)
    return get_descriptive_stats(due_dates, ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis'])

def compute_local_bag_slack(data, jobs, operations, localBag, current_time) -> list:
    """Returns the stats about slack of opeations in the local bag
    """
    slacks = []
    for op in localBag:
        due_date = operations[op]['duedate']
        test_time = operations[op]['averageTestTime']
        slacks.append(due_date - (current_time + test_time))
    return get_descriptive_stats(slacks, ['min', 'max', 'mean', 'std', 'skewness', 'kurtosis'])

def compute_local_bag_out_degree(operations, localBag) -> list:
    """Returns the stats about out-degree of operations in the local bag
    """
    out_degrees = []
    for op in localBag:
        out_degrees.append(operations[op]['outdegree'])
    return get_descriptive_stats(out_degrees, ['min', 'max', 'mean', 'std'])
    
def compute_local_bag_remaining_test_time_of_jobs(jobs, operations, localBag, current_time) -> list:
    """Returns the stats about remaining test time of jobs corresponding to operations in the local bag
    """
    remaining_test_times = compute_remaining_test_times_of_jobs_corresponding_to_ops_in_bag(jobs, operations, localBag, current_time)
    return get_descriptive_stats(remaining_test_times, ['min', 'max', 'mean', 'std'])

def compute_local_bag_remaining_num_of_ops_of_jobs(jobs, operations, localBag) -> list:
    """Returns the stats about remaining number of operations for jobs corresponding to operations in the local bag
    """
    remaining_no_of_ops = compute_remaining_number_of_ops_for_jobs_in_bag(jobs, operations, localBag)
    return get_descriptive_stats(remaining_no_of_ops, ['min', 'max', 'mean', 'std'])

##############################################################################################################################

def contains_invalid_numbers(lst):
    for value in lst:
        # Check for None and NaN using numpy.isnan
        if value is None or np.isnan(value):
            return True
            
        # Check for NaN, inf, and -inf using math.isinf and math.isnan
        if math.isinf(value) or math.isnan(value):
            return True

    return False

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