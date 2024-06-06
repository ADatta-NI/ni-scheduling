from collections import OrderedDict
import obs_utils
import numpy as np

from constants import OperationStatus

"""Set of utility functions used by Scheduling Environment for performing tasks related to action from agent
"""

def normalize_dict_positives_high_is_better(scores: dict) -> None:
    """Normalizes the values in the dictionary such that in resulting dict, the values are in the range [0, 1] and all the values sum up to 1.

    - Assumes that all the values are non-negative.
    - Gives higher normalized value to higher values. This is calculated as below:
        res_i = (val_i / sum(vals))
    """
    if len(scores) == 0:
        return scores

    total = sum(scores.values())    
    num_of_values = len(scores)    

    if num_of_values > 1:
        if total != 0:
            scores.update((key, value / total) for key, value in scores.items())
        else:
            scores.update((key, 1 / num_of_values) for key in scores.keys())
    else:
        scores.update((key, 1) for key in scores.keys())


def normalize_dict_positives_low_is_better(scores: dict) -> dict:
    """Normalizes the values in the dictionary such that in resulting dict, the values are in the range [0, 1] and all the values sum up to 1. Lesser values get higher weight.

    - Assumes that all the values are non-negative.
    - Gives higher normalized value to lower values. This is calculated as below:
        res_i = (1 - (val_i / sum(vals))) / (n-1)
    """
    if len(scores) == 0:
        return scores
    
    total = sum(scores.values())
    num_of_values = len(scores)

    if num_of_values > 1:
        if total != 0:
            scores.update((key, (1 - (value / total)) / (num_of_values - 1)) for key, value in scores.items())
        else:
            scores.update((key, 1 / num_of_values) for key in scores.keys())
    else:
        scores.update((key, 1) for key in scores.keys())

def normalize_dict_pos_neg_high_is_better(scores: dict) -> dict:
    """Normalizes the values in the dictionary such that in resulting dict, the values are in the range [0, 1] and all values sum up to 1
    
    - Works with both positive and negative values
    - If atleast one value is negative, shifts all the numbers to right such that the least number is 0.
    - Then performs regular weighted normalization
    """
    if len(scores) == 0:
        return scores
    
    min_value = min(scores.values())

    if min_value < 0:
        scores.update((key, value - min_value) for key, value in scores.items())

    normalize_dict_positives_high_is_better(scores)


def normalize_dict_pos_neg_low_is_better(scores: dict) -> dict:
    """Normalizes the values in the dictionary such that in resulting dict, the values are in the range [0, 1] and all values sum up to 1
    
    - Works with both positive and negative values
    - If atleast one value is negative, shifts all the numbers to right such that the least number is 0.
    - Then performs normalization such that lower values get higher normalized value. This is calculated as below:
        res_i = (1 - (val_i / sum(vals))) / (n-1)
    """
    if len(scores) == 0:
        return scores
    
    min_value = min(scores.values())

    if min_value < 0:
        scores.update((key, value - min_value) for key, value in scores.items())

    normalize_dict_positives_low_is_better(scores)


def _normalize_list(values):
    """Linearly normalizes a list of values. 
    
    - Assumes all the values are > 0. 
    - If all the values are zeros, no-op 
    """
    total = sum(values)
    if total == 0:
        return values
    return [value / total for value in values]

##############################################################################################################################    

def check_operation_tester_feasibility(data, jobs, operations, testers, configurations, opName, testerName) -> bool:
    """Checks if an operation is feasible to be tested on a tester. 
    Returns:
    - True, if feasible
    - False, if not feasible

    Feasibility Conditions:
    - Tester has atleast one supported configuration which is compatible to test the operation.
    """
    opType = operations[opName]['opType']
    op_config_set = set(data['operations']['items'][opType]['compatibleConfigurations'])
    tester_config_set = set(data['testers']['items'][testerName]['supportedConfigurations'])
    feasible_config_set = op_config_set.intersection(tester_config_set)

    # Check if operation have atleast one compatible configuration suitable on this tester
    if len(feasible_config_set) == 0:
        return False
    
    return True


def compute_msagent_dr_least_setup_plus_test_time(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the setup plus test time of an operation on a tester

    - We compute the feasible set of configurations by taking intersection of operation vs tester config sets
    - We take the average of setup and test times across all feasible configurations
    - To compute the setup of configuration and test time of an operation on that config, we take avg. setup of that config, avg. test time of operation on that config.
    """
    opType = operations[opName]['opType']
    op_config_set = set(data['operations']['items'][opType]['compatibleConfigurations'])
    tester_config_set = set(data['testers']['items'][testerName]['supportedConfigurations'])
    feasible_config_set = op_config_set.intersection(tester_config_set)

    # This check must have already been performed before, so code never passes this if-check.
    if len(feasible_config_set) == 0:
        return np.inf

    avg_setup_plus_test_times = []
    for config in feasible_config_set:
        avg_setup_time = configurations[config]['averageSetupTime']
        avg_test_time = obs_utils.average(data['operations']['items'][opType]['estimatedTestTime'][config])
        avg_setup_plus_test_times.append(avg_setup_time + avg_test_time)
    return np.average(avg_setup_plus_test_times)


def compute_msagent_dr_least_total_setup_plus_test_time_of_local_bag(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the total setup plus test time of corresponding local bag
    """
    return obs_utils.compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, localBags[testerName], testerName)


def compute_msagent_dr_least_num_of_operations_in_local_bag(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the number of operations in corresponding local bag
    """
    return obs_utils.compute_local_bag_num_of_ops(localBags[testerName])[0]


def compute_msagent_dr_most_similar_operations_in_local_bag(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the number of similar operations in the local bag for a given operation.

    Similarity is calculated as an average of number of common configurations between the given operation and each of the operation in local bag. 
    """
    similarties = []
    target_op_config_set = set(data['operations']['items'][operations[opName]['opType']]['compatibleConfigurations'])
    for op in localBags[testerName]:
        op_config_set = set(data['operations']['items'][operations[op]['opType']]['compatibleConfigurations'])
        common_config_set = target_op_config_set.intersection(op_config_set)
        similarties.append(len(common_config_set))
    return len(target_op_config_set) if len(similarties) == 0 else np.average(similarties)


def compute_msagent_dr_due_date_skewness_decrement(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the change in skewness of the duedate distribution from before and after adding the operation to the localBag
            
    (before adding - after adding) -> +ve change is good.
    """
    due_dates = obs_utils.collect_due_dates_of_bag(operations, localBags[testerName])
    before_skewness = obs_utils.get_descriptive_stats(due_dates, ['skewness'])[0]

    op_due_date = operations[opName]['duedate']
    due_dates.append(op_due_date)
    after_skewness = obs_utils.get_descriptive_stats(due_dates, ['skewness'])[0]

    return before_skewness - after_skewness


def compute_msagent_dr_due_date_kurtosis_decrement(data, jobs, operations, testers, configurations, localBags, opName, testerName) -> float:
    """Returns the change in kurtosis of the duedate distribution from before and after adding the operation to the localBag
            
    (before adding - after adding) -> +ve change is good.
    """
    due_dates = obs_utils.collect_due_dates_of_bag(operations, localBags[testerName])
    before_kurtosis = obs_utils.get_descriptive_stats(due_dates, ['kurtosis'])[0]

    op_due_date = operations[opName]['duedate']
    due_dates.append(op_due_date)
    after_kurtosis = obs_utils.get_descriptive_stats(due_dates, ['kurtosis'])[0]

    return before_kurtosis - after_kurtosis


def compute_msagent_dispatching_rule_for_all_testers(data, jobs, operations, testers, configurations, localBags, opName, dispatching_rule_name, weight) -> dict:
    """Compute normalized, weight multiplied priority scores of all testers for the given operation

    - Performs check on whether the given operation is feasible to be tested on a given tester.
    - If non-feasible, score for that tester will be returned as -INF
    """
    scores = {}
    non_feasible_testers = set()
    dr_func, normalization_func = MS_AGENT_DISPATCHING_RULES[dispatching_rule_name] 
    for tester in testers.keys():
        # Check tester feasibility and skip calculating score for it
        if check_operation_tester_feasibility(data, jobs, operations, testers, configurations, opName, tester) == False:
            non_feasible_testers.add(tester)
            continue

        scores[tester] = dr_func(data, jobs, operations, testers, configurations, localBags, opName, tester)
    
    #normalize the dict
    normalization_func(scores)

    #multiply by corresponding weight
    scores.update((key, weight * value) for key, value in scores.items())

    # Make skipped testers score as -np.inf
    for tester in non_feasible_testers:
        scores[tester] = -np.inf
    
    return scores


def select_msagent_tester_for_given_operation(data, jobs, operations, testers, configurations, localBags, opName, weights) -> str:
    """Returns the highest priority tester for the given operation and corresponding weights 
    """
    scores = {}
    for testerName in testers.keys():
        scores[testerName] = 0

    # Linearly normalizing the weights in case the weights provided doesn't add up to 1. Assumes that all values are >= 0.
    # weights = _normalize_list(weights)

    for i, dispatching_rule_name in enumerate(MS_AGENT_DISPATCHING_RULES.keys()):
        score_dict = compute_msagent_dispatching_rule_for_all_testers(data, jobs, operations, testers, configurations, localBags, opName, dispatching_rule_name, weights[i])
        for testerName in score_dict.keys():
            scores[testerName] += score_dict[testerName]

    return max(scores, key=scores.get)


MS_AGENT_DISPATCHING_RULES = OrderedDict(
    LEAST_SETUP_PLUS_TEST_TIME_OF_OPERATION = (compute_msagent_dr_least_setup_plus_test_time, normalize_dict_positives_low_is_better),
    LEAST_TOTAL_SETUP_PLUS_TEST_TIME_OF_ALL_OPS_IN_LOCAL_BAG = (compute_msagent_dr_least_total_setup_plus_test_time_of_local_bag, normalize_dict_positives_low_is_better),
    LEAST_NUM_OF_JOBS_IN_LOCAL_BAG = (compute_msagent_dr_least_num_of_operations_in_local_bag, normalize_dict_positives_low_is_better),
    MOST_SIMILAR_OPS_IN_LOCAL_BAG = (compute_msagent_dr_most_similar_operations_in_local_bag, normalize_dict_positives_high_is_better),
    DUE_DATE_SKEWNESS_DECREASES = (compute_msagent_dr_due_date_skewness_decrement, normalize_dict_pos_neg_high_is_better),
    DUE_DATE_KURTOSIS_DECREASES = (compute_msagent_dr_due_date_kurtosis_decrement, normalize_dict_pos_neg_high_is_better)
)

##############################################################################################################################

def compute_osagent_dr_earliest_due_date(data, jobs, operations, testers, configurations, localBags, testerName, op, currentTime) -> float:
    """Returns the due date of the operation
    """
    return operations[op]['duedate']


def compute_osagent_dr_least_slack(data, jobs, operations, testers, configurations, localBags, testerName, opName, currentTime) -> float:
    """Returns the available slack of the corresponding job
    """
    blocked_ops_setup_plus_test_time = 0
    jobName = operations[opName]['jobName']
    productName = jobs[jobName]['productName']
    duedate = data['products']['items'][productName]['duedate']
    for op in jobs[jobName]['operations']:
        if operations[op]['status'] == OperationStatus.NOT_STARTED:
            blocked_ops_setup_plus_test_time += (operations[op]['averageSetupTime'] + operations[op]['averageTestTime'])

    op_setup_plus_test_time = operations[opName]['averageSetupTime'] + operations[opName]['averageTestTime']
    return duedate - (currentTime + op_setup_plus_test_time + blocked_ops_setup_plus_test_time)


def compute_osagent_dr_least_setup_plus_test_time(data, jobs, operations, testers, configurations, localBags, testerName, opName, currentTime) -> float:
    """Returns the least setup plust test time of an operation on a tester.

    - First, finds the intersection of config sets of tester and operation
    - Calculates setup time component by considering the setup change from current machine configuration to the configuration under consideration from the common config set.
    """
    opType = operations[opName]['opType']
    op_config_set = set(data['operations']['items'][opType]['compatibleConfigurations'])
    tester_config_set = set(data['testers']['items'][testerName]['supportedConfigurations'])
    feasible_config_set = op_config_set.intersection(tester_config_set)
    current_tester_config = testers[testerName]['currentConfiguration']
    current_tester_config_index = data['configurations']['items'][current_tester_config]['index']

    setup_plus_test_times = []
    for config in feasible_config_set:
        setup_time = data['configurations']['items'][config]['setupTimes'][current_tester_config_index]
        test_time = obs_utils.average(data['operations']['items'][opType]['estimatedTestTime'][config])
        setup_plus_test_times.append(setup_time + test_time)

    return min(setup_plus_test_times)


def compute_osagent_dr_least_remaining_test_time_of_job(data, jobs, operations, testers, configurations, localBags, testerName, opName, time) -> float:
    """Returns the remaining test time of the job corresponding to current operation.
    """
    jobName = operations[opName]['jobName']
    return obs_utils.compute_remaining_setup_plus_test_time_of_job(jobs, operations, jobName, time)


def compute_osagent_dr_least_remaining_num_of_ops_of_job(data, jobs, operations, testers, configurations, localBags, testerName, opName, time) -> float:
    """Returns the remaining number of operations of job corresponding to current operation.
    """
    jobName = operations[opName]['jobName']
    return obs_utils.compute_remaining_number_of_ops_for_job(jobs, operations, jobName)


def compute_osagent_dr_max_out_degree_of_op(data, jobs, operations, testers, configurations, localBags, testerName, opName, time) -> float:
    """Returns the out degree of given operation
    """
    return operations[opName]['outdegree']


def compute_osagent_dispatching_rule_for_all_ops(data, jobs, operations, testers, configurations, localBags, testerName, dispatching_rule_name, weight, time) -> dict:
    """Compute normalized, weight multiplied priority scores of all operations in a local bag
    """
    scores = {}
    dr_func, normalization_func = OS_AGENT_DISPATCHING_RULES[dispatching_rule_name]
    for op in localBags[testerName]:
        scores[op] = dr_func(data, jobs, operations, testers, configurations, localBags, testerName, op, time)
    
    #normalize the dict
    normalization_func(scores)

    #multiply by corresponding weight
    scores.update((key, weight * value) for key, value in scores.items())

    return scores


def select_osagent_operation_for_given_tester(data, jobs, operations, testers, configurations, localBags, testerName, weights, time) -> str:
    """Returns the highest priority operation for the given tester and corresponding weights
    """
    scores = {}
    for op in localBags[testerName]:
        scores[op] = 0

    # Linearly normalizing the weights in case the weights provided doesn't add up to 1. Assumes that all values are >= 0.
    # weights = _normalize_list(weights)

    for i, dispatching_rule_name in enumerate(OS_AGENT_DISPATCHING_RULES.keys()):
        score_dict = compute_osagent_dispatching_rule_for_all_ops(data, jobs, operations, testers, configurations, localBags, testerName, dispatching_rule_name, weights[i], time)
        for op in score_dict.keys():
            scores[op] += score_dict[op]

    return max(scores, key=scores.get)


OS_AGENT_DISPATCHING_RULES = OrderedDict(
    EARLIEST_DUE_DATE = (compute_osagent_dr_earliest_due_date, normalize_dict_positives_low_is_better),
    LEAST_SLACK = (compute_osagent_dr_least_slack, normalize_dict_pos_neg_low_is_better),
    LEAST_SETUP_PLUS_TEST_TIME = (compute_osagent_dr_least_setup_plus_test_time, normalize_dict_positives_low_is_better),
    LEAST_REMAINING_TEST_TIME_FOR_CORRESPONDING_JOB = (compute_osagent_dr_least_remaining_test_time_of_job, normalize_dict_positives_low_is_better),
    LEAST_REMAINING_NUM_OF_OPS_FOR_CORRESPONDING_JOB = (compute_osagent_dr_least_remaining_num_of_ops_of_job, normalize_dict_positives_low_is_better),
    MAX_OUT_DEGREE_IN_CORRESPONDING_DEP_GRAPH = (compute_osagent_dr_max_out_degree_of_op, normalize_dict_positives_high_is_better)
)

##############################################################################################################################

def number_of_ops_in_local_bag(data, jobs, operations, testers, configurations, localBags, opName, testerName):
    return obs_utils.compute_local_bag_num_of_ops(localBags[testerName])[0]

def load_of_tester(data, jobs, operations, testers, configurations, localBags, opName, testerName):
    return obs_utils.compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, localBags[testerName], testerName)

def select_msagent_tester_for_given_operation_following_greedy_strategy(data, jobs, operations, testers, configurations, localBags, opName, rule) -> str:
    """Returns the optimal tester as per the greedy strategy rule
    """
    rule_fn, choose_fn = MS_AGENT_GREEDY_RULES[rule]

    scores = {}
    for testerName in testers.keys():
        # Check tester feasibility for this op
        if check_operation_tester_feasibility(data, jobs, operations, testers, configurations, opName, testerName) == False:
            if choose_fn == max: scores[testerName] = -np.inf
            else: scores[testerName] = np.inf
        else:
            scores[testerName] = rule_fn(data, jobs, operations, testers, configurations, localBags, opName, testerName)

    return choose_fn(scores, key=scores.get)

MS_AGENT_GREEDY_RULES = OrderedDict({
    "least-num-ops": (number_of_ops_in_local_bag, min),
    'least-load': (load_of_tester, min)
})

##############################################################################################################################

def setup_plus_test_time_of_op(data, jobs, operations, testers, configurations, localBags, testerName, time, op):
    best_config = obs_utils.compute_best_config_for_operation_on_a_tester(data, operations, op, testerName)
    avg_test_time = obs_utils.average(data['operations']['items'][operations[op]['opType']]['estimatedTestTime'][best_config])
    avg_setup_time = configurations[best_config]['averageSetupTime']
    return avg_setup_time + avg_test_time

def duedate_of_op(data, jobs, operations, testers, configurations, localBags, testerName, time, op):
    return operations[op]['duedate']

def select_osagent_operation_for_given_tester_following_greedy_strategy(data, jobs, operations, testers, configurations, localBags, testerName, rule, time):
    """Returns the optimal operation for the given tester as per the greedy strategy rule
    """
    rule_fn, choose_fn = OS_AGENT_GREEDY_RULES[rule]

    scores = {}
    for op in localBags[testerName]:
        scores[op] = rule_fn(data, jobs, operations, testers, configurations, localBags, testerName, time, op)

    return choose_fn(scores, key=scores.get)

OS_AGENT_GREEDY_RULES = OrderedDict({
    "least-test-time": (setup_plus_test_time_of_op, min),
    'nearest-due-date': (duedate_of_op, min)
})
