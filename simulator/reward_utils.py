import math
import obs_utils
from constants import OperationStatus
from statistics import median,mean
"""Set of utility functions used by Scheduling Environment for performing tasks related to reward calculations
"""

def compute_relative_completion_time_of_bag(data, operations, configurations, bag, testerName) -> float:
    """Returns the completion time of a bag. Essentially the total setup + test time of all the operations in the bag. 
    
    For calculating the total setup and test times in a local bag, we assume that 
    - we process every operation under it's most optimal configuration (optimal -> least test time)
    - we switch configuration when we complete all the operations which are supposed to be tested under that configuration.
    - Instead of estimating the ordering among available configs, we use avg setup times of each configuration as it's setup time
    """
    return obs_utils.compute_total_setup_plus_test_time_of_local_bag(data, operations, configurations, bag, testerName)


def compute_msagent_reward_max_diff_between_completion_times(data, operations, configurations, localBags) -> float:
    """Returns the maximum difference between the completion times across all local bags
    """
    completion_times = []
    for testerName, bag in localBags.items():
        completion_times.append(compute_relative_completion_time_of_bag(data, operations, configurations, bag, testerName))
    return max(completion_times) - min(completion_times)


def compute_msagent_reward_max_difference_in_total_setups_before_and_after_state_change(data, operations, localBags, prevStateAttributes) -> float:
    """Returns the maximum change in total setups to complete all the ops in a bag across all bags as a result of state change
    """
    setup_change = []
    for testerName, bag in localBags.items():
        before_setups = prevStateAttributes['setUps'][testerName]
        after_setups = obs_utils.compute_number_of_setups_of_bag(data, operations, bag, testerName)
        prevStateAttributes['setUps'][testerName] = after_setups
        setup_change.append(after_setups - before_setups)
    return max(setup_change)


def compute_msagent_reward_maximum_duedate_skewness_across_all_testers(operations, localBags) -> float:
    """Returns the maximum skewness in duedate distributions across all local bags
    """
    skewness_values = []
    for bag in localBags.values():
        skewness_values.append(obs_utils.compute_duedate_skewness_of_bag(operations, bag))
    return max(skewness_values)

def compute_msagent_reward_skewness_at_tester(operations, localbag) -> float:
    return obs_utils.compute_duedate_skewness_of_bag(operations, localbag)

def compute_msagent_reward_kurtosis_at_tester(operations, localbag) -> float:
    return obs_utils.compute_duedate_kurtosis_of_bag(operations, localbag)

def compute_msagent_reward_total_setups_difference_at_tester(data, operations, localbags, prevStateAttributes, testerName) -> float:
    before_setups = prevStateAttributes['setUps'][testerName]
    after_setups = obs_utils.compute_number_of_setups_of_bag(data, operations, localbags[testerName], testerName)
    prevStateAttributes['setUps'][testerName] = after_setups
    return before_setups - after_setups

def compute_msagent_reward_difference_in_max_duedate_and_completion_time(data, operations, configurations, localbags, testerName) -> float:
    max_duedate = 0
    completion_time = compute_relative_completion_time_of_bag(data, operations, configurations, localbags[testerName], testerName)
    for op in localbags[testerName]:
        duedate = operations[op]['duedate']
        max_duedate = max(max_duedate, duedate)
    return max_duedate - completion_time

def compute_msagent_reward_skewness_difference_at_tester(operations, localbag, prevStateAttributes, testerName) -> float:
    before_skewness = prevStateAttributes['skewness'][testerName]
    after_skewness = obs_utils.compute_duedate_skewness_of_bag(operations, localbag) 
    prevStateAttributes['skewness'][testerName] = after_skewness
    return before_skewness - after_skewness

def compute_msagent_reward_kurtosis_difference_at_tester(operations, localbag, prevStateAttributes, testerName) -> float:
    before_kurtosis = prevStateAttributes['kurtosis'][testerName]
    after_kurtosis = obs_utils.compute_duedate_kurtosis_of_bag(operations, localbag) 
    prevStateAttributes['kurtosis'][testerName] = after_kurtosis
    return before_kurtosis - after_kurtosis

def compute_msagent_reward_maximum_duedate_kurtosis_across_all_testers(operations, localBags) -> float:
    """Returns the maximum kurtosis in duedate distributions across all local bags
    """
    kurtosis_values = []
    for bag in localBags.values():
        kurtosis_values.append(obs_utils.compute_duedate_kurtosis_of_bag(operations, bag))
    return max(kurtosis_values)


def compute_osagent_reward_tardiness_difference(data, jobs, operations, testers, configurations, prevStateAttributes, testerName, currentTime) -> float:
    """Returns the difference of tardiness of operations bef0re and after state change

    - We compute the due dates of operations by looking at the remaining operations for it's corresponding job, their setup+testtimes and job's duedate.
    - For each of the operation which were assigned to this tester, we sum the tardiness as below and return it.
        - If operation is DONE, tardiness = max(0, completion_time - due_date)
        - Else,
            - If current_time + operation_setup_plus_test_time <= operation_due_date, tardiness = 0
            - If current_time + operation_setup_plus_test_time > operation_due_date, tardiness is the corresponding difference
    """
    current_tardiness = 0
    for opName, opDetails in operations.items():
        if opDetails['assignedTesterName'] == testerName:
            duedate = opDetails['duedate']
            completion_time = None
            if opDetails['status'] == OperationStatus.COMPLETED or opDetails['status'] == OperationStatus.IN_PROGRESS:
                completion_time = opDetails['completionTime']
            elif opDetails['status'] == OperationStatus.IN_LOCAL_BAG:
                completion_time = currentTime + obs_utils.get_total_setup_and_test_time_of_op_on_a_tester(data, operations, testers, configurations, opName, testerName)

            current_tardiness += max(0, completion_time - duedate)

    prev_tardiness = prevStateAttributes['tardiness'][testerName]
    prevStateAttributes['tardiness'][testerName] = current_tardiness
    return prev_tardiness - current_tardiness

def compute_osagent_reward_median_difference(data, jobs, operations, testers, configurations, prevStateAttributes, testerName, currentTime) -> float:
    """Returns the difference of tardiness of operations bef0re and after state change

    - We compute the due dates of operations by looking at the remaining operations for it's corresponding job, their setup+testtimes and job's duedate.
    - For each of the operation which were assigned to this tester, we sum the tardiness as below and return it.
        - If operation is DONE, tardiness = max(0, completion_time - due_date)
        - Else,
            - If current_time + operation_setup_plus_test_time <= operation_due_date, tardiness = 0
            - If current_time + operation_setup_plus_test_time > operation_due_date, tardiness is the corresponding difference
    """
    current_tardiness = []
    current_median = 0
    for opName, opDetails in operations.items():
        if opDetails['assignedTesterName'] == testerName:
            duedate = opDetails['duedate']
            completion_time = None
            if opDetails['status'] == OperationStatus.COMPLETED or opDetails['status'] == OperationStatus.IN_PROGRESS:
                completion_time = opDetails['completionTime']
            elif opDetails['status'] == OperationStatus.IN_LOCAL_BAG:
                completion_time = currentTime + obs_utils.get_total_setup_and_test_time_of_op_on_a_tester(data, operations, testers, configurations, opName, testerName)

            current_tardiness.append(max(0, completion_time - duedate))
    print(len(current_tardiness))
    current_median = median(current_tardiness)

    prev_median = prevStateAttributes['tardiness'][testerName]
    prevStateAttributes['tardiness'][testerName] = current_median
    return prev_median - current_median

def compute_osagent_reward_makespan_difference(data, jobs, operations, testers, configurations, prevStateAttributes, testerName, currentTime) -> float:
    """Returns the difference of tardiness of operations bef0re and after state change

    - We compute the due dates of operations by looking at the remaining operations for it's corresponding job, their setup+testtimes and job's duedate.
    - For each of the operation which were assigned to this tester, we sum the tardiness as below and return it.
        - If operation is DONE, tardiness = max(0, completion_time - due_date)
        - Else,
            - If current_time + operation_setup_plus_test_time <= operation_due_date, tardiness = 0
            - If current_time + operation_setup_plus_test_time > operation_due_date, tardiness is the corresponding difference
    """
    current_makespan = []
    current_max = 0
    for opName, opDetails in operations.items():
        if opDetails['assignedTesterName'] == testerName:
            duedate = opDetails['duedate']
            completion_time = None
            if opDetails['status'] == OperationStatus.COMPLETED or opDetails['status'] == OperationStatus.IN_PROGRESS:
                completion_time = opDetails['completionTime']
            elif opDetails['status'] == OperationStatus.IN_LOCAL_BAG:
                completion_time = currentTime + obs_utils.get_total_setup_and_test_time_of_op_on_a_tester(data, operations, testers, configurations, opName, testerName)

            current_makespan.append(completion_time)

    current_max = max(current_makespan)
    print(len(current_makespan))

    prev_max = prevStateAttributes['tardiness'][testerName]
    prevStateAttributes['tardiness'][testerName] = current_max
    return prev_max - current_max

def compute_osagent_reward_mean_difference(data, jobs, operations, testers, configurations, prevStateAttributes, testerName, currentTime) -> float:
    """Returns the difference of tardiness of operations bef0re and after state change

    - We compute the due dates of operations by looking at the remaining operations for it's corresponding job, their setup+testtimes and job's duedate.
    - For each of the operation which were assigned to this tester, we sum the tardiness as below and return it.
        - If operation is DONE, tardiness = max(0, completion_time - due_date)
        - Else,
            - If current_time + operation_setup_plus_test_time <= operation_due_date, tardiness = 0
            - If current_time + operation_setup_plus_test_time > operation_due_date, tardiness is the corresponding difference
    """
    current_tardiness = []
    current_mean = 0
    for opName, opDetails in operations.items():
        if opDetails['assignedTesterName'] == testerName:
            duedate = opDetails['duedate']
            completion_time = None
            if opDetails['status'] == OperationStatus.COMPLETED or opDetails['status'] == OperationStatus.IN_PROGRESS:
                completion_time = opDetails['completionTime']
            elif opDetails['status'] == OperationStatus.IN_LOCAL_BAG:
                completion_time = currentTime + obs_utils.get_total_setup_and_test_time_of_op_on_a_tester(data, operations, testers, configurations, opName, testerName)

            current_tardiness.append(max(0, completion_time - duedate))
    print(len(current_tardiness))
    current_mean = mean(current_tardiness)

    prev_mean = prevStateAttributes['tardiness'][testerName]
    prevStateAttributes['tardiness'][testerName] = current_mean
    return prev_mean - current_mean




def compute_sigmoid(x):
  if x > 16: return 1
  if x < -16: return 0
  return 1 / (1 + math.exp(-x))

def hamacher_product(a, b):
    """The hamacher (t-norm) product of a and b.

    computes (a * b) / ((a + b) - (a * b))

    Args:
        a (float): 1st term of hamacher product.
        b (float): 2nd term of hamacher product.

    Raises:
        ValueError: a and b must range between 0 and 1

    Returns:
        float: The hammacher product of a and b
    """
    if not ((0.0 <= a <= 1.0) and (0.0 <= b <= 1.0)):
        raise ValueError("a and b must range between 0 and 1")

    denominator = a + b - (a * b)
    h_prod = ((a * b) / denominator) if denominator > 0 else 0

    assert 0.0 <= h_prod <= 1.0
    return h_prod