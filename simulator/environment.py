"""
Scheduling Environment following the Partially Observable Stochastic Game (POSG) model, developed 
using RLLib's MultiAgentEnv interface.

Provides interactions for m+1 agents, where there are: 
1 -> Machine Selection Agent (MSAgent)
m -> Operation Sequencing Agents (OSAgent)


Using Simplex action spaces for the OS Agent and MS Agent for actions between [0,1] and summing to 1 
"""

from typing import Tuple, Dict
import gymnasium as gym
import numpy as np
import ray
from ray.rllib.utils.spaces import simplex
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

import glob
import numpy as np
import json
import random
import math

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from constants import (
    MS_AGENT, OS_AGENT_PREFIX, HYPHEN, SCHEDULING_ENV_RANDOM_SEED,
    JobStatus, OperationStatus, TesterStatus, TimeJumpReason,
)
from bag import Bag
import obs_utils, action_utils, reward_utils

class SchedulingEnv(MultiAgentEnv):
    """A multi-agent scheduling environment
    
    A particular scheduling problem instance is encoded as a config json, which is parsed and 
    instantiated in the environment during initialization.
    """

    def __init__(self, config=None):
        # Set Random Seed for reproducibility
        random.seed(SCHEDULING_ENV_RANDOM_SEED)

        # Parse the scheduling problem instance into environment attributes        
        config = config or {}
        self.config = config
        self.data = self._get_static_config_data(self.config)
        self.staticConfigFileName = self.config.get("staticConfigurationFilePath")
        self.scName = self.staticConfigFileName.removeprefix("data/").removesuffix(".json")
        self.maxSteps = self.config.get("maxSteps")
        self.setupRendering = self.config.get("setupRendering")
        self.sampleRandomProblem = self.config.get("sampleRandomProblem")
        self.followGreedyStrategy = self.config.get("followGreedyStrategy")
        self.msGreedyStrategy = self.config.get("msGreedyStrategy")
        self.osGreedyStrategy = self.config.get("osGreedyStrategy")

        # Agents
        self.agents = {MS_AGENT}
        for testerName in self.data['testers']['items'].keys():
            self.agents.add(OS_AGENT_PREFIX + testerName)
        self._agent_ids = set(self.agents)

        # Observation Space
        self._obs_space_in_preferred_format = True
        self.observation_space = gym.spaces.Dict({
            MS_AGENT: gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(52,), dtype=np.float32)
        })
        for testerName in self.data['testers']['items'].keys():
            self.observation_space[OS_AGENT_PREFIX + testerName] = gym.spaces.Box(low=np.float32(-np.inf), high=np.float32(np.inf), shape=(26,), dtype=np.float32) 

        # Action Space
        self._action_space_in_preferred_format = True
        self.action_space = gym.spaces.Dict({
            MS_AGENT: simplex.Simplex(shape=(6,)) ,
        })
        for testerName in self.data['testers']['items'].keys():
            self.action_space[OS_AGENT_PREFIX + testerName] = simplex.Simplex(shape=(6,))

        super().__init__()

    
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Resets the environment to initial state and returns 2 agent-id keyed dicts
        
        Returns:
            observations dict
            infos dict
        """
        if(self.sampleRandomProblem):
            self.config['staticConfigurationFilePath'] = random.choice(glob.glob('data/static_configuration_*.json'))
            self.__init__(self.config)

        print("Env initiated with ", self.staticConfigFileName)

        # Initialize internal state
        self._init_internal_state(self.data)

        # Rendering initialization
        if self.setupRendering == True:
            font = {'size': 16}
            mpl.rc('font', **font)
            plt.figure(figsize=(16, 9), dpi=80)
            plt.xlabel('time')
            plt.ylabel('testers')
            plt.title('Test Schedule')

            for tester in self.testers.keys():
                plt.barh(y=tester, width=0, left=0, height=0.1, color='white')

        self.terminateds = set()
        self.truncateds = set()

        # Build and return observations
        observations = self._build_observations_dict()

        for agent, observation in observations.items():
            for obs in observation:
                if obs is None or math.isnan(obs) or obs == float('inf') or obs == float('-inf'):
                    raise Exception('Observation calculation invalid value', agent, observation)

        infos = {}

        print("In Reset: ")
        print("Observations: ", observations)

        return observations, infos
    

    def step(self, actions: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Steps the environment by executing the actions provided and returns 5 agent-id keyed dicts

        Returns:
            observations dict
            rewards dict
            terminateds dict
            truncateds dict
            infos dict
        """
        #print("In Step: ")
        #print("Actions: ", actions)


        #print('items in Global bag')
        #self.globalBag.show()

        for item in self.localBags.values():
            #print('items in bag')
            item.show()


        observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        self.steps += 1

        # Apply actions and update state
        for agent, action in actions.items():
            for weight in action:
                if weight is None or math.isnan(weight) or weight == float('inf') or weight == float('-inf'):
                    raise Exception('Action weight invalid value', agent, action)
            
        for agentName, action in actions.items():
            if agentName == MS_AGENT:
                testerName = self._handle_msagent_action(action)
            else:
                self._handle_osagent_action(agentName, action)
        
        # Compute Rewards
        for agentName in actions.keys():
            if agentName == MS_AGENT:
                rewards[agentName] = self._calculate_msagent_reward(testerName)
            else:
                rewards[agentName] = self._calculate_osagent_reward(agentName)

        for agent, reward in rewards.items():
            if reward is None or math.isnan(reward) or reward == float('inf') or reward == float('-inf'):
                raise Exception('Reward invalid value', agent, reward)

        # for agentName, reward in rewards.items():
        #     rewards[agentName] = self._transform_rew(reward, agentName)

            # if math.isnan(rewards[agentName]) or rewards[agentName] == float('inf') or rewards[agentName] == float('-inf'):
            #     raise Exception('OSAgent transformed Reward invalid value: ', rewards[agentName], "it is originally: ",reward)

        print("Rewards: ", rewards)

        # Time travel
        #print('time_before_jump', self.time)
        self.time, jump_reason = self._find_best_possible_time_jump()
        #print('time_after_jump', self.time)

        # Check if any product arrived at this time and add corresponding root operations to global bag
        self._check_and_add_products_to_global_bag()
        
        # add job setup time here so that it can be collected later 
        # visualise all possible time related variables 
        # cyclomatic complexity of the dependency trees
        # no of jobs count of job names 
        # ratio of longest to shortest test times 
        # variance of test times 
        # tightness of the cap for the resources 
        # arrival time 
        # 
        
        # Update operation and job progress/completion
        for testerName, testerDetails in self.testers.items():
            if testerDetails['testerStatus'] == TesterStatus.BUSY:
                currentOp = testerDetails['currentOperationName']
                names = currentOp.split(HYPHEN)
                jobName = names[0] + HYPHEN + names[1]
                # Check for operation completion
                if self.operations[currentOp]['completionTime'] <= self.time:
                    # Mark operation, job and tester status
                    self.operations[currentOp]['status'] = OperationStatus.COMPLETED
                    testerDetails['testerStatus'] = TesterStatus.IDLE
                    testerDetails['currentOperationName'] = None
                    self.jobs[jobName]['remainingNumOfOperations'] -= 1
                    self.jobs[jobName]['estimatedRemainingTime'] = obs_utils.compute_remaining_setup_plus_test_time_of_job(self.jobs, self.operations, jobName, self.time)
                    if self.jobs[jobName]['remainingNumOfOperations'] == 0:
                        self.jobs[jobName]['status'] = JobStatus.COMPLETED
                        self.jobs[jobName]['completionTime'] = self.operations[currentOp]['completionTime']
                    
                    # Unlock depending operations    
                    for childOpNum in self.operations[currentOp]['childOperations']:
                        childOpName = names[0] + HYPHEN + names[1] + HYPHEN + str(childOpNum)
                        self.operations[childOpName]['indegree'] -= 1
                        if self.operations[childOpName]['indegree'] == 0:
                            self._put_op_in_global_bag(childOpName)                   
                    

        # Update terminateds, truncateds, infos
        for agentName in actions.keys():
            terminateds[agentName] = False
            truncateds[agentName] = False
        terminateds["__all__"] = self._check_termination()
        truncateds["__all__"] = True if self.maxSteps != None and self.steps == self.maxSteps else False

        self.closeEnv = False
        if terminateds["__all__"] == True or truncateds['__all__'] == True:
            self.closeEnv = True
            infos["__common__"] = {
                "products": self.products,
                "jobs": self.jobs,
                "operations": self.operations,
                "testers": self.testers,
            }
        else:
            # Build observations
            observations = self._build_observations_dict()
            for agentName in observations.keys():
                infos[agentName] = {}

            for agent, observation in observations.items():
                for obs in observation:
                    if obs is None or math.isnan(obs) or obs == float('inf') or obs == float('-inf'):
                        raise Exception('Observation calculation invalid value', agent, observation)

        print("Observations: ", observations)
        if not observations:
            print("Terminateds: ", terminateds)
            print("Truncateds: ", truncateds)

        return observations, rewards, terminateds, truncateds, infos


    def render(self) -> None:
        """Renders the schedule as per the current state and actions history
        """
        setupCmap = cm.get_cmap('Dark2')
        testCmap = cm.get_cmap('Pastel1')
        setupNorm = colors.Normalize(vmin=0, vmax=self.data['configurations']['count'])
        testNorm = colors.Normalize(vmin=0, vmax=self.data['products']['count'])
        id = self.historyIdx
        for event in self.history[self.historyIdx:]:
            setupTime = self.operations[event['opId']]['setupTime']
            testTime = self.operations[event['opId']]['estimatedTestTime']
            configIdx = self.data['configurations']['items'][self.operations[event['opId']]['configurationUsed']]['index']
            productIdx = self.data['products']['items'][self.operations[event['opId']]['productName']]['index'] 
            
            plt.barh(y=event['testerName'], width=setupTime, left=event['time'], height=0.4, color=setupCmap(setupNorm(configIdx)))
            plt.barh(y=event['testerName'], width=testTime, left=event['time']+setupTime, height=0.2, color=testCmap(testNorm(productIdx)))
            plt.barh(y=event['testerName'], width=0.2, left=event['time']+setupTime+testTime, height=0.4, color='black')

            center_x = event['time'] + setupTime
            center_y = event['testerName']
            plt.text(x=center_x, y=center_y, s=event['opId'], ha='left', va='bottom', rotation='horizontal', fontdict={'size': 18, 'family': 'serif'})

            plt.pause(0.1)
            id += 1
    
        self.historyIdx = id

        if self.setupRendering == True and self.closeEnv == True:
            plt.savefig(self.staticConfigFileName.split('.')[0] + ('_greedy' if self.followGreedyStrategy else '_`model') + '.png')
            plt.close()

    def _check_and_add_products_to_global_bag(self):
        """Checks if any new products arrived at this time point and adds the corresponding root operations to Global Bag
        """
        for productName, productDetails in self.data['products']['items'].items():
            #print('before_adding',productName)
            arrivalTime = productDetails['arrival']

            #print('arrival',arrivalTime)
            if arrivalTime == self.time and self.products[productName]['started'] == False:
            #if not self.products[productName]['started']:
                #print('after_adding',productName)
                #print('time_after_jump',self.time)
                self.products[productName]['started'] = True
                for jobName in self.products[productName]['jobs']:
                    for opName in self.jobs[jobName]['operations']:
                        if self.operations[opName]['indegree'] == 0:
                            self._put_op_in_global_bag(opName)


    def _find_best_possible_time_jump(self) -> float:
        """Returns the maximum possible time for which nothing happens in the environment which needs simulation
        
        - If global bag is non-empty, we don't jump ahead until it becomes empty.
        - If a tester is IDLE and it's local bag is non-empty, we don't jump until we make it busy
        - The minimum duration for which all the testers will be in BUSY state
        - The arrival times of new products
        """

        if self.globalBag.size() > 0:
            return self.time, TimeJumpReason.NO_JUMP

        jump_times = []
        for testerName, testerDetails in self.testers.items():
            if testerDetails['testerStatus'] == TesterStatus.IDLE and self.localBags[testerName].size() > 0:
                return self.time, TimeJumpReason.NO_JUMP
            if testerDetails['testerStatus'] == TesterStatus.BUSY:
                jump_times.append((self.operations[testerDetails['currentOperationName']]['completionTime'], TimeJumpReason.TEST_DONE))
        
        for productName, productDetails in self.data['products']['items'].items():
            if productDetails['arrival'] > self.time:
                jump_times.append((productDetails['arrival'], TimeJumpReason.NEW_ARRIVAL))
        jump = min(jump_times, key=lambda x: x[0])
        #print(jump)
        return min(jump_times, key=lambda x: x[0])
    

    # def _calculate_msagent_reward(self) -> float:
    #     """Returns the reward for MSAgent. Calculated as below:

    #     Reward is the sum of:
    #     - | Max difference of completion times across all testers | 
    #     - | Max difference between total number of setups before and after state change across all testers |
    #     - maximum skewness across all testers
    #     - maximum kurtosis across all testers
    #     """

    #     a = reward_utils.compute_msagent_reward_max_diff_between_completion_times(self.data, self.operations, self.configurations, self.localBags)
    #     b = reward_utils.compute_msagent_reward_max_difference_in_total_setups_before_and_after_state_change(self.data, self.operations, self.localBags, self.prevStateAttributes)
    #     c = reward_utils.compute_msagent_reward_maximum_duedate_skewness_across_all_testers(self.operations, self.localBags)
    #     d = reward_utils.compute_msagent_reward_maximum_duedate_kurtosis_across_all_testers(self.operations, self.localBags)

    #     if math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d) or a == float('inf') or b == float('inf') or c == float('inf') or  d == float('inf') or a == float('-inf') or b == float('-inf') or c == float('-inf') or d == float('-inf'):
    #         raise Exception('MSAgent Reward invalid value: ', a, b, c, d)

    #     return -sum([a,
    #                  b,
    #                  c,
    #                  d])


    # def _calculate_msagent_reward(self, testerName) -> float:
    #     """Returns the reward for MSAgent. Calculated as below:

    #     Reward is the sum of sigmoids of below components
    #     - a. At this tester, (Number of setups before - Number of setups after this action)
    #     - b. At this tester, (Max. due date - Last completion time across all operations in this local bag)
    #     - c. (-ve of skewness of duedates of operations at the assigned tester)
    #     - d. (-ve of kurtosis of duedates of operations at the assigned tester)
    #     """

    #     a = reward_utils.compute_msagent_reward_total_setups_difference_at_tester(self.data, self.operations, self.localBags, self.prevStateAttributes, testerName)
    #     b = reward_utils.compute_msagent_reward_difference_in_max_duedate_and_completion_time(self.data, self.operations, self.configurations, self.localBags, testerName)
    #     c = -reward_utils.compute_msagent_reward_skewness_at_tester(self.operations, self.localBags[testerName])
    #     d = -reward_utils.compute_msagent_reward_kurtosis_at_tester(self.operations, self.localBags[testerName])

    #     if math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d) or a == float('inf') or b == float('inf') or c == float('inf') or  d == float('inf') or a == float('-inf') or b == float('-inf') or c == float('-inf') or d == float('-inf'):
    #         raise Exception('MSAgent Reward invalid value: ', a, b, c, d)

    #     # Handling setups component
    #     neg = True if a < 0 else False
    #     a = reward_utils.compute_sigmoid(abs(a)) * (-1 if neg else 1)

    #     # Handling completion component
    #     neg = True if b < 0 else False
    #     b = reward_utils.compute_sigmoid(abs(b)) * (-1 if neg else 1)

    #     # Handling skewness component
    #     neg = True if c < 0 else False
    #     c = reward_utils.compute_sigmoid(abs(c)) * (-1 if neg else 1)

    #     # Handling kurtosis component
    #     neg = True if d < 0 else False
    #     d = reward_utils.compute_sigmoid(abs(d)) * (-1 if neg else 1)

    #     return a + b + c + d
    

    def _calculate_msagent_reward(self, testerName) -> float:
        """Returns the reward for MSAgent. Calculated as below:

        Reward is the mammacher product of exponential of below components
        - a. At this tester, (Number of setups before - Number of setups after this action)
        - c. At this tester, difference of skewness of duedates before and after applying this action
        - d. At this tester, difference of kurtosis of duedates before and after applying this action
        """

        a = reward_utils.compute_msagent_reward_total_setups_difference_at_tester(self.data, self.operations, self.localBags, self.prevStateAttributes, testerName)
        # b = reward_utils.compute_msagent_reward_difference_in_max_duedate_and_completion_time(self.data, self.operations, self.configurations, self.localBags, testerName)
        b = 0
        c = -reward_utils.compute_msagent_reward_skewness_difference_at_tester(self.operations, self.localBags[testerName], self.prevStateAttributes, testerName)
        d = -reward_utils.compute_msagent_reward_kurtosis_difference_at_tester(self.operations, self.localBags[testerName], self.prevStateAttributes, testerName)

        if math.isnan(a) or math.isnan(b) or math.isnan(c) or math.isnan(d) or a == float('inf') or b == float('inf') or c == float('inf') or  d == float('inf') or a == float('-inf') or b == float('-inf') or c == float('-inf') or d == float('-inf'):
            raise Exception('MSAgent Reward invalid value: ', a, b, c, d)

        a = 1 if a >= 0 else 0
        c = 1 if c >= 0 else 0
        d = 1 if d >= 0 else 0

        return reward_utils.hamacher_product(a, reward_utils.hamacher_product(c, d))

        # # Handling setups component
        # neg = True if a < 0 else False
        # a = reward_utils.compute_sigmoid(abs(a)) * (-1 if neg else 1)

        # # Handling completion component
        # neg = True if b < 0 else False
        # b = reward_utils.compute_sigmoid(abs(b)) * (-1 if neg else 1)

        # # Handling skewness component
        # neg = True if c < 0 else False
        # c = reward_utils.compute_sigmoid(abs(c)) * (-1 if neg else 1)

        # # Handling kurtosis component
        # neg = True if d < 0 else False
        # d = reward_utils.compute_sigmoid(abs(d)) * (-1 if neg else 1)

        # return a + b + c + d


    # def _calculate_osagent_reward(self, agentName) -> float:
    #     """Returns the reward for OSAgent. Calculated as below:

    #     - | Difference of tardiness before and after state update for the corresponding tester |
    #     """
    #     testerName = agentName[len(OS_AGENT_PREFIX):]

    #     a = reward_utils.compute_osagent_reward_tardiness_difference(self.data, self.jobs, self.operations, self.testers, self.configurations, self.prevStateAttributes, testerName, self.time)

    #     if math.isnan(a) or a == float('inf') or a == float('-inf'):
    #         raise Exception('OSAgent Reward invalid value: ', a)

    #     return -a


    # def _calculate_osagent_reward(self, agentName) -> float:
    #     """Returns the reward for OSAgent. Calculated as below:

    #     - | Difference of tardiness before and after state update for the corresponding tester |
    #     """
    #     testerName = agentName[len(OS_AGENT_PREFIX):]

    #     a = reward_utils.compute_osagent_reward_tardiness_difference(self.data, self.jobs, self.operations, self.testers, self.configurations, self.prevStateAttributes, testerName, self.time)

    #     if math.isnan(a) or a == float('inf') or a == float('-inf'):
    #         raise Exception('OSAgent Reward invalid value: ', a)

    #     neg = True if a < 0 else False
    #     a = reward_utils.compute_sigmoid(abs(a)) * (-1 if neg else 1)

    #     return a

    def _calculate_osagent_reward(self, agentName) -> float:
        """Returns the reward for OSAgent. Calculated as below:

        - | Difference of tardiness before and after state update for the corresponding tester |
        """
        testerName = agentName[len(OS_AGENT_PREFIX):]

        #a = reward_utils.compute_osagent_reward_tardiness_difference(self.data, self.jobs, self.operations, self.testers, self.configurations, self.prevStateAttributes, testerName, self.time)

        a = reward_utils.compute_osagent_reward_median_difference(self.data, self.jobs, self.operations, self.testers, self.configurations, self.prevStateAttributes, testerName, self.time)
        if math.isnan(a) or a == float('inf') or a == float('-inf'):
            raise Exception('OSAgent Reward invalid value: ', a)

        a = reward_utils.compute_sigmoid(a)

        return a
    

    def _handle_msagent_action(self, action) -> None:
        """Applies the msagent action to reflect the environment state and performs any side-effects
        
        - Computes priority scores of each tester using the weights in action parameter
        - Moves the operation from global bag to local bag corresponding to the highest priority tester
        """
        operation = self._msagent_operation

        # Select best tester for the given operation based on the action weights
        if self.followGreedyStrategy == True:
            testerName = action_utils.select_msagent_tester_for_given_operation_following_greedy_strategy(self.data, self.jobs, self.operations, self.testers, self.configurations, self.localBags, operation, self.msGreedyStrategy)
        else:
            testerName = action_utils.select_msagent_tester_for_given_operation(self.data, self.jobs, self.operations, self.testers, self.configurations, self.localBags, operation, action)

        # Delete from global bag
        self.globalBag.remove(operation)

        # Insert in local bag
        self._put_op_in_local_bag(operation, testerName)

        return testerName

    
    def _handle_osagent_action(self, agentName, action) -> None:
        """Applies the osagent action to reflect the environment state and performs any side-effects

        - Computes the priority scores of each of the operations in the local bag using the weights in action
        - Selects the operation with highest priority score and starts it on the tester by removing from local bag
        """
        #print("handling action of the agent", agentName, action)
        testerName = agentName[len(OS_AGENT_PREFIX):]
        
        # Compute the operation priority scores
        if self.followGreedyStrategy == True:
            operation = action_utils.select_osagent_operation_for_given_tester_following_greedy_strategy(self.data, self.jobs, self.operations, self.testers, self.configurations, self.localBags, testerName, self.osGreedyStrategy, self.time) 
        else:
            operation = action_utils.select_osagent_operation_for_given_tester(self.data, self.jobs, self.operations, self.testers, self.configurations, self.localBags, testerName, action, self.time)

        # Delete from local bag

        self.localBags[testerName].remove(operation)
        #print('item_removed')

        # Find the best configuration for the operation on this tester
        best_config = obs_utils.compute_best_config_for_operation_on_a_tester(self.data, self.operations, operation, testerName)

        # Start the operation on the tester
        self.testers[testerName].update({
            'currentOperationName': operation,
            'testerStatus': TesterStatus.BUSY,
            'previousConfiguration': self.testers[testerName]['currentConfiguration'],
            'currentConfiguration': best_config
        })
        #print('tester made busy', testerName)
        # Update Job Status
        #print(self.operations[operation]['isRootOp'])
        #print(operation)
        if self.operations[operation]['isRootOp'] == True:

            #print('started processing root operation', operation)
            self.jobs[self.operations[operation]['jobName']]['status'] = JobStatus.IN_PROGRESS

        # Update operation status
        prevConfig = self.testers[testerName]['previousConfiguration']
        currConfig = self.testers[testerName]['currentConfiguration']
        self.operations[operation]['status'] = OperationStatus.IN_PROGRESS
        self.operations[operation]['configurationUsed'] = currConfig
        self.operations[operation]['previousConfiguration'] = prevConfig
        self.operations[operation]['startTime'] = self.time 
        self.operations[operation]['setupTime'] = 0 if prevConfig == currConfig else self.data['configurations']['items'][currConfig]['setupTimes'][self.data['configurations']['items'][prevConfig]['index']]
        #print(operation)
        self.operations[operation]['estimatedTestTime'] = obs_utils.compute_estimated_operation_test_time_under_configuration(self.data, self.operations[operation]['opType'], self.operations[operation]['configurationUsed'])
        #print('time :', self.time)
        #print('setuptime :', self.operations[operation]['setupTime'])
        #print('estimatedtime:', self.operations[operation]['estimatedTestTime'])
        self.operations[operation]['completionTime'] = self.time + self.operations[operation]['setupTime'] + self.operations[operation]['estimatedTestTime']

        # Track history
        self.history.append({
            'time': self.time,
            'opId': operation,
            'testerName': testerName
        })


    def _check_termination(self) -> bool:
        """Checks if the environment reached the terminal state.

        Returns True when the global bag and all local bags are empty with corresponding testers in non-busy state.
        """
        """
        if self.globalBag.size() > 0:
            return False
        for bag in self.localBags.values():
            if bag.size() > 0:
                return False
        for tester in self.testers.values():
            if tester['testerStatus'] == TesterStatus.BUSY:
                return False
        return True
        """
        if self.globalBag.size() > 0:
            return False
        for bag in self.localBags.values():
            if bag.size() > 0:
                return False
        for tester in self.testers.values():
            if tester['testerStatus'] == TesterStatus.BUSY:
                return False
        for productName, productDetails in self.data['products']['items'].items():
            if self.products[productName]['started'] == False:
                return False
        return True
    def _build_observations_dict(self) -> dict:    
        """Builds observations dictionary using internal environment state

        If there is atleast one operation in the global queue, the observations dict only contains an entry for MSAgent, 
        else, the observations dict contains an entry for each of the idle machines with non-empty local bag.
        """
        observations = {}
        if self.globalBag.size() > 0:
            operation = self.globalBag.sample()
            self._msagent_operation = operation
            observations[MS_AGENT] = self._build_msagent_observation(operation) 
        else:
            self._msagent_operation = None
            for testerName, testerDetails in self.testers.items():
                if testerDetails['testerStatus'] == TesterStatus.IDLE and self.localBags[testerName].size() > 0:
                    observations[OS_AGENT_PREFIX + testerName] = self._build_osagent_observation(self.localBags[testerName], testerName)
            
        for agentName, obs in observations.items():
            observations[agentName] = self._transform_obs(obs)
            
        return observations
    
    def _transform_obs(self, obs):
        for i, _obs in enumerate(obs):
            if _obs > 0:
                obs[i] = math.log(_obs + np.exp(1))
            elif _obs < 0:
                obs[i] = -math.log(-(_obs) + np.exp(1))
        return obs
    
    def _transform_rew(self, rew, agentName):
        if rew < 0:
            rew = -math.log((-rew) + np.exp(1))
        return rew

    def _get_static_config_data(self, config) -> dict:
        """Parses the scheduling problem in `config` json file and returns corresponding dict object.
        """
        staticConfigurationFilePath = config.get("staticConfigurationFilePath")
        with open(staticConfigurationFilePath) as staticConfigurationFile:
            data = json.load(staticConfigurationFile)
        return data


    def _init_internal_state(self, data):
        """Initializes the internal runtime state using parsed static configuration
        """
        self.time = 0

        self.steps = 0

        # Initialize testers state
        self.testers = {}
        for testerId, (testerName, testerDetails) in enumerate(data['testers']['items'].items()):
            self.testers[testerName] = ({
                # Static
                "id": testerId,

                # Dynamic
                "previousConfiguration": None,
                "currentConfiguration": random.choice(list(data['configurations']['items'].keys())),
                "currentOperationName": None,
                "testerStatus": TesterStatus.IDLE,
            })

        # Initialize configurations state
        self.configurations = {}
        for configName, configDetails in data['configurations']['items'].items():
            self.configurations[configName] = ({
                # Static
                "averageSetupTime": np.average(configDetails['setupTimes'])
            })

        # Initialize products state
        self.products = {}
        for productName, productDetails in data['products']['items'].items():
            self.products[productName] = ({
                # Static
                "jobs": [],
                "duedate": productDetails['duedate'],

                # Dynamic
                "started": False
            })

        # Initialize jobs and operations state
        self.jobs = {}
        self.operations = {}
        for productName, productDetails in data['products']['items'].items():

            # Computing operation due-dates, in-degree, out-degree
            operation_adjacency_list: Dict[int, list] = {}
            op_due_dates = {}
            in_degree = {}
            out_degree = {}
            avg_setup_time = {}
            avg_test_time = {}
            for opNum, opType in enumerate(data['products']['items'][productName]['operations']):
                operation_adjacency_list[opNum] = []
                op_due_dates[opNum] = None
                in_degree[opNum] = 0
                out_degree[opNum] = 0
                avg_setup_time[opNum] = obs_utils.compute_avg_operation_setup_time(self.data, self.configurations, opType)
                avg_test_time[opNum] = obs_utils.compute_avg_operation_test_time(self.data, opType)

            for dependency in data['products']['items'][productName]['dependencies']:
                independentOpNum = dependency[0]
                dependentOpNum = dependency[1]
                operation_adjacency_list[independentOpNum].append(dependentOpNum)
                out_degree[independentOpNum] += 1
                in_degree[dependentOpNum] += 1

            #print('in degrees of Operation')
            #for op in range(len(data['products']['items'][productName]['operations'])):
                #print(op,in_degree[op])

            roots = []
            for op in range(len(data['products']['items'][productName]['operations'])):
                if in_degree[op] == 0:
                    roots.append(op)
            #print(roots)
            for rootOp in roots:
                self._find_op_due_date(operation_adjacency_list, productName, rootOp, op_due_dates, avg_setup_time, avg_test_time)
            """
            for op in range(len(data['products']['items'][productName]['operations'])):
                if op_due_dates[op] <= 0:
                    print(op_due_dates[op])
                    raise Exception("Negative duedate for an operation")
            """
            # Creating jobs, operations
            for jobNum in range(productDetails['quantity']):
                jobName = productName + HYPHEN + str(jobNum)
                self.products[productName]['jobs'].append(jobName)
                self.jobs[jobName] = {
                    # Static
                    'productName': productName,
                    'operations': [],
                    
                    # Dynamic
                    'status': JobStatus.NOT_STARTED,
                    'estimatedRemainingTime': "",
                    'remainingNumOfOperations': len(productDetails['operations']),
                    'completionTime': None,
                }





                for opNum, opType in enumerate(productDetails['operations']):
                    opName = jobName + HYPHEN + str(opNum)
                    self.operations[opName] = {
                        # Static
                        'logicalOperationId': opNum,
                        'opType': opType,
                        'isRootOp': True if opName in roots else False,
                        'jobName': jobName,
                        'productName': productName, 
                        'duedate': op_due_dates[opNum],
                        'averageSetupTime': avg_setup_time[opNum],
                        'averageTestTime': avg_test_time[opNum],
                        'childOperations': operation_adjacency_list[opNum],
                        'outdegree': out_degree[opNum],

                        # Dynamic
                        'indegree': in_degree[opNum],
                        'status': OperationStatus.NOT_STARTED,
                        'assignedTesterName': None,
                        'startTime': None,
                        'estimatedTestTime': None,
                        'configurationUsed': None,
                        'completionTime': None,
                        'setupTime': None,
                        'previousConfiguration': None
                    }
                    self.jobs[jobName]['operations'].append(opName)
                    self.jobs[jobName]['estimatedRemainingTime'] = obs_utils.compute_remaining_setup_plus_test_time_of_job(self.jobs, self.operations, jobName, self.time)
                    #print('operations',opName, self.operations[opName]['isRootOp'])
        # Prev state attributes - Essentially the attributes from prev. state that needs to be saved for reward calculations.
        self.prevStateAttributes = {
            'setUps': {},
            'tardiness': {},
            'skewness': {},
            'kurtosis': {}
        } 
        
        # Initialize Global Bag
        self.globalBag: Bag = Bag()
        self._check_and_add_products_to_global_bag()

        # Initialize Local Bags
        self.localBags: Dict[str, Bag] = {}
        for testerName in data['testers']['items'].keys():
            self.localBags[testerName] = Bag()
            self.prevStateAttributes['setUps'][testerName] = 0
            self.prevStateAttributes['tardiness'][testerName] = 0
            self.prevStateAttributes['skewness'][testerName] = 0
            self.prevStateAttributes['kurtosis'][testerName] = 0

        # Initialize history to track events for rendering
        self.history = []
        self.historyIdx = 0

    
    def _put_op_in_global_bag(self, opName):
        self.globalBag.insert(opName)
        self.operations[opName]['status'] = OperationStatus.IN_GLOBAL_BAG

    def _put_op_in_local_bag(self, operation, testerName):
        self.localBags[testerName].insert(operation)
        self.operations[operation]['status'] = OperationStatus.IN_LOCAL_BAG
        self.operations[operation]['assignedTesterName'] = testerName

    def _find_op_due_date(self, adjacencyList, productName, root, op_due_dates, avg_setup_time, avg_test_time) -> float:
        """Performs depth first search on the product graph starting from provided root and finds due-dates of encountered nodes(operations)
        """
        if op_due_dates[root] == None:
            if len(adjacencyList[root]) == 0:
                op_due_dates[root] = self.data['products']['items'][productName]['duedate']
            else:
                min_due_date = np.inf
                for child in adjacencyList[root]:
                    min_due_date = min(min_due_date, self._find_op_due_date(adjacencyList, productName, child, op_due_dates, avg_setup_time, avg_test_time))
                op_due_dates[root] = min_due_date
        # return op_due_dates[root] - avg_setup_time[root] - avg_test_time[root]
        return op_due_dates[root] - avg_test_time[root]
            

    def _build_msagent_global_observation_list(self) -> list:
        """Builds the global bag attributes for MSAgent observation

        Contains: (23)
        - Number of operations
        - D.S of Setup + test time
        - D.S of number of compatible configurations
        - D.S of due dates
        - D.S of remaining test time of jobs
        - D.S of remaining number of operations of jobs
        
        D.S means Descriptive Statistics (Ex: mean, std, min, max, skewness, kurtosis etc...)
        """
        
        return obs_utils.flatten([
            obs_utils.compute_global_number_of_operations(self.globalBag),
            obs_utils.compute_global_setup_plus_test_times(self.operations, self.globalBag),
            obs_utils.compute_global_number_of_compatible_configurations(self.data, self.operations, self.globalBag),
            obs_utils.compute_global_due_dates(self.operations, self.globalBag),
            obs_utils.compute_global_remaining_test_time_of_jobs(self.data, self.jobs, self.operations, self.globalBag, self.time),
            obs_utils.compute_global_remaining_number_of_operations_for_jobs(self.jobs, self.operations, self.globalBag)
        ])


    def _build_msagent_opspecific_observation_list(self, opName) -> list:
        """Builds the operation specific attributes for MSAgent observation
        
        Contains: (5)
        - Avg. Setup + test time
        - Number of compatible configurations
        - Due date
        - Remaining test time of it's job
        - Remaining number of operations of it's job
        """
        return [
            obs_utils.compute_opspecific_avg_setup_plus_test_time(self.data, self.jobs, self.operations, opName),
            obs_utils.compute_opspecific_no_of_compatible_configurations(self.data, self.jobs, self.operations, opName),
            obs_utils.compute_opspecific_due_date_of_job(self.data, self.jobs, self.operations, opName),
            obs_utils.compute_opspecific_remaining_test_time_of_job(self.jobs, self.operations, opName, self.time),
            obs_utils.compute_opspecific_no_of_remaining_ops_of_job(self.jobs, self.operations, opName),
        ]


    def _build_msagent_local_observation_list(self) -> list:
        """Builds the local bag attributes for MSAgent observation
        
        Contains: (24)
        - D.S of Number of operations
        - D.S of Setup + test time
        - D.S of setups needed
        - D.S of due date skewness
        - D.S of due date kurtosis
        - D.S of ratio of setup to test times

        D.S means Descriptive Statistics (Ex: mean, std, min, max, skewness, kurtosis, etc...)
        """
        return obs_utils.flatten([
            obs_utils.compute_local_number_of_operations(self.localBags),
            obs_utils.compute_local_total_setup_plus_test_times(self.data, self.operations, self.configurations, self.localBags),
            obs_utils.compute_local_number_of_setups(self.data, self.operations, self.localBags),
            obs_utils.compute_local_duedate_skewness(self.operations, self.localBags),
            obs_utils.compute_local_duedate_kurtosis(self.operations, self.localBags),
            obs_utils.compute_local_ratio_of_setup_to_test_time(self.data, self.operations, self.configurations, self.localBags)
        ])


    def _build_msagent_observation(self, opName) -> list:
        """Builds the complete observation for MSAgent.

        Contains attributes related to below contexts: (52)
        - Global Bag (23)
        - Operation Specific (5)
        - Local bags (24)
        """
        obs = self._build_msagent_global_observation_list()
        obs.extend(self._build_msagent_opspecific_observation_list(opName))
        obs.extend(self._build_msagent_local_observation_list())
        return obs


    def _build_osagent_observation(self, localBag, testerName) -> list:
        """Builds the local bag attributes for OSAgent observation

        Contains: (26)
        - D.S of duedates
        - D.S of slacks
        - D.S of out-degrees
        - D.S of remaining test time of corresponding jobs
        - D.S of remaining number of operations of corresponding job
        - Total Setup + test time of all the operations in the local bag
        """
        return obs_utils.flatten([
            obs_utils.compute_local_bag_num_of_ops(localBag),
            obs_utils.compute_local_bag_total_setup_plus_test_time(self.data, self.operations, self.configurations, localBag, testerName),
            obs_utils.compute_local_bag_due_date(self.operations, localBag),
            obs_utils.compute_local_bag_slack(self.data, self.jobs, self.operations, localBag, self.time),
            obs_utils.compute_local_bag_out_degree(self.operations, localBag),
            obs_utils.compute_local_bag_remaining_test_time_of_jobs(self.jobs, self.operations, localBag, self.time),
            obs_utils.compute_local_bag_remaining_num_of_ops_of_jobs(self.jobs, self.operations, localBag)
        ])


if __name__ == "__main__":
    # Set Random Seed for reproducibility
    random.seed(20230522)

    # Config for instantiating the environment
    config = {
        "staticConfigurationFilePath": "data/real_life_data/static_configuration_2H_0.json",
        "maxSteps": 100,
        "setupRendering": False,
        "sampleRandomProblem": True,
    }

    # Instantiate the scheudling environment
    env = SchedulingEnv(config=config)
    print(env.reset())
    observations, rewards, terminateds, truncateds, infos = env.step({
        MS_AGENT: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        # OS_AGENT_PREFIX + 'T1': [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    })
    print(infos)
    env.render()
    print("DONE!")
