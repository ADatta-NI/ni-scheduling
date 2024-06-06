# Imports
from ortools.linear_solver import pywraplp
from ortools.linear_solver import linear_solver_pb2
from ortools.init.python import init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps
import random
import re
import json
from typing import Final
from typing import Optional
from typing import Sequence
import datetime
import argparse
import pickle
from google.protobuf import text_format

# Configurations
DATA_ROOT: Final[str] = 'data/'
SCHEDULING_RANDOM_SEED: Final[int] = 20231207
HYPHEN: Final[str] = '-'


class SchedulingMILP:
    """A mixed integer linear programming solution for the scheduling problem.
    """

    def __init__(self, config=None):
        # Set Random Seed for reproducibility
        self.randSeed = random.randrange(0, 2147483647)
        # self.randSeed = 1736117837
        random.seed(SCHEDULING_RANDOM_SEED)

        # OR-Tools initialization
        init.CppBridge.init_logging("scheduling_milp.py")
        cpp_flags = init.CppFlags()
        cpp_flags.stderrthreshold = 0
        cpp_flags.log_prefix = False
        init.CppBridge.set_flags(cpp_flags)

        # Parse the scheduling problem instance into internal attributes
        config = config or {}
        self.config = config
        self.data = self._get_static_config_data(self.config)
        self.staticConfigFileName = self.config.get("staticConfigurationFilePath")
        self.scName = re.sub(r'.*/', '', self.staticConfigFileName).removesuffix(".json")
        self.concurrentThreads = config.get("concurrentThreads")
        self.timeLimit = config.get("timeLimit")
        self.relativeMIPGap = config.get("relativeMIPGap")
        self.solverName = config.get("solverName")
        self.startTime = config.get("startTime")
        self.solverLogging = config.get("solverLogging")

        # Initialize internal state
        self._init_internal_state(self.data)

        # Declare the MIP solver
        self.solver = pywraplp.Solver.CreateSolver(self.solverName)
        if not self.solver:
            return
        if self.solverLogging:
            self.solver.EnableOutput()
        self.solver.SetNumThreads(self.concurrentThreads)
        self.solver.SetTimeLimit(self.timeLimit * 1000)
        self.solver.SetSolverSpecificParametersAsString("randomization/randomseedshift = " + str(self.randSeed))

        # Apply the solver parameters
        self.params = pywraplp.MPSolverParameters()
        self.params.SetDoubleParam(self.params.RELATIVE_MIP_GAP, self.relativeMIPGap)

        # Big-M coefficient
        self.G = 10000
        self.infinity = self.solver.infinity()

    def setTimeLimit(self, timeLimit):
        self.timeLimit = timeLimit
        self.solver.SetTimeLimit(self.timeLimit * 1000)

    def solve(self):
        """ Solves the mixed integer linear programming problem for scheduling

        - Creates all the necessary variables
        - Defines the objective constraint
        - Defines all the scheduling constraints
        - Solves the program to find state configuration to optimize the objective
        - Displays the found solution
        """

        # Add variables
        self._add_variables()
        print("All Variables added! A total of ", self.solver.NumVariables())

        # Add constraints
        self._add_constraints()
        print("All Constraints added! A total of ", self.solver.NumConstraints())

        # Define the objective
        self.solver.Minimize(self._var('obj_var'))

        # Call the MIP solver
        print(f'Solving with {self.solver.SolverVersion()}')
        status = self.solver.Solve(self.params)
        self.result = status

        # Resuming from checkpoint
        # # Call the MIP solver
        # # TODO: Solve using proto. set solver_time_limit, relative_mip_gap
        # print(f'Solving with {self.solver.SolverVersion()}')
        # model_proto = self._load_model()
        # request = linear_solver_pb2.MPModelRequest()
        # response = linear_solver_pb2.MPSolutionResponse()
        # # model_message_string = model_proto.MergeFromString(model_proto)
        # text_format.Parse(model_proto, request)
        # self.solver.SolveWithProto(model_request=request, response=response)
        # status = response.status

        # Display the solution
        if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.UNBOUNDED or status == pywraplp.Solver.ABNORMAL or status == pywraplp.Solver.NOT_SOLVED:
            print('PROBLEM!!!!')

        elif status == pywraplp.Solver.FEASIBLE:
            print('Stopped by limit!!!')
            # Print the solution
            self._print_solution()

            # Plot and save the solution
            self._plot_and_save()

            # Print Metrics
            self._print_metrics()

        elif status == pywraplp.Solver.OPTIMAL:
            print('Optimal Solution!!!')
            # Print the solution
            self._print_solution()

            # Verify the solution
            self.solver.VerifySolution(0.0001, True)

            # Plot and save the solution
            self._plot_and_save()

            # Print Metrics
            self._print_metrics()

        # Save the model
        self._save_model()

    def _save_model(self):
        """Save the model by serializing the protocol buffer
        """
        model_proto = linear_solver_pb2.MPModelProto()
        self.solver.ExportModelToProto(model_proto)
        # with open(DATA_ROOT + "_".join([str(item) for item in [self.startTime, self.scName, self.relativeMIPGap, self.timeLimit, self.concurrentThreads, self.solverName]]) + '.txt', "wb") as model_file:
        #     pickle.dump(model_proto, model_file)

        with open(DATA_ROOT + "_".join([str(item) for item in
                                        [self.startTime, self.scName, self.relativeMIPGap, self.timeLimit,
                                         self.concurrentThreads, self.solverName]]) + '.txt', "wb") as model_file:
            pickle.dump(model_proto.SerializeToString(), model_file)

    def _load_model(self):
        """Load the model from the serialized protocol buffer
        """
        model_proto = None
        with open(DATA_ROOT + "_".join([str(item) for item in
                                        [self.startTime, self.scName, self.relativeMIPGap, self.timeLimit,
                                         self.concurrentThreads, self.solverName]]) + '.txt', "rb") as model_file:
            model_proto = str(pickle.load(model_file))
        # self.solver.LoadModelFromProto(model_proto)
        return model_proto

    def _init_internal_state(self, data):
        """Initializes the internal state using parsed static configuration

        - Creates the products, jobs, operations, configurations and testers internal state.
        """
        # Create configurations
        self.configurations = {}
        for configName, configDetails in data['configurations']['items'].items():
            self.configurations[configName] = ({
                "index": configDetails['index'],
                "setupTimes": configDetails['setupTimes'],
            })

        # Create testers
        self.testers = {}
        for testerId, (testerName, testerDetails) in enumerate(data['testers']['items'].items()):
            self.testers[testerName] = ({
                "id": testerId,
                "compatibleConfigurations": testerDetails['supportedConfigurations'],
                "initialConfiguration": random.choice(testerDetails['supportedConfigurations']),
            })

        # Create products
        self.products = {}
        for productName, productDetails in data['products']['items'].items():
            self.products[productName] = ({
                "jobs": [],
                "duedate": productDetails['duedate'],
                "arrival": productDetails['arrival'],
                "quantity": productDetails['quantity']
            })

            # Create jobs and operations
        self.jobs = {}
        self.operations = {}
        uniqueJobId = 0
        for productName, productDetails in data['products']['items'].items():
            # Computing operation in-degree, out-degree, adjacency list, parent list
            operation_adjacency_list: Dict[int, list] = {}
            operation_parent_list: Dict[int, list] = {}
            in_degree = {}
            out_degree = {}
            for opNum, opType in enumerate(data['products']['items'][productName]['operations']):
                operation_adjacency_list[opNum] = []
                operation_parent_list[opNum] = []
                in_degree[opNum] = 0
                out_degree[opNum] = 0

            for dependency in data['products']['items'][productName]['dependencies']:
                independentOpNum = dependency[0]
                dependentOpNum = dependency[1]
                operation_adjacency_list[independentOpNum].append(dependentOpNum)
                operation_parent_list[dependentOpNum].append(independentOpNum)
                out_degree[independentOpNum] += 1
                in_degree[dependentOpNum] += 1

            roots = []
            for op in range(len(data['products']['items'][productName]['operations'])):
                if in_degree[op] == 0:
                    roots.append(op)

            # Creating job instances
            for jobNum in range(productDetails['quantity']):
                jobName = productName + HYPHEN + str(jobNum)
                self.products[productName]['jobs'].append(jobName)
                self.jobs[jobName] = {
                    'uniqueJobId': uniqueJobId,
                    'productName': productName,
                    'operations': [],
                    'arrival': self.products[productName]['arrival'],
                    'duedate': self.products[productName]['duedate'],
                }
                uniqueJobId += 1

                # Creating operation instances
                for opNum, opType in enumerate(productDetails['operations']):
                    opName = jobName + HYPHEN + str(opNum)
                    compatibleConfigurations = data['operations']['items'][opType]['compatibleConfigurations']
                    self.jobs[jobName]['operations'].append(opName)
                    self.operations[opName] = {
                        'logicalOperationId': opNum,  # 0, 1, 2, ...
                        'opType': opType,  # O1, O2, O3, ...
                        'isRootOp': True if opNum in roots else False,
                        'jobName': jobName,
                        'productName': productName,
                        'childOperations': operation_adjacency_list[opNum],
                        'parentOperations': operation_parent_list[opNum],
                        'compatibleConfigurations': compatibleConfigurations,
                        'testTime': {  # TODO: instead of mean, can sample from the distribution
                            configName: data['operations']['items'][opType]['estimatedTestTime'][configName]['mean'] for
                            configName in compatibleConfigurations
                        }
                    }

    def _add_variables(self):
        # Create variables
        ## Decision Variable to be used for building objective constraint
        self.solver.Var(0, self.infinity, False, 'obj_var')

        ## Problem specific variables
        for jobName, jobDetails in self.jobs.items():
            self.solver.Var(0, self.infinity, False, f'C_{jobName}')  # Completion Time C_j of a job j
            self.solver.Var(0, self.infinity, False, f'T_{jobName}')  # Tardiness T_j of a job j
        for opName, opDetails in self.operations.items():
            self.solver.Var(0, self.infinity, False, f's_{opName}')  # Start Time s_jo
            self.solver.Var(0, self.infinity, False, f'RT_{opName}')  # Reconfiguration Time taken by jo
            self.solver.Var(0, self.infinity, False, f'PT_{opName}')  # Processing Time taken by jo
            for testerName, testerDetails in self.testers.items():
                self.solver.BoolVar(
                    f'ybar_{opName}_{testerName}')  # Ybar ybar_jo_m - True if jo is the first test on tester m
                self.solver.BoolVar(
                    f'ycap_{opName}_{testerName}')  # Ycap ycap_jo_m - True if jo is the last test on tester m
                for configurationName in testerDetails['compatibleConfigurations']:
                    self.solver.BoolVar(
                        f'z_{opName}_{testerName}_{configurationName}')  # Z z_jo_mk      - True if jo is tested on mk
                for anotherOpName, anotherOpDetails in self.operations.items():
                    if anotherOpName != opName:
                        self.solver.BoolVar(
                            f'y_{opName}_{anotherOpName}_{testerName}')  # Y_jo_jdashodash_m - True if jdashodash follows jo o tester m

    def _add_constraints(self):
        # Define objective constraint
        tardiness = 0
        for jobName, jobDetails in self.jobs.items():
            tardiness += self._var(f'T_{jobName}')
        self.solver.Add(self._var('obj_var') >= tardiness, f'Objective_Constraint')

        # Define scheduling constraints
        ## Constraint 1: Each should should be performed only once
        for opName, opDetails in self.operations.items():
            sum = 0
            for testerName, testerDetails in self.testers.items():
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    sum += self._var(f'z_{opName}_{testerName}_{configurationName}')
            self.solver.Add(sum == 1, f'Constraint_1_{opName}')

        ## Constraint 2: Resource ordering constraint on a tester
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                for anotherOpName, anotherOpDetails in self.operations.items():
                    if anotherOpName != opName:
                        self.solver.Add(self._var(f's_{opName}') >= self._var(f's_{anotherOpName}') + self._var(
                            f'PT_{anotherOpName}') + self._var(f'RT_{opName}') + self.G * (
                                                    self._var(f'y_{anotherOpName}_{opName}_{testerName}') - 1),
                                        f'Constraint_2_{testerName}_{opName}_{anotherOpName}')

        # New Constraint 13: Start time of the first operation is after its reconfiguration is done (edge-case for Constraint #2)
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                self.solver.Add(self._var(f's_{opName}') >= self._var(f'RT_{opName}') + self.G * (
                            self._var(f'ybar_{opName}_{testerName}') - 1), f'Constraint_13_{testerName}_{opName}')

        ## Constraint 3: A test should either be the first one or have a predecessor on that tester
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                ysum = 0
                zsum = 0
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
                ysum += self._var(f'ybar_{opName}_{testerName}')
                for anotherOpName, anotherOpDetails in self.operations.items():
                    if (opName != anotherOpName):
                        ysum += self._var(f'y_{anotherOpName}_{opName}_{testerName}')
                self.solver.Add(ysum == zsum, f'Constraint_3_{testerName}_{opName}')

        ## Constraint 4: A test should either be the last one or have a successor on that tester
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                ysum = 0
                zsum = 0
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
                ysum += self._var(f'ycap_{opName}_{testerName}')
                for anotherOpName, anotherOpDetails in self.operations.items():
                    if (opName != anotherOpName):
                        ysum += self._var(f'y_{opName}_{anotherOpName}_{testerName}')
                self.solver.Add(ysum == zsum, f'Constraint_4_{testerName}_{opName}')

        ## Constraint 5: If a tester is used at all, it should have some starting operation
        for testerName, testerDetails in self.testers.items():
            zsum = 0
            ysum = 0
            for opName, opDetails in self.operations.items():
                ysum += self._var(f'ybar_{opName}_{testerName}')
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
            self.solver.Add(self.G * ysum >= zsum, f'Constraint_5_{testerName}')

        ## Constraint 6: If a tester is used at all, it should have some ending operation
        for testerName, testerDetails in self.testers.items():
            zsum = 0
            ysum = 0
            for opName, opDetails in self.operations.items():
                ysum += self._var(f'ycap_{opName}_{testerName}')
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
            self.solver.Add(self.G * ysum >= zsum, f'Constraint_6_{testerName}')

        ## Constraint 7: On a tester, at max there should be only one starting operation
        for testerName in self.testers:
            ysum = 0
            for opName in self.operations:
                ysum += self._var(f'ybar_{opName}_{testerName}')
            self.solver.Add(ysum <= 1, f'Constraint_7_{testerName}')

        ## Constraint 8: On a tester, at max there should be only one ending operation
        for testerName in self.testers:
            ysum = 0
            for opName in self.operations:
                ysum += self._var(f'ycap_{opName}_{testerName}')
            self.solver.Add(ysum <= 1, f'Constraint_8_{testerName}')

        ## Constraint 9: Precedence constraint between two tests of a particular DUT where 2nd can't be performed before first
        for jobName, jobDetails in self.jobs.items():
            for opName in jobDetails['operations']:
                for parentOpNum in self.operations[opName]['parentOperations']:
                    parentOpName = jobName + HYPHEN + str(parentOpNum)
                    self.solver.Add(
                        self._var(f's_{opName}') >= self._var(f's_{parentOpName}') + self._var(f'PT_{parentOpName}'),
                        f'Constraint_9_{opName}_{parentOpName}')

        ## Constraint 10: Reconfiguration time of a test should satisfy this condition
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                for configurationName in possibleConfigurations:
                    configurationId = self.configurations[configurationName]['index']
                    for anotherOpName, anotherOpDetails in self.operations.items():
                        if (opName != anotherOpName):
                            anotherOperationConfigurations = anotherOpDetails['compatibleConfigurations']
                            anotherPossibleConfigurations = set(anotherOperationConfigurations) & set(
                                testerConfigurations)
                            for anotherConfigurationName in anotherPossibleConfigurations:
                                setupTime = self.configurations[anotherConfigurationName]['setupTimes'][configurationId]
                                self.solver.Add(self._var(f'RT_{opName}') >= setupTime + self.G * (
                                            self._var(f'z_{opName}_{testerName}_{configurationName}') + self._var(
                                        f'z_{anotherOpName}_{testerName}_{anotherConfigurationName}') + self._var(
                                        f'y_{anotherOpName}_{opName}_{testerName}') - 3),
                                                f'Constraint_10_{testerName}_{opName}_{configurationName}_{anotherOpName}_{anotherConfigurationName}')

        ## Constraint 11: Reconfiguration time of a test which is running first on a machine should satisfy this condition
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                operationConfigurations = opDetails['compatibleConfigurations']
                testerConfigurations = testerDetails['compatibleConfigurations']
                possibleConfigurations = set(operationConfigurations) & set(testerConfigurations)
                initialConfiguration = testerDetails['initialConfiguration']
                for configurationName in possibleConfigurations:
                    configurationId = self.configurations[configurationName]['index']
                    setupTime = self.configurations[initialConfiguration]['setupTimes'][configurationId]
                    self.solver.Add(self._var(f'RT_{opName}') >= setupTime + self.G * (
                                self._var(f'z_{opName}_{testerName}_{configurationName}') + self._var(
                            f'ybar_{opName}_{testerName}') - 2),
                                    f'Constraint_11_{testerName}_{opName}_{configurationName}')

        ## Constraint 12: Processing time of an operation should satisfy this condition
        for testerName, testerDetails in self.testers.items():
            for opName, opDetails in self.operations.items():
                testerConfigurations = testerDetails['compatibleConfigurations']
                operationConfigurations = opDetails['compatibleConfigurations']
                possibleConfigurations = set(testerConfigurations) & set(operationConfigurations)
                for configurationName in possibleConfigurations:
                    testTime = opDetails['testTime'][configurationName]
                    self.solver.Add(self._var(f'PT_{opName}') >= testTime + self.G * (
                                self._var(f'z_{opName}_{testerName}_{configurationName}') - 1),
                                    f'Constraint_12_{testerName}_{opName}_{configurationName}')

        ## Constraint 14: (Needed to break circular dependency) If more than 2 tests are scheduled on a tester, a particular test can't be both the starting and ending one
        for testerName, testerDetails in self.testers.items():
            zsum = 0
            for opName, opDetails in self.operations.items():
                testerConfigurations = testerDetails['compatibleConfigurations']
                operationConfigurations = opDetails['compatibleConfigurations']
                possibleConfigurations = set(testerConfigurations) & set(operationConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
            for opName, opDetails in self.operations.items():
                self.solver.Add(self.G * (2 - self._var(f'ybar_{opName}_{testerName}') - self._var(
                    f'ycap_{opName}_{testerName}')) >= (zsum - 1), f'Constraint_14_{testerName}_{opName}')

        ## Constraint 15: There should be n-1 number of transitions on a tester where n tests are scheduled
        for testerName, testerDetails in self.testers.items():
            ysum = 0
            zsum = 0
            for opName, opDetails in self.operations.items():
                for anotherOpName, anotherOpDetails in self.operations.items():
                    if (opName != anotherOpName):
                        ysum += self._var(f'y_{anotherOpName}_{opName}_{testerName}')
                testerConfigurations = testerDetails['compatibleConfigurations']
                operationConfigurations = opDetails['compatibleConfigurations']
                possibleConfigurations = set(testerConfigurations) & set(operationConfigurations)
                for configurationName in possibleConfigurations:
                    zsum += self._var(f'z_{opName}_{testerName}_{configurationName}')
            self.solver.Add(ysum == zsum - 1)

        ## Constraint 16: Tardiness T_j defining constraint
        for jobName, jobDetails in self.jobs.items():
            self.solver.Add(self._var(f'T_{jobName}') >= self._var(f'C_{jobName}') - jobDetails['duedate'],
                            f'Constraint_16_{jobName}')

        ## Constraint 17: Completion time C_j defining constraint
        for jobName, jobDetails in self.jobs.items():
            for opName in jobDetails['operations']:
                self.solver.Add(self._var(f'C_{jobName}') >= self._var(f's_{opName}') + self._var(f'PT_{opName}'),
                                f'Constraint_17_{opName}')

        ## Constraint 18: Start time of an operation should be after it's arrival time. Note that it's setup on corresonding tester can be done before it's start time.
        for opName, opDetails in self.operations.items():
            if opDetails['isRootOp']:
                arrivalTime = self.jobs[opDetails['jobName']]['arrival']
                self.solver.Add(self._var(f's_{opName}') >= arrivalTime, f'Constraint_18_{opName}')

    def _var(self, name):
        return self.solver.LookupVariable(name)

    def _print_metrics(self):
        print('\nMetrics:')
        print('Problem solved in %f milliseconds' % self.solver.wall_time())
        print('Problem solved in %d iterations' % self.solver.iterations())
        print('Problem solved in %d branch-and-bound nodes' % self.solver.nodes())
        print('Objective value =', self.solver.Objective().Value())
        print('Chosen Seed: %d' % self.randSeed)
        print('Start time: ' + str(self.startTime))
        status = self.result
        if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.UNBOUNDED or status == pywraplp.Solver.ABNORMAL or status == pywraplp.Solver.NOT_SOLVED:
            print('PROBLEM!!!!')
        elif status == pywraplp.Solver.FEASIBLE:
            print('Stopped by limit!!!')
        elif status == pywraplp.Solver.OPTIMAL:
            print('Optimal Solution!!!')
        else:
            print('WHAT!!!')

    def _plot_and_save(self):
        # Plotting the Gantt Chart
        fig, ax = plt.subplots()
        end_max = 0

        # Map integer values to a color from a colormap
        cmap = colormaps['viridis']
        jobColors = cmap(np.linspace(0, 1, len(self.jobs.keys())))
        for opName, opDetails in self.operations.items():
            testerName, configurationName = self._find_tester_scheduled(opName)
            testerId = self.testers[testerName]['id']
            s = self._var(f's_{opName}').solution_value()
            rt = self._var(f'RT_{opName}').solution_value()
            pt = self._var(f'PT_{opName}').solution_value()
            end_max = max(end_max, rt + s + pt)

            # Add horizontal bars and labels
            ax.barh(y=testerId, width=rt, left=s - rt, height=0.5, color='red')
            ax.barh(y=testerId, width=pt, left=s, height=0.5,
                    color=jobColors[self.jobs[self.operations[opName]['jobName']]['uniqueJobId']])
            center_x = s
            center_y = testerId + 0.3
            ax.text(x=center_x, y=center_y, s=str([opName, configurationName]), ha='center', va='bottom', rotation=45)

            # Set axis labels and limits
            ax.set_xlabel('Time')
            ax.set_ylabel('Testers')
            ax.set_ylim([-0.5, len(self.testers.keys()) - 0.5])
            ax.set_xlim([0, end_max + 1])

        # Save the plot
        plt.savefig(DATA_ROOT + "_".join([str(item) for item in
                                          [self.startTime, self.scName, self.relativeMIPGap, self.timeLimit,
                                           self.concurrentThreads, self.solverName]]) + '.png')

        # Show the plot
        plt.show()

    def _print_solution(self):
        print('Objective value =', self.solver.Objective().Value())
        print('Number of variables =', self.solver.NumVariables())
        print('Number of constraints =', self.solver.NumConstraints())

        # What are running on each tester:
        print("What are running on each tester?")
        for testerName, testerDetails in self.testers.items():
            print(f"On Tester: {testerName}")
            for configurationName in testerDetails['compatibleConfigurations']:
                for opName in self.operations:
                    if self._var(f'z_{opName}_{testerName}_{configurationName}').solution_value():
                        print(f"{opName}, {configurationName}")

        # What is the starting operation on a tester:
        print("What is the starting operation on each tester?")
        for testerName, testerDetails in self.testers.items():
            lst = []
            for opName in self.operations:
                if self._var(f'ybar_{opName}_{testerName}').solution_value():
                    lst.append(opName)
            print(f"Starting Op on M: {testerName}:: ", lst)

        # What is the ending operation on a tester:
        print("What is the ending operation on each tester?")
        for testerName in self.testers:
            lst = []
            for opName in self.operations:
                if self._var(f'ycap_{opName}_{testerName}').solution_value():
                    lst.append(opName)
            print(f"Ending Op on M: {testerName}:: ", lst)

        # Checking dependencies on a tester:
        print("What are the dependencies on each tester?")
        for testerName, testerDetails in self.testers.items():
            print(f"On Tester: {testerName}", end='\n')
            for anotherOpName, anotherOpDetails in self.operations.items():
                for opName, opDetails in self.operations.items():
                    if (opName != anotherOpName):
                        if self._var(f"y_{anotherOpName}_{opName}_{testerName}").solution_value():
                            print(f'({anotherOpName}) -> ({opName})')

        # What is the tardiness for each of the jobs:
        print("What is the tardiness for each of the jobs?")
        for jobName in self.jobs:
            print(f"For Job: {jobName}", self._var(f'T_{jobName}').solution_value())

        # Printing operation-wise results
        print("Operation wise results:")
        self._print_op_wise_results()

    def _find_tester_scheduled(self, opName):
        for testerName, testerDetails in self.testers.items():
            for configurationName in testerDetails['compatibleConfigurations']:
                if self._var(f'z_{opName}_{testerName}_{configurationName}').solution_value():
                    return testerName, configurationName

    def _print_op_wise_results(self):
        for opName, opDetails in self.operations.items():
            print(f"Operation: ({opName})", f"Tester: {self._find_tester_scheduled(opName)}",
                  f"Start: {self._var(f's_{opName}').solution_value()}",
                  f"RT: {self._var(f'RT_{opName}').solution_value()}",
                  f"PT: {self._var(f'PT_{opName}').solution_value()}")

    def _get_static_config_data(self, config) -> dict:
        """Parses the scheduling problem in `config` json file and returns corresponding dict object.
        """
        staticConfigurationFilePath = config.get("staticConfigurationFilePath")
        with open(staticConfigurationFilePath) as staticConfigurationFile:
            data = json.load(staticConfigurationFile)
        return data


def parse_arguments(argv: Optional[Sequence[str]] = None) -> dict:
    """Defines cmd args for this program and parses the provided arg values and returns them.
    """
    # Define command line arguments and parse them
    parser = argparse.ArgumentParser()

    ## Arg: static config filepath
    parser.add_argument(
        '-scf', '--static-config-filepath',
        default="data/xsmall_sc.json",
        help='Specify the filepath of the static config file which needs to be solved. (Default: %(default)s)'
    )

    ## Arg: number of concurrent threads
    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=1,
        help='Specify the number of concurrent threads to be used for solving. (Default: %(default)s)'
    )

    ## Arg: time limit for solving
    parser.add_argument(
        '-tl', '--time-limit',
        type=int,
        default=2592000,
        help='Specify the time limit in seconds to stop the search. (Default: %(default)s)'
    )

    ## Arg: RELATIVE_MIP_GAP for stopping the search
    parser.add_argument(
        '-rmg', '--relative-mip-gap',
        type=float,
        default=0,
        help='Specify the relative mip gap to stop the search. (Default: %(default)s)'
    )

    ## Arg: solver name
    parser.add_argument(
        '-s', '--solver-name',
        default='SCIP',
        choices=('SCIP', 'GUROBI', 'CBC', 'CPLEX', 'XPRESS', 'GLPK'),
        help='Specify the solver using which the problem has to be solved. (Default: %(default)s)'
    )

    ## Arg: enable solver logging
    parser.add_argument(
        '-log', '--solver-logging',
        action='store_true',
        help='Specify whether solver output logging should be enabled.'
    )

    # Parse and return argument values
    return parser.parse_args(argv)


if __name__ == "__main__":
    start_time = datetime.datetime.now().time()

    # Parse CMD Arguments
    args = parse_arguments()

    print(start_time, args)

    # Define the problem instance to solve
    config = {
        "staticConfigurationFilePath": args.static_config_filepath,
        "concurrentThreads": args.threads,
        "timeLimit": args.time_limit,
        "relativeMIPGap": args.relative_mip_gap,
        "solverName": args.solver_name,
        "startTime": start_time,
        "solverLogging": args.solver_logging,
    }

    # Instantiate the SchedulingMILP solver
    milp = SchedulingMILP(config=config)

    # Solve the problem instance
    milp.solve(initial=True)

## Sample Run Command
# python scheduling_milp.py -s SCIP -th 1 -scf "data/small_sc.json" -rmg 10 -tl 300
