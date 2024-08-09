import random
import json
import networkx as nx
import numpy as np
import datetime
from collections import defaultdict
from datetime import datetime
from typing_extensions import final
DATA_GENERATOR_TRAIN_SEED = 100
DATA_GENERATOR_TEST_SEED = 42


def _process_data(zeno_production_schedule_path, prod_map_alt_boms_path, extra_path):
    # Read data from the Zeno production schedule file
    with open(zeno_production_schedule_path, "r") as file:
        zeno_production_schedule = json.load(file)

    # Read data from the prod_map_alt_boms file
    with open(prod_map_alt_boms_path, "r") as file:
        prod_map_alt_boms = json.load(file)

    with open(extra_path, "r") as file:
        extra_resources = json.load(file)

    # Create a mapping from product_pn to entry in prod_map_alt_boms
    product_pn_mapping = {}
    for entry in prod_map_alt_boms["prod_map_alt_boms"]:
        product_pn = entry["product_pn"]
        operation_sequence = entry["op_sequence"]
        # print(product_pn,  operation_sequence)
        required_asset_pn = entry["required_asset_pn"]
        configuration = entry["configuration"]
        if product_pn not in product_pn_mapping:
            product_pn_mapping[product_pn] = {"configuration": [configuration],
                                              "tester": [required_asset_pn],
                                              "operation": [operation_sequence]}
        elif configuration not in product_pn_mapping[product_pn]["configuration"] \
                and required_asset_pn not in product_pn_mapping[product_pn]["tester"] \
                and operation_sequence not in product_pn_mapping[product_pn]["operation"]:
            product_pn_mapping[product_pn]["configuration"].append(configuration)
            product_pn_mapping[product_pn]["tester"].append(required_asset_pn)
            product_pn_mapping[product_pn]["operation"].append(operation_sequence)

    print(product_pn_mapping)

    # Initialize lists to store unique required_asset_pn and configurations
    # store the operations which are using the FI and FA table
    unique_testers = set()
    unique_configurations = set()
    final_operations_mapping = {}

    for entry in extra_resources["resource_code_mapping"]:
        final_operation = entry["op_sequence"]
        final_tester = entry['mapping']
        final_configuration = entry['configuration']
        if final_operation not in final_operations_mapping:
            final_operations_mapping[final_operation] = {'tester': final_tester, 'configuration': final_configuration}
        elif configuration not in final_operations_mapping[final_operation]["configuration"] \
                and required_asset_pn not in final_operations_mapping[final_operation]["tester"]:
            final_operations_mapping[final_operation]["tester"].append(final_tester)
            final_operations_mapping[final_operation]["configuration"].append(final_configuration)

    # include the operations of final assembly and final inspection in both
    # unique testers and configurations and the product_pn_mapping
    # Iterate over the zeno_production_schedule entries
    for entry in zeno_production_schedule["zeno_production_schedule"]:
        part_number = entry.get("part_number")
        operation = entry.get('operation_sequence')
        # print(part_number)
        # Check if part_number exists and if it's in the product_pn_mapping
        if part_number and part_number in product_pn_mapping:
            product_entry = product_pn_mapping[part_number]
            # Add to the lists for storing unique required_asset_pn and configurations

            if operation in final_operations_mapping.keys():
                product_pn_mapping[part_number]['operation'].append(operation)
                product_pn_mapping[part_number]['tester'].append(final_operations_mapping[operation]['tester'])
                product_pn_mapping[part_number]['configuration'].append(
                    final_operations_mapping[operation]['configuration'])
                print(final_operations_mapping[operation]['tester'])
                unique_testers.update(product_entry["tester"])
                unique_configurations.update(product_entry["configuration"])

            elif operation not in final_operations_mapping.keys():
                unique_testers.update(product_entry["tester"])
                unique_configurations.update(product_entry["configuration"])

            # unique_testers.update(final_operations_mapping[operation]['tester'])
            # unique_configurations.update(final_operations_mapping[operation]['configuration'])

    # Convert unique_required_asset_pn and unique_configurations to lists
    # unique_required_asset_pn = list(unique_required_asset_pn)
    # unique_configurations = list(unique_configurations)

    return unique_testers, unique_configurations, product_pn_mapping


# TODO Create a operation to product mapping
# TODO map the test time for each product in the config files
# TODO study the zeno dataset and find the relationship of products and operations with due dates
# in the environment instead of configuration used apply product being tested
# only for the estimated time and the other related codes if necessary
# mostly it is used only in the sample part

# TODO convert the integer priority to float priority
# Add the priority field to the products


# TODO change all the wip identity to product part number


class DataGenerator:
    def __init__(self, config=None, unique_testers=None, unique_configurations=None, product_pn_mapping=None):
        self.config = config or {}
        self.dirPath = self.config.get('dirPath')
        self.purpose = self.config.get('purpose')
        self.productPath = self.config.get('productPath')
        self.configPath = self.config.get('configPath')
        self.originalconfigPath = self.config.get('originalconfigPath')
        self.unique_testers = unique_testers
        self.unique_configurations = unique_configurations
        self.product_pn_mapping = product_pn_mapping
        random.seed(DATA_GENERATOR_TRAIN_SEED if self.purpose == 'train' else DATA_GENERATOR_TEST_SEED)

    def generate_and_save(self):
        self._num_of_static_configuration_files = self.config.get('numOfStaticConfigurationFiles')
        for idx in range(self._num_of_static_configuration_files):
            self._generate_and_save_single_config(idx + 1)

    def _generate_and_save_single_config(self, idx):
        # Initialize data
        self.data = {}
        self.resource_config_mapping = {}
        # self.opseq_config_mapping = {}
        self.product_operation_mapping = {}
        self.operation_product_mapping = {}
        self.product_arrival_mapping = {}
        self.product_operation_testtime_mapping = {}
        self.due_date_mapping = {}
        self.product_quantity_mapping = {}
        self.operation_config_mapping = {}
        # self.opseq_org_config_mapping = {}
        # TODO add priority mapping
        # if null assign 2500 then normalised 1
        # normalise means divide by 1

        ## Generate mappings
        self._create_mappings()

        # Generate configurations
        self._generate_configurations()

        # Generate testers
        self._generate_testers()

        # Generate operations
        self._generate_operations()

        # Generate products
        self._generate_products()

        fileIndex = self.productPath.split('/')[-1]

        filePath = self.dirPath + 'static_configuration_plan_' + '_all_' + str(fileIndex)
        with open(filePath, 'w') as f:
            json.dump(self.data, f)

    def _create_mappings(self):
        # Read JSON file

        print('called_1')
        with open(self.configPath, 'r') as json_file:
            config_data = json.load(json_file)
        with open(self.originalconfigPath, 'r') as json_file:
            config_org_data = json.load(json_file)
        with open(self.productPath, 'r') as json_file:
            product_data = json.load(json_file)
            # print(product_data)

        for entry in product_data["zeno_production_schedule"]:
            part_number = entry["part_number"]
            operation = entry["operation_sequence"]
            # print('operation', operation)

            ## change when we get the updated tams resource file
            if part_number in self.product_pn_mapping.keys():
                output = self.product_pn_mapping[part_number]['operation']
                # print("extracted:",operation, output)
                # print(self.product_pn_mapping['199223J-01L'])
                # Check if part_number exists and if it's in the product_pn_mapping
            if operation in output and part_number in self.product_pn_mapping:
                product_entry = self.product_pn_mapping[part_number]
                # print(product_entry)
                # Add to the lists for storing unique required_asset_pn and configurations
                if f'O{operation}' not in self.operation_config_mapping:
                    self.operation_config_mapping[f'O{operation}'] = {
                        "compatibleConfig": {f'P{part_number}': product_entry['configuration']}}

                elif part_number not in self.operation_config_mapping[f'O{operation}']["compatibleConfig"]:

                    self.operation_config_mapping[f'O{operation}']["compatibleConfig"][f'P{part_number}'] = \
                    product_entry['configuration']

        print(self.operation_config_mapping)

        for entry in product_data["zeno_production_schedule"]:
            product_sequence = f'P{entry["part_number"]}'
            op_sequence = f'O{entry["operation_sequence"]}'

            # print("generated:", op_sequence)
            ## product to operation mapping
            if op_sequence in self.operation_config_mapping.keys():
                if product_sequence in self.operation_config_mapping[op_sequence]['compatibleConfig']:
                    if product_sequence not in self.product_operation_mapping:
                        self.product_operation_mapping[product_sequence] = {'operations': [op_sequence]}
                    elif op_sequence not in self.product_operation_mapping[product_sequence]['operations']:
                        self.product_operation_mapping[product_sequence]['operations'].append(op_sequence)

                ## operation to product mapping for the distinct estimated time
                if product_sequence in self.operation_config_mapping[op_sequence]['compatibleConfig']:
                    if op_sequence not in self.operation_product_mapping:
                        self.operation_product_mapping[op_sequence] = {'products': [product_sequence]}
                    elif product_sequence not in self.operation_product_mapping[op_sequence]['products']:
                        print('adding product', product_sequence)
                        self.operation_product_mapping[op_sequence]['products'].append(product_sequence)

        print('product to operation mapping', (self.product_operation_mapping))
        print('operation to product mapping', (self.operation_product_mapping))
        # print('operation to product mapping', self.operation_product_mapping)

        # Create a mapping between "required_asset_pn" and "configuration"
        for tester in self.unique_testers:
            if tester not in self.resource_config_mapping:
                self.resource_config_mapping[tester] = {"configuration": [tester.replace('T', 'K')]}

        for entry in product_data["zeno_production_schedule"]:

            product_sequence = f'P{entry["part_number"]}'
            quantity = entry["planned_quantity"]

            if product_sequence not in self.product_quantity_mapping:
                self.product_quantity_mapping[product_sequence] = {"quantity": [quantity]}
            elif quantity not in self.product_quantity_mapping[product_sequence]["quantity"]:
                self.product_quantity_mapping[product_sequence]["quantity"].append(quantity)

        creation_date = []
        for entry in product_data["zeno_production_schedule"]:
            creation_date.append(datetime.strptime(entry["job_creation_date"], "%Y-%m-%d %H:%M:%S"))
            min_creation_date = min(creation_date)
        print('start_time:', min_creation_date)

        for entry in product_data["zeno_production_schedule"]:
            product_sequence = f'P{entry["part_number"]}'
            # creation_date = datetime.strptime(entry["job_creation_date"], "%Y-%m-%d %H:%M:%S")
            imported_date = datetime.strptime(entry["job_imported_date"], "%Y-%m-%d %H:%M:%S")
            arrival_time = (imported_date - min_creation_date).total_seconds() / 60

            if product_sequence not in self.product_arrival_mapping:
                self.product_arrival_mapping[product_sequence] = {"arrival_time": [arrival_time]}
            elif arrival_time not in self.product_arrival_mapping[product_sequence]["arrival_time"]:
                self.product_arrival_mapping[product_sequence]["arrival_time"].append(arrival_time)

        ##use in operation estimated test time
        for entry in product_data["zeno_production_schedule"]:
            product_sequence = f'P{entry["part_number"]}'
            part_number = entry["part_number"]
            # print(product_sequence)
            op_sequence = f'O{entry["operation_sequence"]}'
            # print(op_sequence)
            # print(op_sequence)
            lot_time = entry["sum_lot_time"]
            item_time = entry["sum_item_time"]
            quantity = entry["job_quantity"]
            total_time = (lot_time + quantity * item_time) * 60
            ## use this double mapping to correctly assign the unique test times for each product operation pair
            ## to the correct product config for each distinct operation
            ## product remains the same but the total time is different under each operation
            if op_sequence in self.operation_config_mapping.keys():
                if product_sequence in self.operation_config_mapping[op_sequence]['compatibleConfig']:
                    if product_sequence not in self.product_operation_testtime_mapping:
                        self.product_operation_testtime_mapping[product_sequence] = {
                            "estimated_time": {op_sequence: total_time}}
                    elif op_sequence not in self.product_operation_testtime_mapping[product_sequence]["estimated_time"]:
                        self.product_operation_testtime_mapping[product_sequence]["estimated_time"][
                            op_sequence] = total_time

        # print('check mapping of product P156860B-02L to operation test time', self.product_operation_testtime_mapping['P156860B-02L'])
        # print(self.opseq_config_mapping['O140_copy_2313']['configuration'])

        ##use this in compute due date
        for entry in product_data["zeno_production_schedule"]:
            product_sequence = f'P{entry["part_number"]}'
            # creation_date = datetime.strptime(entry["job_creation_date"], "%Y-%m-%d %H:%M:%S")
            due_date = datetime.strptime(entry["due_date"], "%Y-%m-%d %H:%M:%S")
            print("due date:", due_date)
            due_date_time = ((due_date - min_creation_date).total_seconds()) / 60
            print("due date time:", due_date_time)
            ## make the changes here as well
            if product_sequence not in self.due_date_mapping:
                self.due_date_mapping[product_sequence] = {"due_date_time": [due_date_time]}
            elif due_date_time < min(self.due_date_mapping[product_sequence]["due_date_time"]):
                self.due_date_mapping[product_sequence] = {"due_date_time": [due_date_time]}

    def _generate_products(self):
        ''' Generates the products section of data.

        - The number of products should be in the range ['minProducts', 'maxProducts'].
        - The number of operations for a product should be in the range ['minNumOperationsPerProduct', 'maxNumOperationsPerProduct'], if numOfOperations is less than min or max of operationsPerProduct, then numOfOperations should be considered.
        - The number of edges in each product graph should be tried to be fit in this percentage range ['minProductEdgesPercent', 'maxProductEdgesPercent'], where percentage is with respect to a complete graph.
        '''

        self.numOfProducts = len(self.product_operation_mapping)
        self.data['products'] = {
            'count': self.numOfProducts,
            'items': {
                product_ids: {
                    'index': product_ids[1:],
                    'arrival': '',
                    'quantity': self.product_quantity_mapping[product_ids]["quantity"][0],
                    'duedate': '',
                    'operations': '',
                    'dependencies': []
                } for product_ids in self.product_operation_mapping.keys()
            }
        }
        print(self.data['products']['items'])
        # Generating product graphs and computing due dates
        ops = list(self.data['operations']['items'].keys())
        initial_product = list(self.data['products']['items'].keys())[0]
        for productName in self.data['products']['items'].keys():
            numOps = len(self.product_operation_mapping[productName]['operations'])  # length of operation
            edgePercent = random.uniform(self.config.get('minProductEdgesPercent'),
                                         self.config.get('maxProductEdgesPercent'))
            # edgePercent = 1.00
            selectedOps = self.product_operation_mapping[productName]['operations']
            G = nx.Graph()
            nodes, edges = self._generate_chain_connected_dag(numOps, G)
            # print('operations', nodes)
            # print('dependencies', edges)

            self.data['products']['items'][productName]['operations'] = selectedOps
            self.data['products']['items'][productName]['dependencies'] = edges
            self.data['products']['items'][productName]['arrival'] = \
            self.product_arrival_mapping[productName]["arrival_time"][0]
            self.data['products']['items'][productName]['duedate'] = \
            self.due_date_mapping[productName]["due_date_time"][0]

    def _compute_due_date_for_product(self, productName, strictness_weight=1.0):
        ''' Computes the due date for a product.

        - Since we currently assume parallel processing of operations, this method currently approximates the test time of the graph to be
        the test time of the heaviest path in the DAG.
        - Here, the heaviest path refers to the path having maximum setup plus test time.
        - Multiplies with the quantity and adds arrival time for an approximate optimistic due date.
        - Multiply by a weight that controls the strictness of the duedate (less weight => stricter duedate)
        '''
        # Compute Heaviest Path
        edge_list = [(u, v) for [u, v] in self.data['products']['items'][productName]['dependencies']]
        G = nx.from_edgelist(edge_list, create_using=nx.DiGraph)
        processing_time = self._find_most_weighted_path_in_the_graph(G, productName)

        # # Compute processing time of the longest path
        # longest_path = nx.dag_longest_path(G)
        # processing_time = 0
        # for op in longest_path:
        #     opName = self.data['products']['items'][productName]['operations'][op]
        #     opDetails = self.data['operations']['items'][opName]
        #     test_time = 0
        #     setup_time = 0
        #     for config in opDetails['compatibleConfigurations']:
        #         test_time += opDetails['estimatedTestTime'][config]['mean']
        #         setup_time += np.average(self.data['configurations']['items'][config]['setupTimes'])
        #     test_time = test_time / len(opDetails['compatibleConfigurations'])
        #     setup_time = setup_time / len(opDetails['compatibleConfigurations'])
        #     processing_time += (test_time + setup_time)

        # Approx. due date
        due_date = self.data['products']['items'][productName]['arrival']
        quantity_per_tester = self.data['products']['items'][productName]['quantity'] / self.data['testers']['count']
        due_date += processing_time * (1 if quantity_per_tester < 1 else quantity_per_tester)

        return due_date * strictness_weight

    def _find_weight_of_path(self, path, productName):
        ''' For the given path from added source (-1) to added sink (-2), we compute the sum of setup plus test time of all the modes except source and sink
        '''
        weight = 0
        for node in path:
            if node == -1 or node == -2:
                continue

            opName = self.data['products']['items'][productName]['operations'][node]
            opDetails = self.data['operations']['items'][opName]
            test_time = 0
            setup_time = 0
            for config in opDetails['compatibleConfigurations']:
                test_time += opDetails['estimatedTestTime'][config]['mean']
                setup_time += np.average(self.data['configurations']['items'][config]['setupTimes'])
            test_time = test_time / len(opDetails['compatibleConfigurations'])
            setup_time = setup_time / len(opDetails['compatibleConfigurations'])
            weight += (test_time + setup_time)

        return weight

    def _find_most_weighted_path_in_the_graph(self, G, productName):
        ''' Finds the most weighted path from added source (-1) to added sink (-2) in the provided product graph.

        - Here weight refers to the setup plus test time of the path.
        '''

        sources = [x for x in G.nodes() if G.in_degree(x) == 0]
        targets = [x for x in G.nodes() if G.out_degree(x) == 0]

        # Added dummy source (-1) and dummy sink (-2).
        G.add_node(-1)
        G.add_node(-2)

        # Add edges between dummy source to all original sources
        for source in sources:
            G.add_edge(-1, source)

        # Add edges from all original targets to dummy sink
        for target in targets:
            G.add_edge(target, -2)

        weightest_path = max((path for path in nx.all_simple_paths(G, -1, -2)),
                             key=lambda path: self._find_weight_of_path(path, productName))

        # Remove added dummy nodes and corresponding edges
        for source in sources:
            G.remove_edge(-1, source)
        for target in targets:
            G.remove_edge(target, -2)
        G.remove_node(-1)
        G.remove_node(-2)

        return self._find_weight_of_path(weightest_path, productName)

    def _generate_operations(self):
        ''' Generates the operations section of data.

        - The number of operations should be in the range ['minOperations', 'maxOperations'].
        - The distribution of estimatedTestTime is assumed to be 'normal'
        - The mean of estimatedTestTime distribution should be in the range ['minMeanEstimatedTestTime', 'maxMeanEstimatedTestTime']
        - The std of estimatedTestTime distribution should be in the range ['minStdEstimatedTestTime', 'maxStdEstimatedTestTime']
        - The number of supportedConfigurations should be in the range ['minSupportedConfigurationsPerOperation', 'maxSupportedConfigurationsPerOperation'],
            if numOfConfigs is less than min or max supported configurations per operation, numOfConfigs should be used.
        '''
        self.numOfOperations = len(self.operation_config_mapping)
        ## use product operation test time double mapping to correctly
        ## assign the unique test times for each product operation pair
        ## to the correct product config for each distinct operation
        ## product remains the same but the total time is different under each operation

        self.data['operations'] = {
            'count': self.numOfOperations,
            'items': {}
        }

        for ops in self.operation_config_mapping.keys():
            # print(ops)
            config_list = self.operation_config_mapping[ops]['compatibleConfig']
            product_list = self.operation_product_mapping[ops]['products']
            print(config_list)
            print(ops, product_list)
            # compatible_configs = [config for config in config_list if config in self.unique_configurations]
            self.data['operations']['items'][ops] = {
                'estimatedTestTime': {},
                'compatibleConfigurations': config_list,
                'compatibleProducts': product_list

            }

        # compatible_config_count = sum(len(entry['compatibleConfigurations']) for entry in self.data['operations']['items'].values())
        # print("Number of compatible configurations:", compatible_config_count)

        # Generating estimatedTestTime for operations on corresponding products
        # changed it from the earlier implementation
        # Generating estimatedTestTime for operations on corresponding configurations

        minStdEstimatedTestTime = self.config.get('minStdEstimatedTestTime')
        maxStdEstimatedTestTime = self.config.get('maxStdEstimatedTestTime')
        for op in self.data['operations']['items'].keys():
            # print(op)
            for product in product_list:
                # print(product)
                # print(self.product_operation_testtime_mapping[product])
                for product, inner_dict in self.product_operation_testtime_mapping.items():
                    # print('product inner dict pair', product, inner_dict)
                    for operation_keys in inner_dict['estimated_time'].keys():
                        # print('operation within inner dict', operation_keys)
                        if op in operation_keys:
                            # print(self.product_operation_testtime_mapping[product]["estimated_time"][op])

                            self.data['operations']['items'][op]['estimatedTestTime'][product] = {
                                'distribution': 'normal',
                                'mean': self.product_operation_testtime_mapping[product]["estimated_time"][op],
                                'std': random.uniform(minStdEstimatedTestTime, maxStdEstimatedTestTime)
                            }
        print(self.data['operations']['items']['O140'])

    def _generate_testers(self):
        ''' Generates the testers section of data.

        - The number of testers should be in the range ['minTesters', 'maxTesters'].
        - The tester names are prefixed with 'T', followed by the random index assigned to it + 1.
        - The number of supported configurations for a tester should be in the range ['minSupportedConfigurationsPerTester', 'maxSupportedConfigurationsPerTester'],
            if numOfConfigs is less than min or max supported configurations per tester, numOfConfigs should be used.
        '''

        self.numOfTesters = len(self.unique_testers)
        self.data['testers'] = {
            'count': self.numOfTesters,
            'items': {
                testers: {
                    'supportedConfigurations': self.resource_config_mapping[testers]['configuration']

                } for testers in self.unique_testers
            }
        }

        # self._verify_and_add_unassigned_configurations()

    def _verify_and_add_unassigned_configurations(self):
        ''' This checks if any configuration is left unassigned to any tester and adds it to a randomly selected tester.
        This has to be done since every configuration should be assigned to atleast one tester.
        '''
        configs = list(self.data['configurations']['items'].keys())
        for config in configs:
            found = False
            for testerDetails in self.data['testers']['items'].values():
                if config in testerDetails['supportedConfigurations']:
                    found = True
                    break
            if not found:
                # print(list(self.data['testers']['items'].keys()))
                tester = random.choice(list(self.data['testers']['items'].keys()))
                self.data['testers']['items'][tester]['supportedConfigurations'].append(config)

    def _generate_configurations(self):
        ''' Generates the configurations section of data.

        - The number of configurations should be in the range ['minConfigurations', 'maxConfigurations']
        - The configuration names are prefixed with 'K', followed by the random index assigned to it + 1.
        - The 'setupTimes' of configurations should be in the range ['minSetupTime', 'maxSetupTime']
        - The setupTime from a configuration to the same configuration should be 0.
        '''

        self.numOfConfigs = len(self.unique_configurations)
        self.data['configurations'] = {
            'count': self.numOfConfigs,
            'items': {
                config: {
                    'index': index,
                    'setupTimes': self._generate_setup_times(index)
                } for index, config in enumerate(self.unique_configurations)
            }
        }

    def _generate_setup_times(self, index):
        ''' Generates the setup times for a particular configuration.

        - The 'setupTimes' of configurations should be in the range ['minSetupTime', 'maxSetupTime']
        - The setupTime from a configuration to the same configuration should be 0.
        '''
        minSetupTime = self.config.get('minSetupTime')
        maxSetupTime = self.config.get('maxSetupTime')

        setupTimes = [random.randint(minSetupTime, maxSetupTime) for _ in range(self.numOfConfigs)]
        setupTimes[index] = 0

        return setupTimes

    def _generate_random_connected_dag(self, numOfNodes, percentOfEdges):
        ''' Generates a random connected directed acyclic graph containing 'numOfNodes' number of nodes and possibly 'percentOfEdges' percent of total possible edges.

        Procedure:
        - Create a full lower triangular matrix (assured to be DAG)
        - Randomly remove edges if that edge doesn't loose the connectivity and DAG-ness of the graph
        '''
        adj_mat = np.tri(numOfNodes, numOfNodes, -1)

        G = nx.from_numpy_array(adj_mat, create_using=nx.DiGraph)

        totalNumOfEdges = (numOfNodes * (numOfNodes - 1)) / 2
        edgesThreshold = percentOfEdges * totalNumOfEdges
        triesThreshold = totalNumOfEdges
        numOfTries = 0

        while G.number_of_edges() > edgesThreshold and numOfTries < triesThreshold:
            edges = list(G.edges)

            random_edge = random.choice(edges)

            G.remove_edge(random_edge[0], random_edge[1])

            if not (nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G)):
                G.add_edge(random_edge[0], random_edge[1])

            numOfTries += 1

        nodes = list(G.nodes)
        edges = list(G.edges)

        return nodes, edges

    def _generate_chain_connected_dag(self, numOfNodes, G):
        ''' Generates a random connected directed acyclic graph containing 'numOfNodes' number of nodes and possibly 'percentOfEdges' percent of total possible edges.

        Procedure:
        - Create a full lower triangular matrix (assured to be DAG)
        - Randomly remove edges if that edge doesn't loose the connectivity and DAG-ness of the graph
        '''

        for i in range(numOfNodes):
            G.add_node(int(i))

        for i in range(numOfNodes - 1):
            G.add_edge(int(i), int(i + 1))

        print(G)
        nodes = list(G.nodes)
        edges = list(G.edges)

        return nodes, edges


if __name__ == "__main__":
    ## change this to read the float portion of the dataset
    config = {


        'dirPath': 'intermediate_files',
        'productPath': 'D_0.json',
        'configPath': 'compatible_file.json',
        'originalconfigPath': 'compatible_file.json',
        'purpose': 'train',
        'numOfStaticConfigurationFiles': 1,

        #
        # The estimated avg standard deviation of test time of an operation
        'minStdEstimatedTestTime': 2,
        'maxStdEstimatedTestTime': 5,
        # The setup time needed to change from one configuration/mode to another.
        'minSetupTime': 0,
        'maxSetupTime': 0,


    }
    unique_testers, unique_configurations, product_pn_mapping = _process_data('D_0.json',
                                                                           'compatible_new_file.json',
                                                                           'extra_new_file.json'
                                                                           )

    datagen = DataGenerator(config, unique_testers, unique_configurations, product_pn_mapping)
    datagen.generate_and_save()
