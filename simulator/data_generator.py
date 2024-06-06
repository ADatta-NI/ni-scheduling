import random
import json
import networkx as nx
import numpy as np
from constants import DATA_GENERATOR_TRAIN_SEED, DATA_GENERATOR_TEST_SEED 

# add job setup time here so that it can be collected later 
# visualise all possible time related variables 
# cyclomatic complexity of the dependency trees
# no of jobs count of job names 
# ratio of longest to shortest test times 
# variance of test times 
# tightness of the cap for the resources 
# arrival time 
# both measure and record 
class DataGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.dirPath = self.config.get('dirPath')
        self.purpose = self.config.get('purpose')
        random.seed(DATA_GENERATOR_TRAIN_SEED if self.purpose == 'train' else DATA_GENERATOR_TEST_SEED)

    def generate_and_save(self):
        self._num_of_static_configuration_files = self.config.get('numOfStaticConfigurationFiles')
        for idx in range(self._num_of_static_configuration_files):
            self._generate_and_save_single_config(idx + 1)
        

    def _generate_and_save_single_config(self, idx):
        # Initialize data 
        self.data = {}

        # Generate configurations
        self._generate_configurations()

        # Generate testers
        self._generate_testers()

        # Generate operations
        self._generate_operations()

        # Generate products
        self._generate_products()

        filePath = self.dirPath + 'static_configuration_' + str(idx) + '.json'
        with open(filePath, 'w') as f:
            json.dump(self.data, f)


    def _generate_products(self):
        ''' Generates the products section of data.

        - The number of products should be in the range ['minProducts', 'maxProducts'].
        - The number of operations for a product should be in the range ['minNumOperationsPerProduct', 'maxNumOperationsPerProduct'], if numOfOperations is less than min or max of operationsPerProduct, then numOfOperations should be considered.
        - The number of edges in each product graph should be tried to be fit in this percentage range ['minProductEdgesPercent', 'maxProductEdgesPercent'], where percentage is with respect to a complete graph.
        '''
        self.numOfProducts = random.randint(self.config.get('minProducts'), self.config.get('maxProducts'))

        self.data['products'] = {
            'count': self.numOfProducts,
            'items': {
                'P' + str(index + 1) : {
                    'index': index,
                    'arrival': '',
                    'quantity': random.randint(self.config.get('minQuantity'), self.config.get('maxQuantity')),
                    'duedate': '',
                    'operations': '',
                    'dependencies': []
                } for index in range(self.numOfProducts)
            }
        }

        # Generating product graphs and computing due dates
        ops = list(self.data['operations']['items'].keys())
        arrival = 0
        for productName in self.data['products']['items'].keys():
            numOps = random.randint(min(self.numOfOperations, self.config.get('minNumOperationsPerProduct')), min(self.numOfOperations, self.config.get('maxNumOperationsPerProduct')))
            edgePercent = random.uniform(self.config.get('minProductEdgesPercent'), self.config.get('maxProductEdgesPercent'))
            selectedOps = random.sample(ops, numOps)

            nodes, edges = self._generate_random_connected_dag(numOps, edgePercent)

            self.data['products']['items'][productName]['operations'] = selectedOps
            self.data['products']['items'][productName]['dependencies'] = edges
            self.data['products']['items'][productName]['arrival'] = arrival
            self.data['products']['items'][productName]['duedate'] = self._compute_due_date_for_product(productName)
            
            
            arrival += random.uniform(self.config.get('minArrivalTimeGap'), self.config.get('maxArrivalTimeGap'))


    def _compute_due_date_for_product(self, productName, strictness_weight=0.9):
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
        # record the weights in a dataset
        return weight


    def _find_most_weighted_path_in_the_graph(self, G, productName):
        ''' Finds the most weighted path from added source (-1) to added sink (-2) in the provided product graph.

        - Here weight refers to the setup plus test time of the path.
        '''
        
        sources = [x for x in G.nodes() if G.in_degree(x)==0]
        targets = [x for x in G.nodes() if G.out_degree(x)==0]

        # Added dummy source (-1) and dummy sink (-2). 
        G.add_node(-1)
        G.add_node(-2)

        # Add edges between dummy source to all original sources
        for source in sources:
            G.add_edge(-1, source)

        # Add edges from all original targets to dummy sink
        for target in targets:
            G.add_edge(target, -2)

        weightest_path = max((path for path in nx.all_simple_paths(G, -1, -2)), key=lambda path: self._find_weight_of_path(path, productName))

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
        self.numOfOperations = random.randint(self.config.get('minOperations'), self.config.get('maxOperations'))

        minSupportedConfigurationsPerOperation = min(self.config.get('minSupportedConfigurationsPerOperation'), self.numOfConfigs)
        maxSupportedConfigurationsPerOperation = min(self.config.get('maxSupportedConfigurationsPerOperation'), self.numOfConfigs)
        self.data['operations'] = {
            'count': self.numOfOperations,
            'items': {
                'O' + str(index + 1) : {
                    'estimatedTestTime': {},
                    'compatibleConfigurations': [
                        'K' + str(config + 1) for config in random.sample(range(self.numOfConfigs), random.randint(minSupportedConfigurationsPerOperation, maxSupportedConfigurationsPerOperation))
                    ]
                } for index in range(self.numOfOperations)
            }
        }

        # Generating estimatedTestTime for operations on corresponding configurations
        minMeanEstimatedTestTime = self.config.get('minMeanEstimatedTestTime')
        maxMeanEstimatedTestTime = self.config.get('maxMeanEstimatedTestTime')
        minStdEstimatedTestTime = self.config.get('minStdEstimatedTestTime')
        maxStdEstimatedTestTime = self.config.get('maxStdEstimatedTestTime')
        for op in self.data['operations']['items'].keys():
            for config in self.data['operations']['items'][op]['compatibleConfigurations']:
                self.data['operations']['items'][op]['estimatedTestTime'][config] = {
                    'distribution': 'normal',
                    'mean': random.uniform(minMeanEstimatedTestTime, maxMeanEstimatedTestTime),
                    'std': random.uniform(minStdEstimatedTestTime, maxStdEstimatedTestTime)
                }


    def _generate_testers(self):
        ''' Generates the testers section of data.

        - The number of testers should be in the range ['minTesters', 'maxTesters'].
        - The tester names are prefixed with 'T', followed by the random index assigned to it + 1.
        - The number of supported configurations for a tester should be in the range ['minSupportedConfigurationsPerTester', 'maxSupportedConfigurationsPerTester'], 
            if numOfConfigs is less than min or max supported configurations per tester, numOfConfigs should be used.
        '''
        self.numOfTesters = random.randint(self.config.get('minTesters'), self.config.get('maxTesters'))

        minSupportedConfigurationsPerTester = min(self.config.get('minSupportedConfigurationsPerTester'), self.numOfConfigs)
        maxSupportedConfigurationsPerTester = min(self.config.get('maxSupportedConfigurationsPerTester'), self.numOfConfigs)
        self.data['testers'] = {
            'count': self.numOfTesters,
            'items': {
                'T' + str(index + 1) : {
                    'supportedConfigurations': [
                        'K' + str(config + 1) for config in random.sample(range(self.numOfConfigs), random.randint(minSupportedConfigurationsPerTester, maxSupportedConfigurationsPerTester))
                    ] 
                } for index in range(self.numOfTesters)
            }
        }

        self._verify_and_add_unassigned_configurations()
    

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
                tester = random.choice(list(self.data['testers']['items'].keys()))
                self.data['testers']['items'][tester]['supportedConfigurations'].append(config)


    def _generate_configurations(self):
        ''' Generates the configurations section of data.

        - The number of configurations should be in the range ['minConfigurations', 'maxConfigurations']
        - The configuration names are prefixed with 'K', followed by the random index assigned to it + 1.
        - The 'setupTimes' of configurations should be in the range ['minSetupTime', 'maxSetupTime']
        - The setupTime from a configuration to the same configuration should be 0.
        '''
        self.numOfConfigs = random.randint(self.config.get('minConfigurations'), self.config.get('maxConfigurations'))

        self.data['configurations'] = {
            'count': self.numOfConfigs,
            'items': {
                'K' + str(index + 1) : {
                    'index': index,
                    'setupTimes': self._generate_setup_times(index)
                } for index in range(self.numOfConfigs)
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

            if not(nx.is_weakly_connected(G) and nx.is_directed_acyclic_graph(G)):
                G.add_edge(random_edge[0], random_edge[1])

            numOfTries += 1

        nodes = list(G.nodes)
        edges = list(G.edges)
        ## add nodes and edges to a dataset for analysis 
        return nodes, edges


if __name__ == "__main__":
    config = {
        'dirPath': 'data/',
        'purpose': 'train',
        'numOfStaticConfigurationFiles': 100,

        # The number of configurations/modes available in the system (across all testers)
        'minConfigurations': 2,
        'maxConfigurations': 20 ,

        # The setup time needed to change from one configuration/mode to another.
        'minSetupTime': 0,
        'maxSetupTime': 10,

        # The number of testers in the system
        'minTesters': 10,
        'maxTesters': 40,

        # The number of configurations/modes per tester
        'minSupportedConfigurationsPerTester': 2,
        'maxSupportedConfigurationsPerTester': 20,

        # The number of unique operations/tests that can be performed in the system (across all configurations)
        'minOperations': 100,
        'maxOperations': 200,

        # The estimated avg test time of an operation
        'minMeanEstimatedTestTime': 2,
        'maxMeanEstimatedTestTime': 500,

        # The estimated avg standard deviation of test time of an operation
        'minStdEstimatedTestTime': 2,
        'maxStdEstimatedTestTime': 5,

        # Context: An operation/test can be performed using different configurations.
        # The number of configurations which can support an operation
        'minSupportedConfigurationsPerOperation': 5,
        'maxSupportedConfigurationsPerOperation': 15,

        # The number of products to be considered for the schedule (unique test entities / product graphs)
        'minProducts': 2,
        'maxProducts': 10,

        # The number of items of each product which are requested for testing.
        'minQuantity': 5,
        'maxQuantity': 20,

        # The number of test operations/nodes needed to complete testing of a product
        'minNumOperationsPerProduct': 5,
        'maxNumOperationsPerProduct': 15,

        # The arrival time distribution of the products (how frequently does the new products arrive)
        'minArrivalTimeGap': 50,
        'maxArrivalTimeGap': 100,

        # Context: If there are n nodes, a complete graph contains n*(n-1)/2 nodes (less than that for it to be a DAG, but we'll consider this),
        # But in a product graph we won't have all possible dependencies.
        # These attributes capture the minimum and maximum percentage of connectivity of product graphs which will is used for generating
        # different sparsely connected directed acyclic graphs.
        #
        # Ex: If there are 5 nodes (tests) needed for testing a product, how many dependencies could be there?
        # (Here a dependency between ('a' and 'b') means 'b' can be performed only once 'a' is done)
        #
        # The number of edges percentage
        'minProductEdgesPercent': 0.25,
        'maxProductEdgesPercent': 0.5,

    }

    datagen = DataGenerator(config)
    datagen.generate_and_save()
