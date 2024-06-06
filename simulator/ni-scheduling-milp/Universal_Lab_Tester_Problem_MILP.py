# Import the linear solver wrapper
from ortools.linear_solver import pywraplp
from ortools.init.python import init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps

# Configurations
DATA_ROOT = 'small/'
PT_FILE_NAME = DATA_ROOT + 'pt_jo_mk.data'
RT_FILE_NAME = DATA_ROOT + 'rt_m_k1k.data'
INIT_RT_FILE_NAME = DATA_ROOT + 'rt_m_0k.data'
D_FILE_NAME = DATA_ROOT + 'd_j.data'
COMPAT_CONFIG_FILE_NAME = DATA_ROOT + 'k_jo.data'
PRECEDENCE_SET_FILE_NAME = DATA_ROOT + 'E_jo.data'
J = 2                       # Number of DUTs - J
O = 4                       # Number of tests per job - O
M = 3                       # Number of Testers - M
K = 3                       # Number of personalities - K

# Declare the MIP solver
solver = pywraplp.Solver.CreateSolver('SCIP')

# p_jo_mk - Read processing times - Time taken to test test-o of DUT-J on tester-M with personality-R 
def read_test_times():
    p_jo_mk = np.empty((0, M, K))
    f = open(PT_FILE_NAME, 'r')
    for line in f.readlines():
        l = np.array([float(x) for x in line.split('\t')])
        l = np.reshape(l, (M, K))
        p_jo_mk = np.append(p_jo_mk, np.array([l]), axis=0)
    p_jo_mk = np.reshape(p_jo_mk, (J, O, M, K))
    return p_jo_mk

# d_j - Read duedates of jobs - Due date of DUT-J
def read_due_dates():
    d_j = np.empty((0))
    f = open(D_FILE_NAME, 'r')
    for line in f.readlines():
        d = float(line)
        d_j = np.append(d_j, d)
    return d_j

# rt_m_kdashk - Read reconfiguration times - Time taken to change a personality from k1 to k on a tester
def read_reconfig_times():
    rt_m_k1k = np.empty((0, K, K))
    f = open(RT_FILE_NAME, 'r')
    for line in f.readlines():
        l = np.array([float(x) for x in line.split('\t')])
        l = np.reshape(l, (K, K))
        rt_m_k1k = np.append(rt_m_k1k, np.array([l]), axis=0)
    rt_m_k1k = np.reshape(rt_m_k1k, (M, K, K))
    return rt_m_k1k 

# rt_m_0k - Read initial reconfiguration times - Time taken to configure a personality on a plain tester with no prev personality
def read_init_reconfig_times():
    rt_m_0k = np.empty((0, K))
    f = open(INIT_RT_FILE_NAME, 'r')
    for line in f.readlines():
        l = np.array([float(x) for x in line.split('\t')])
        rt_m_0k = np.append(rt_m_0k, np.array([l]), axis=0)
    rt_m_0k = np.reshape(rt_m_0k, (M, K))
    return rt_m_0k

# k_jo - Read compatible configuration set - The set of configurations compatible to execute the test jo
def read_compatible_config_set():
    k_jo = np.empty((0))
    f = open(COMPAT_CONFIG_FILE_NAME, 'r')
    for line in f.readlines():
        l = {int(x) - 1 for x in line.split('\t')}
        k_jo = np.append(k_jo, np.array([l]), axis=0)
    k_jo = np.reshape(k_jo, (J, O))
    return k_jo

# E_jo - Read precedence contraint set - The set of operations in this job that have to end before test jo can start
def read_precedence_set():
    E_jo = np.empty((0))
    f = open(PRECEDENCE_SET_FILE_NAME, 'r')
    for line in f.readlines():
        l = {int(x) - 1 for x in line.split('\t') if x != '\n'}
        E_jo = np.append(E_jo, np.array([l]), axis=0)
    E_jo = np.reshape(E_jo, (J, O))
    return E_jo

def var(name):
    return solver.LookupVariable(name)

def find_machine_scheduled(j, o, k_jo):
    for m in range(M):
        for k in k_jo[j][o]:
            if var(f'z_{j}_{o}_{m}_{k}').solution_value():
                return m, k

def print_result(k_jo):
    for j in range(J):
        for o in range(O):
            print(f"Operation: ({j}, {o})", f"Machine: {find_machine_scheduled(j, o, k_jo)}", f"Start: {var(f's_{j}_{o}').solution_value()}", f"RT: {var(f'RT_{j}_{o}').solution_value()}", f"PT: {var(f'PT_{j}_{o}').solution_value()}")

def main(): 
    if not solver:
        return

    # Parameters
    G = 10000                                               # Big +ve integer number - G
    p_jo_mk = read_test_times()                             # Test times (4 dim array j-o-m-k)
    rt_m_k1k = read_reconfig_times()                        # Reconfiguration times (3 dim array m-k1-k)
    rt_m_0k = read_init_reconfig_times()                    # Initial Reconfiguration times (2 dim array m-k)
    k_jo = read_compatible_config_set()                     # Compatibility Cofigurations set (2 dim array j-o containing a set)
    E_jo = read_precedence_set()                            # Precedence set (2 dim array j-o containing a set)
    D_j = read_due_dates()                                  # Due dates of jobs (1 dim array j containing floats)

    # Define the variables
    infinity = solver.infinity()
    solver.Var(0, infinity, False, 'obj_var')               # Decision Variable to be used for building objective
    for j in range(J):
        solver.Var(0, infinity, False, f'C_{j}')             # Completion Time C_j of a job j
        solver.Var(0, infinity, False, f'T_{j}')             # Tardiness T_j of a job j
        for o in range(O):
            solver.Var(0, infinity, False, f's_{j}_{o}')     # Start Time s_jo
            solver.Var(0, infinity, False, f'RT_{j}_{o}')    # Reconfiguration Time taken by jo
            solver.Var(0, infinity, False, f'PT_{j}_{o}')    # Processing Time taken by jo
            for m in range(M):
                solver.BoolVar(f'ybar_{j}_{o}_{m}')          # Ybar ybar_jo_m - True if jo is the first test on tester m
                solver.BoolVar(f'ycap_{j}_{o}_{m}')          # Ycap ycap_jo_m - True if jo is the last test on tester m
                for k in range(K):
                    solver.BoolVar(f'z_{j}_{o}_{m}_{k}')      # Z z_jo_mk      - True if jo is tested on mk
                for jdash in range(J):
                    for odash in range(O):
                        if (j == jdash) and (o == odash):
                            continue
                        solver.BoolVar(f'y_{j}_{o}_{jdash}_{odash}_{m}')      # Y_jo_jdashodash_m - True if jdashodash follows jo o tester m

    print("All Variables added!")
    
    # Define the constraints,
    ## Objective Decision Variable related constraints
    # ### Makespan
    # for j in range(J):
    #     for o in range(O):
    #         solver.Add(var('obj_var') >= (var(f's_{j}_{o}') + var(f'PT_{j}_{o}')), f'Objective_Constraint_{j}_{o}')
    ### Tardiness
    tardiness = 0
    for j in range(J):
        tardiness += var(f'T_{j}')
    solver.Add(var('obj_var') >= tardiness, f'Objective_Constraint')

    ## Constraint 1: Each should should be performed only once
    for j in range(J):
        for o in range(O):
            sum = 0
            for m in range(M):
                for k in k_jo[j][o]:
                    sum += var(f'z_{j}_{o}_{m}_{k}')
            solver.Add(sum == 1, f'Constraint_1_{j}_{o}')

    ## Constraint 2: Resource ordering constraint on a tester
    for m in range(M):
        for j in range(J):
            for o in range(O):
                for jdash in range(J):
                    for odash in range(O):
                        if (j == jdash) and (o == odash):
                            continue
                        solver.Add(var(f's_{j}_{o}') >= var(f's_{jdash}_{odash}') + var(f'PT_{jdash}_{odash}') + var(f'RT_{j}_{o}') + G * (var(f'y_{jdash}_{odash}_{j}_{o}_{m}') - 1), f'Constraint_2_{m}_{j}_{o}_{jdash}_{odash}')

    # New Constraint 13: Start time of the first operation is after its reconfiguration is done (edge-case for Constraint #2)
    for m in range(M):
        for j in range(J):
            for o in range(O):
                solver.Add(var(f's_{j}_{o}') >= var(f'RT_{j}_{o}') + G * (var(f'ybar_{j}_{o}_{m}') - 1), f'Constraint_13_{m}_{j}_{o}')

    ## Constraint 3: A test should either be the first one or have a predecessor on that tester
    for m in range(M):
        for j in range(J):
            for o in range(O):
                ysum = 0
                zsum = 0
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
                ysum += var(f'ybar_{j}_{o}_{m}')
                for jdash in range(J):
                    for odash in range(O):
                        if (j == jdash) and (o == odash): 
                            continue
                        ysum += var(f'y_{jdash}_{odash}_{j}_{o}_{m}')
                solver.Add(ysum == zsum, f'Constraint_3_{m}_{j}_{o}')

    ## Constraint 4: A test should either be the last one or have a successor on that tester
    for m in range(M):
        for j in range(J):
            for o in range(O):
                ysum = 0
                zsum = 0
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
                ysum += var(f'ycap_{j}_{o}_{m}')
                for jdash in range(J):
                    for odash in range(O):
                        if (j == jdash) and (o == odash):
                            continue
                        ysum += var(f'y_{j}_{o}_{jdash}_{odash}_{m}')
                solver.Add(ysum == zsum, f'Constraint_4_{m}_{j}_{o}')

    ## Constraint 5: If a tester is used at all, it should have some starting operation
    for m in range(M):
        zsum = 0
        ysum = 0
        for j in range(J):
            for o in range(O):
                ysum += var(f'ybar_{j}_{o}_{m}')
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
        solver.Add(G * ysum >= zsum, f'Constraint_5_{m}')

    ## Constraint 6: If a tester is used at all, it should have some ending operation
    for m in range(M):
        zsum = 0
        ysum = 0
        for j in range(J):
            for o in range(O):
                ysum += var(f'ycap_{j}_{o}_{m}')
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
        solver.Add(G * ysum >= zsum, f'Constraint_6_{m}')

    ## Constraint 7: On a tester, at max there should be only one starting operation
    for m in range(M):
        ysum = 0
        for j in range(J):
            for o in range(O):
                ysum += var(f'ybar_{j}_{o}_{m}')
        solver.Add(ysum <= 1, f'Constraint_7_{m}')

    ## Constraint 8: On a tester, at max there should be only one ending operation
    for m in range(M):
        ysum = 0
        for j in range(J):
            for o in range(O):
                ysum += var(f'ycap_{j}_{o}_{m}')
        solver.Add(ysum <= 1, f'Constraint_8_{m}')
        
    ## Constraint 9: Precedence constraint between two tests of a particular DUT where 2nd can't be performed before first
    for j in range(J):
        for o in range(O):
            for odash in E_jo[j][o]:
                solver.Add(var(f's_{j}_{o}') >= var(f's_{j}_{odash}') + var(f'PT_{j}_{odash}'), f'Constraint_9_{j}_{o}_{odash}')

    ## Constraint 10: Reconfiguration time of a test should satisfy this condition
    for m in range(M):
        for j in range(J):
            for o in range(O):
                for k in k_jo[j][o]:
                    for jdash in range(J):
                        for odash in range(O):
                            if (j == jdash) and (o == odash):
                                continue
                            for kdash in k_jo[jdash][odash]:
                                solver.Add(var(f'RT_{j}_{o}') >= rt_m_k1k[m][kdash][k] + G * (var(f'z_{j}_{o}_{m}_{k}') + var(f'z_{jdash}_{odash}_{m}_{kdash}') + var(f'y_{jdash}_{odash}_{j}_{o}_{m}') - 3), f'Constraint_10_{m}_{j}_{o}_{k}_{jdash}_{odash}_{kdash}')

    ## Constraint 11: Reconfiguration time of a test which is running first on a machine should satisfy this condition
    for m in range(M):
        for j in range(J):
            for o in range(O):
                for k in k_jo[j][o]:
                    solver.Add(var(f'RT_{j}_{o}') >= rt_m_0k[m][k] + G * (var(f'z_{j}_{o}_{m}_{k}') + var(f'ybar_{j}_{o}_{m}') - 2), f'Constraint_11_{m}_{j}_{o}_{k}')

    ## Constraint 12: Processing time of an operation should satisfy this condition
    for m in range(M):
        for j in range(J):
            for o in range(O):
                for k in k_jo[j][o]:
                    solver.Add(var(f'PT_{j}_{o}') >= p_jo_mk[j][o][m][k] + G * (var(f'z_{j}_{o}_{m}_{k}') - 1), f'Constraint_12_{m}_{j}_{o}_{k}')

    ## Constraint 14: (Needed to break circular dependency) If more than 2 tests are scheduled on a tester, a particular test can't be both the starting and ending one
    for m in range(M):
        zsum = 0
        for j in range(J):
            for o in range(O):
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
        for j in range(J):
            for o in range(O):
                solver.Add(G * (2 - var(f'ybar_{j}_{o}_{m}') - var(f'ycap_{j}_{o}_{m}')) >= (zsum - 1), f'Constraint_14_{m}_{j}_{o}')

    ## Constraint 15: There should be n-1 number of transitions on a tester where n tests are scheduled
    for m in range(M):
        ysum = 0
        zsum = 0
        for j in range(J):
            for o in range(O):
                for jdash in range(J):
                    for odash in range(O):
                        if (j == jdash) and (o == odash):
                            continue
                        ysum += var(f'y_{jdash}_{odash}_{j}_{o}_{m}')
                for k in k_jo[j][o]:
                    zsum += var(f'z_{j}_{o}_{m}_{k}')
        solver.Add(ysum == zsum - 1)

    ## Constraint 16: Tardiness T_j defining constraint
    for j in range(J):
        solver.Add(var(f'T_{j}') >= var(f'C_{j}') - D_j[j], f'Constraint_16_{j}')

    ## Constraint 17: Completion time C_j defining constraint
    for j in range(J):
        for o in range(O):
            solver.Add(var(f'C_{j}') >= var(f's_{j}_{o}') + var(f'PT_{j}_{o}'), f'Constraint_17_{j}_{o}')

    print("All Constraints added!")

    # Define the objective
    solver.Minimize(var('obj_var'))

    # Call the MIP solver
    print(f'Solving with {solver.SolverVersion()}')
    status = solver.Solve()

    # Display the solution
    if status == pywraplp.Solver.INFEASIBLE or status == pywraplp.Solver.UNBOUNDED or status == pywraplp.Solver.ABNORMAL or status == pywraplp.Solver.NOT_SOLVED:
        print('PROBLEM!!!!') 

    elif status == pywraplp.Solver.OPTIMAL:
        print('OPTIMAL Solution')
        print('Objective value =', solver.Objective().Value())
        print('Number of variables =', solver.NumVariables())
        print('Number of constraints =', solver.NumConstraints())
        
        # What are running on each machine:
        print("What are running on each machine:")
        for m in range(M):
            print(f"On Machine: {m}")
            for k in range(K):
                for j in range(J):
                    for o in range(O):
                        if var(f'z_{j}_{o}_{m}_{k}').solution_value():
                            print(f"{j}, {o}, {k}")
        
        # What is the starting operation on a machine:
        print("What is the starting operation on each machine")
        for m in range(M):
            lst = []
            for j in range(J):
                for o in range(O):
                    if var(f'ybar_{j}_{o}_{m}').solution_value():
                        lst.append((j, o))
            print(f"Starting Op on M: {m}:: ", lst)

        # What is the ending operation on a machine:
        print("What is the ending operation on each machine")
        for m in range(M):
            lst = []
            for j in range(J):
                for o in range(O):
                    if var(f'ycap_{j}_{o}_{m}').solution_value():
                        lst.append((j, o))
            print(f"Ending Op on M: {m}:: ", lst)

        # Checking dependencies on a machine:
        print("What are the dependencies on each machine:")
        for m in range(M):
            print(f"On Machine: {m}", end='\n')
            for jdash in range(J):
                for odash in range(O):
                    for j in range(J):
                        for o in range(O):
                            if (j == jdash) and (o == odash):
                                continue
                            if var(f"y_{jdash}_{odash}_{j}_{o}_{m}").solution_value():
                                print(f'({jdash}, {odash}) -> ({j}, {o})')
        
        # What is the tardiness for each of the jobs:
        print("What is the tardiness for each of the jobs")
        for j in range(J):
            print(f"For Job: {j}", var(f'T_{j}').solution_value())

        # Printing operation-wise results
        print("Operation wise results:")
        print_result(k_jo)

        solver.VerifySolution(0.0001, True)

        # Plotting the Gantt Chart
        fig, ax = plt.subplots()
        end_max = 0
        # Map integer values to a color from a colormap
        cmap = colormaps['viridis']
        norm = colors.Normalize(vmin=0, vmax=J)
        for j in range(J):
            for o in range(O):
                m, k = find_machine_scheduled(j, o, k_jo)
                s = var(f's_{j}_{o}').solution_value()
                rt = var(f'RT_{j}_{o}').solution_value()
                pt = var(f'PT_{j}_{o}').solution_value()
                end_max = max(end_max, rt + s + pt)

                # Add horizontal bars and labels
                ax.barh(y=m, width=rt, left=s-rt, height=0.5, color='yellow')
                ax.barh(y=m, width=pt, left=s, height=0.5, color=cmap(norm(j)))
                center_x = s
                center_y = m+0.3
                ax.text(x=center_x, y=center_y, s=str([j, o, k]), ha='center', va='bottom')

                # Set axis labels and limits
                ax.set_xlabel('Time')
                ax.set_ylabel('Machines')
                ax.set_ylim([-0.5, M-0.5])
                ax.set_xlim([0, end_max+1])

        # Save the plot
        plt.savefig(DATA_ROOT + 'myplot.png')

        # Show the plot
        plt.show()

    
    elif status == pywraplp.Solver.FEASIBLE:
        print('FEASIBLE Solution')
        print('Objective value =', solver.Objective().Value())

if __name__ == '__main__':
    init.CppBridge.init_logging("Universal_Lab_Tester_Problem_MILP.py")
    cpp_flags = init.CppFlags()
    cpp_flags.stderrthreshold = 0
    cpp_flags.log_prefix = False
    init.CppBridge.set_flags(cpp_flags)

    main()
    
    print('\nMetrics:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())
