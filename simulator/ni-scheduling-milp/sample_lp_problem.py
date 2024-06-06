############################################################
# Maximize 3x + y subject to the following constraints:
#    0 ≤ x ≤ 1
#    0 ≤ y ≤ 2
#    x + y ≤ 2
############################################################


# Imports
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit


def main():
    # Create the linear solver with the GLOP backend.
    # pywraplp is a Python wrapper for the underlying C++ solver. The argument "GLOP" specifies GLOP, the OR-Tools linear solver.
    solver = pywraplp.Solver.CreateSolver('GLOP')
    if not solver:
        return

    # Create the variables x and y.
    x = solver.NumVar(0, 1, 'x')
    y = solver.NumVar(0, 2, 'y')

    print('Number of variables =', solver.NumVariables())

    # Create a linear constraint, 0 <= x + y <= 2.
    # The method SetCoefficient sets the coefficients of x and y in the expression for the constraint.
    ct = solver.Constraint(0, 2, 'ct')
    ct.SetCoefficient(x, 1)
    ct.SetCoefficient(y, 1)

    print('Number of constraints =', solver.NumConstraints())

    # Create the objective function, 3 * x + y.
    # The method SetMaximization declares this to be a maximization problem.
    objective = solver.Objective()
    objective.SetCoefficient(x, 3)
    objective.SetCoefficient(y, 1)
    objective.SetMaximization()

    # Invoke the solver and display the results
    solver.Solve()

    print('Solution:')
    print('Objective value =', objective.Value())
    print('x =', x.solution_value())
    print('y =', y.solution_value())


if __name__ == '__main__':
    pywrapinit.CppBridge.InitLogging('sample_lp_problem.py')
    cpp_flags = pywrapinit.CppFlags()
    cpp_flags.logtostderr = True
    cpp_flags.log_prefix = False
    pywrapinit.CppBridge.SetFlags(cpp_flags)

    main()
