############################################################
# Maximize x + 10y subject to the following constraints:
# 
#     x + 7y ≤ 17.5
#     0 ≤ x ≤ 3.5
#     0 ≤ y
#     x, y integers
############################################################

from ortools.linear_solver import pywraplp


def main():
    # The default OR-Tools MIP solver is SCIP.
    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver('SAT')
    if not solver:
        return

    infinity = solver.infinity()
    # x and y are integer non-negative variables.
    x = solver.IntVar(0.0, infinity, 'x')
    y = solver.IntVar(0.0, infinity, 'y')

    print('Number of variables =', solver.NumVariables())

    # The following code defines the constraints for the problem.
    # x + 7 * y <= 17.5.
    solver.Add(x + 7 * y <= 17.5)

    # x <= 3.5.
    solver.Add(x <= 3.5)

    print('Number of constraints =', solver.NumConstraints())

    # Maximize x + 10 * y.
    solver.Maximize(x + 10 * y)

    print(f'Solving with {solver.SolverVersion()}')
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        print('Solution:')
        print('Objective value =', solver.Objective().Value())
        print('x =', x.solution_value())
        print('y =', y.solution_value())
    else:
        print('The problem does not have an optimal solution.')

    print('\nAdvanced usage:')
    print('Problem solved in %f milliseconds' % solver.wall_time())
    print('Problem solved in %d iterations' % solver.iterations())
    print('Problem solved in %d branch-and-bound nodes' % solver.nodes())


if __name__ == '__main__':
    main()
