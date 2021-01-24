from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from main import Emas

if __name__ == '__main__':
    problem = Rastrigin(100)

    algorithm = Emas(
        number_of_islands=10,
        init_island_population_size=20,
        problem=problem,
        mutation=PolynomialMutation(1.0/problem.number_of_variables, 0.2),
        crossover=SBXCrossover(0.9, 20.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=20000)
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    for x in result:
        for z in x:
            print('Solution: {}'.format(z.solution.variables))
            print('Fitness: {}'.format(z.solution.objectives[0]))
            print('Energy: {}'.format(z.energy))
    print('Computing time: {}'.format(algorithm.total_computing_time))
    print(len(result))
    print("Best fitness: " + str(min([z.fitness() for x in result for z in x])))