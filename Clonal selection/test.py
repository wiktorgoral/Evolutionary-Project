from clonalg import CloneAlg
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover, UniformMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.observer import Observer
from jmetal.operator.selection import RouletteWheelSelection, BestSolutionSelection
from jmetal.core.problem import FloatProblem, FloatSolution
import matplotlib.pyplot as plt
from jmetal.core.operator import Crossover
import scikit_posthocs as sp
from scipy import stats
import numpy as np
import math
import random, copy
from typing import List
import seaborn as sns
sns.set()


class Ackle(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super().__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-32.768 for _ in range(number_of_variables)]
        self.upper_bound = [32.768 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        n_rev = 1 / solution.number_of_variables
        x = solution.variables
        sum_of_sqr = 0
        sum_of_cos = 0

        for i in range(solution.number_of_variables):
            sum_of_sqr += x[i] * x[i]
            sum_of_cos += math.cos(2 * math.pi * x[i])

        result = -20 * math.exp(-0.2 * math.sqrt(n_rev * sum_of_sqr)) \
                 - math.exp(n_rev * sum_of_cos) + 20 + math.exp(1)

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return "Ackle's"


class Schwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super().__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 418.9829
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result -= x[i] * sin(sqrt(abs(x[i])))

        solution.objectives[0] = result
        return solution

    def get_name(self) -> str:
        return "Schwefel's"


class DiscreteCrossover(Crossover[FloatSolution, FloatSolution]):
    def __init__(self, probability: float):
        super(DiscreteCrossover, self).__init__(probability=probability)

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != 2:
            raise Exception('The number of parents is: {}, expected: 2'.format(len(parents)))

        offspring = [copy.deepcopy(parents[0]), copy.deepcopy(parents[1])]
        rand = random.random()

        if rand <= self.probability:
            permut_len = offspring[0].number_of_variables
            for i in range(permut_len):
                rand = random.random()
                offspring[0].variables[i] = parents[0].variables[i] if rand <= 0.5 else parents[1].variables[i]
                offspring[1].variables[i] = parents[0].variables[i] if rand > 0.5 else parents[1].variables[i]

        return offspring

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2

    def get_name(self):
        return 'Discrete crossover'


class DataObserver(Observer):

    def __init__(self, frequency: float = 1.0, data = []) -> None:
        """ Show the number of evaluations, best fitness and computing time.
        :param frequency: Display frequency. """
        self.display_frequency = frequency
        self.data = data

    def update(self, *args, **kwargs):
        computing_time = kwargs['COMPUTING_TIME']
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives
            self.data.append(fitness[0])


def solve(problem, cloning_param, mutation):
    final_data = []
    final_problem = problem(dim)
    final_mutation = mutation(mut_pb, 20)

    for x in range(repetitions):
        algorithm = CloneAlg(
            problem=final_problem,
            population_size=100,
            offspring_population_size=100,
            mutation=final_mutation,
            cloning_param=cloning_param,
            termination_criterion=StoppingByEvaluations(max_evaluations=5000)
        )
        data = []
        dataobserver = DataObserver(1.0, data)
        algorithm.observable.register(observer=dataobserver)
        algorithm.run()
        final_data.append(data)

    trans_list = np.array(final_data).T.tolist()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    bp = ax.boxplot(trans_list)
    plt.title("Problem: {0} benchmark, dim: {1}, cloning_param: {2}, mutation: {3}".format(
        final_problem.get_name(),
        dim,
        algorithm.get_cloning_param(),
        final_mutation.get_name()))
    plt.show()

    # Kruskal-Wallis and Dunn tests
    print(stats.kruskal(trans_list[0], trans_list[1], trans_list[-1]))
    sp.posthoc_dunn([trans_list[0], trans_list[1], trans_list[-1]], p_adjust='holm')


repetitions = 10
dim = 50
mut_pb = 0.8

[solve(problem, cloning_param, mutation)
 for problem in [Ackle, Schwefel]
 for cloning_param in [0.1, 0.3, 0.6, 0.9]
 for mutation in [UniformMutation, PolynomialMutation]]