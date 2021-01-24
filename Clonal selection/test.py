from clonalg import CloneAlg
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, SBXCrossover, UniformMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.observer import Observer
import matplotlib.pyplot as plt
import scikit_posthocs as sp
from scipy import stats
import numpy as np

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


def runProblem():
    alldata = []
    for x in range(10):
        algorithm = CloneAlg(
            problem=Rastrigin(50),
            population_size=100,
            offspring_population_size=200,
            mutation=UniformMutation(0.5),
            termination_criterion=StoppingByEvaluations(max_evaluations=5000)
        )
        data = []
        dataobserver = DataObserver(1.0, data)
        algorithm.observable.register(observer=dataobserver)
        algorithm.run()
        alldata.append(data)

    # transpose!
    numpy_array = np.array(alldata)
    transpose = numpy_array.T
    transpose_list = transpose.tolist()

    # print(transpose_list)

    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    bp = ax.boxplot(transpose_list)

    # show plot
    plt.show()

    # perform Kruskal-Wallis Test
    print(stats.kruskal(transpose_list[0], transpose_list[1], transpose_list[-1]))

    # perform dunn test
    sp.posthoc_dunn([transpose_list[0], transpose_list[1], transpose_list[-1]], p_adjust='holm')


runProblem()