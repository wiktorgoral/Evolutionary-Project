import multiparentcrossover
from jmetal.core.observer import Observer
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import BinaryTournamentSelection, PolynomialMutation, UniformMutation
from jmetal.util.termination_criterion import StoppingByEvaluations
from scipy import stats
import scikit_posthocs as sp
import numpy as np
import matplotlib.pyplot as plt


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


def evaluate(crosssover_algo, problem):
  alldata = []
  series = []
  for x in range(10):
    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables, 20.0),
        crossover=crosssover_algo,
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max_evaluations=500000)
    )
    data = []
    dataobserver = DataObserver(1.0, data)
    algorithm.observable.register(observer=dataobserver)
    algorithm.run()
    result = algorithm.get_result().objectives[0]
    series.append(result) 
    alldata.append(data)

  numpy_array = np.array(alldata)
  transpose = numpy_array.T
  transpose_list = transpose.tolist()

    
  fig = plt.figure(figsize =(60, 42)) 
     
  ax = fig.add_axes([0, 0, 1, 1]) 
    
  bp = ax.boxplot(transpose_list) 
   
  plt.show()

  print(stats.kruskal(transpose_list[0],transpose_list[1],transpose_list[-1]))
  series = [series] 
  print(np.average(series))

  sp.posthoc_dunn([transpose_list[0],transpose_list[1],transpose_list[-1]], p_adjust = 'holm')


problem = Rastrigin(10)


print("DifferentialEvolutionCrossover")
evaluate(multiparentcrossover.DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5), problem)

print("SelectiveMultiParentCrossover")
evaluate(multiparentcrossover.SelectiveMultiParentCrossover(CR=1.0, F=0.5, K=0.5, PN=5), problem)

print("RandomMultiParentCrossoverWithStep")
evaluate(multiparentcrossover.RandomMultiParentCrossoverWithStep(PN = 5), problem)

print("RandomMultiParentCrossover")
evaluate(multiparentcrossover.RandomMultiParentCrossover(PN = 5), problem)
