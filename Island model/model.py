import numpy as np
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.operator import BinaryTournamentSelection, UniformMutation, SBXCrossover
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from enum import Enum
import random
from typing import TypeVar, List, Generic, Mapping
from copy import deepcopy
S = TypeVar('S')


class Mode(Enum):
    RING_EQUAL = 1
    RANDOM_EQUAL = 2
    RING_INEQUAL = 3
    RANDOM_INEQUAL = 4


class Migration(Generic[S]):
    def __init__(self, migration_rate: int, random_sized_groups: bool, random_destination: bool):
        assert migration_rate > 0 and migration_rate < 1
        self.migration_rate = migration_rate
        self.island_inhabitants = {}
        self.random_sized_groups = random_sized_groups
        self.random_destination = random_destination

    # zbieramy rozwiązania wykonanych algorytmów
    def collect_inhabitants(self, island: int, inhabitants: List[S]) -> None:
        self.island_inhabitants[str(island)] = inhabitants

    # mieszamy tablice, tak, by zawierały w sobie migrantów
    def allocate_migrants(self) -> Mapping[str, List[S]]:
        immigrants = []

        # key0 = list(self.island_inhabitants.keys())[0]
        keys = list(self.island_inhabitants.keys())
        if self.random_destination:
            random.shuffle(keys)
        key0 = keys[0]
        for key in keys:
            size = len(self.island_inhabitants[key])
            imsize = len(immigrants)

            # od 0.5 do 1.5 w przybliżeniu będzie podobny współczynnik
            migration_factor = 1 if not self.random_sized_groups else random.random() + 1/2
            number_of_immigrants = int(
                size * self.migration_rate * migration_factor)
            self.island_inhabitants[key] += deepcopy(immigrants)
            immigrants = []
            for i in range(number_of_immigrants):
                immigrants.append(self.island_inhabitants[key].pop(
                    random.randint(0, size-imsize-i-1)))

        self.island_inhabitants[key0] += immigrants

        return self.island_inhabitants


# ALGO




class Islands:
    islands = {}
    interval = 10
    islands_len = 2
    max_evaluations = 10000
    problem = Rastrigin(50)

    def __init__(self, islands_len, iterval, migration_rate, random_groups, random_destination, max_evaluations, problem):
        self.islands_len = islands_len
        self.iterval = iterval
        self.max_evaluations = max_evaluations
        self.problem = problem
        self.migration = Migration(
            migration_rate, random_groups, random_destination)
        for i in range(islands_len):
            # possible to override problem later self.problem
            self.islands[str(i)] = GeneticAlgorithm(
                problem=self.problem,
                population_size=100,
                offspring_population_size=100,
                mutation=UniformMutation(0.006, 20.0),
                crossover=SBXCrossover(0.3, 19.0),
                selection=BinaryTournamentSelection(),
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=iterval)
            )
        print(self.islands)

    def run(self):
        x = 0
        new_populations = {}
        while x < self.max_evaluations:
            for key, algo in self.islands.items():
                if str(key) in new_populations:
                    algo.problem = Rastrigin(len(new_populations[key]))
                    algo.solutions = new_populations[key]
                algo.run()
                self.migration.collect_inhabitants(
                    key, algo.get_result().variables)
            x = x+self.interval
            # print("Computed populations ", new_populations)
            new_populations = self.migration.allocate_migrants()
            # print("Migrated populations ", new_populations)
        for key, algo in self.islands.items():
            result = algo.get_result()

            # print('Algorithm: {}'.format(algo.get_name()))
            # print('Solution: {}'.format(result.variables))
            # print('Fitness: {}'.format(result.objectives[0]))
            # print('Computing time: {}'.format(algo.total_computing_time))
