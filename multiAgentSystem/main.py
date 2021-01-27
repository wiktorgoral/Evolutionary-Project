import time
from typing import List
import random

from jmetal.config import store
from jmetal.core.algorithm import Algorithm, R, S
from jmetal.core.operator import Crossover, Mutation
from jmetal.core.problem import Problem
from jmetal.operator import SPXCrossover, BitFlipMutation
from jmetal.problem.singleobjective.knapsack import Knapsack
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion, StoppingByEvaluations


class SolutionAgent():
    def __init__(self, solution, energy=1000):
        self.energy = energy
        self.solution = solution

    def change_energy(self, value):
        self.energy += value

    def change_solution(self, value):
        self.solution = value

    def fitness(self):
        return sum(self.solution.objectives)


class Emas(Algorithm[S, R]):

    def __init__(self,
                 number_of_islands,
                 init_island_population_size,
                 problem: Problem[S],
                 crossover: Crossover,
                 mutation: Mutation,
                 reproduction_level=3000,
                 death_level=100,
                 migration_level=3200,
                 migration_probability=0.5,
                 termination_criterion: TerminationCriterion = StoppingByEvaluations(max_evaluations=25000),
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):
        super().__init__()
        self.number_of_islands = number_of_islands
        self.init_island_population_size = init_island_population_size
        self.population_generator = population_generator
        self.reproduction_level = reproduction_level
        self.problem = problem
        self.mutation_operator = mutation
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.population_evaluator = population_evaluator
        self.crossover = crossover
        self.evaluations = 0
        self.migration_probability = migration_probability
        self.death_level = death_level
        self.migration_level = migration_level

    def get_observable_data(self) -> dict:
        self.evaluations = self.evaluations + 1
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    def create_initial_solutions(self) -> List[S]:
        solutions = []
        for island in range(self.number_of_islands):
            solutions.append([SolutionAgent(self.population_generator.new(self.problem)) for _ in
                              range(self.init_island_population_size)])
        return solutions

    def init_progress(self) -> None:
        self.evaluations = self.evaluations + 1

        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def evaluate(self, population: List[S]):
        solutions = []
        for island in population:
            island_agents = []
            new_solutions = self.population_evaluator.evaluate([agent.solution for agent in island], self.problem)
            for agent, solution in zip(island, new_solutions):
                agent.change_solution(solution)
                island_agents.append(agent)
            solutions.append(island_agents)
        return solutions

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        new_solution = []
        solution = self.evaluate(self.solutions)
        for solutions_island in solution:
            solutions_island = self.meet(solutions_island)
            solutions_island = self.reproduction(solutions_island)
            solutions_island = self.death(solutions_island)
            new_solution.append(solutions_island)

        self.solutions = self.move_to_another_island(new_solution)

    def meet(self, island):
        random.shuffle(island)
        for i in range(0, len(island) - 1, 2):
            fitness_1 = island[i].fitness()
            fitness_2 = island[i + 1].fitness()
            if fitness_1 < fitness_2:
                island[i].change_energy(100)
                island[i + 1].change_energy(-100)
            elif fitness_1 > fitness_2:
                island[i + 1].change_energy(100)
                island[i].change_energy(-100)
            else:
                if island[i].energy > island[i + 1].energy:
                    island[i].change_energy(100)
                    island[i + 1].change_energy(-100)
                else:
                    island[i + 1].change_energy(100)
                    island[i].change_energy(-100)

        return island

    def reproduction(self, island):
        to_repr = [agent for agent in island if agent.energy > self.reproduction_level]
        if len(to_repr) % 2 == 1:
            to_repr=to_repr[:-1]
        for index, agent in enumerate(to_repr):
            if index % 2 == 0:
                if agent.energy > self.reproduction_level:
                        child = SolutionAgent(self.crossover.execute([agent.solution, island[index + 1].solution])[0], int(self.reproduction_level))
                        agent.change_energy(int(-self.reproduction_level/2))
                        to_repr[index + 1].change_energy(int(-self.reproduction_level / 2))
                        self.mutate(child)
                        island.append(child)
        return island

    def death(self, island):
        for index, agent in enumerate(island):
            if agent.energy < self.death_level:
                en = island[index].energy
                island.pop(index)
                for x in island:
                    x.change_energy(int(en / len(island)))
                island[-1].change_energy(en - int(en / len(island)) * len(island))
        return island

    def move_to_another_island(self, solution):
        for island_index, island in enumerate(solution):
            for agent_index, agent in enumerate(island):
                if agent.energy > self.migration_level and random.random() > self.migration_probability or len(island) == 1:
                    traveller = island.pop(agent_index)
                    destination = random.choice(solution)
                    while (destination == island and len(solution) > 1):
                        destination = random.choice(solution)
                    destination.append(traveller)
        return solution

    def mutate(self, agent):
        agent.change_solution(self.mutation_operator.execute(agent.solution))

    def get_name(self) -> str:
        return 'Evoulutionary multi agent algorithm'

    def get_result(self) -> R:
        return self.solutions

    def update_progress(self):
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)


"""def main():
    problem = Knapsack(from_file=True, filename='resources/KnapsackInstance_500_1_19.kp')

    algorithm = Emas(
        number_of_islands=10,
        init_island_population_size=100,
        problem=problem,
        crossover=SPXCrossover(probability=0.8),
        mutation=BitFlipMutation(probability=0.1),
        termination_criterion=StoppingByEvaluations(max_evaluations=5000)
    )
    algorithm.run()

    front = algorithm.get_result()
    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    max_energy = 0
    for x in front:
        for z in x:
            print('Solution: {}'.format(z.solution.variables))
            print('Fitness: {}'.format(-z.solution.objectives[0]))
            print('Energy: {}'.format(z.energy))
            if -z.solution.objectives[0] > max_energy:
                max_energy = -z.solution.objectives[0]
                best_agent = z
    print('Computing time: {}'.format(algorithm.total_computing_time))
    print(f"Problem Maximum Capacity: {problem.capacity}")

    print("Best oof the best:")
    print('Solution: {}'.format(best_agent.solution.variables))
    print('Fitness: {}'.format(-best_agent.solution.objectives[0]))
    print('Energy: {}'.format(best_agent.energy))"""


if __name__ == "__main__":
    main()
