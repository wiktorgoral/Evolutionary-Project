from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import BoundedArchive
from jmetal.util.comparator import DominanceComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion
from copy import copy
import numpy as np
import random
from typing import TypeVar, Generic, List, Optional

S = TypeVar('S')
R = TypeVar('R')


class StandardPSO(ParticleSwarmOptimization):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 leaders: Optional[BoundedArchive],
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 swarm_generator: Generator = store.default_generator,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 omega: float = 1.0,
                 phi_p: float = 2.0,
                 phi_g: float = 2.0,
                 learning_rate: float = 1.0):
        """ This class implements a standard PSO algorithm.
        """
        super(StandardPSO, self).__init__(
            problem=problem,
            swarm_size=swarm_size)
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.leaders = leaders
        self.omega = omega # Weight of previous speed.
        self.phi_p = phi_p # How the difference between local best and particle influences the speed change.
        self.phi_g = phi_g # How the difference between global best and particle influences the speed change.
        self.learning_rate = learning_rate # How speed influences position change.

        self.dominance_comparator = DominanceComparator()

        self.speed = np.zeros((self.swarm_size, self.problem.number_of_variables), dtype=float)

    def create_initial_solutions(self) -> List[FloatSolution]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[FloatSolution]):
        return self.swarm_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            for var in range(swarm[i].number_of_variables):
                lower_bound = self.problem.lower_bound[var]
                upper_bound = self.problem.upper_bound[var]
                abs_difference = abs(upper_bound - lower_bound)
                velocity = random.uniform(-abs_difference, abs_difference)
                self.speed[i][var] = velocity

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            best_particle = copy(swarm[i].attributes['local_best'])
            best_global = self.select_global_best()

            for var in range(swarm[i].number_of_variables):
                r_p = random.random()
                r_g = random.random()
                self.speed[i][var] = self.omega * self.speed[i][var] + \
                                     self.phi_p * r_p * (best_particle.variables[var] - swarm[i].variables[var]) + \
                                     self.phi_g * r_g * (best_global.variables[var] - swarm[i].variables[var])

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            particle = swarm[i]

            for j in range(particle.number_of_variables):
                particle.variables[j] += self.learning_rate * self.speed[i][j]

                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                    self.speed[i][j] *= -1

                if particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]
                    self.speed[i][j] *= -1

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    def select_global_best(self) -> FloatSolution:
        leaders = self.leaders.solution_list

        if len(leaders) > 2:
            particles = random.sample(leaders, 2)

            if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
                best_global = copy(particles[0])
            else:
                best_global = copy(particles[1])
        else:
            best_global = copy(self.leaders.solution_list[0])

        return best_global

    def init_progress(self) -> None:
        self.evaluations = self.swarm_size
        self.leaders.compute_density_estimator()

        self.initialize_velocity(self.solutions)
        self.initialize_particle_best(self.solutions)
        self.initialize_global_best(self.solutions)
    
    def step(self):
        self.update_velocity(self.solutions)
        self.update_position(self.solutions)
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def update_progress(self) -> None:
        self.evaluations += self.swarm_size
        self.leaders.compute_density_estimator()

        observable_data = self.get_observable_data()
        observable_data['SOLUTIONS'] = self.leaders.solution_list
        self.observable.notify_all(**observable_data)

    def get_result(self) -> List[FloatSolution]:
        return self.leaders.solution_list

    def get_name(self) -> str:
        return 'StandardPSO'
    
    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass
