import copy
import random
from typing import List
from jmetal.core.operator import Crossover
from jmetal.core.solution import Solution, FloatSolution, BinarySolution, PermutationSolution, IntegerSolution, \
    CompositeSolution
from jmetal.util.ckecking import Check
from random import randrange
from statistics import mean 


class DifferentialEvolutionCrossover(Crossover[FloatSolution, FloatSolution]):
    """ This operator receives two parameters: the current individual and an array of three parent individuals. The
    best and rand variants depends on the third parent, according whether it represents the current of the "best"
    individual or a random_search one. The implementation of both variants are the same, due to that the parent selection is
    external to the crossover operator.
    """

    def __init__(self, CR: float, F: float, K: float):
        super(DifferentialEvolutionCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        """ Execute the differential evolution crossover ('best/1/bin' variant in jMetal).
        """
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)
        if(child == None):
          child = copy.deepcopy(parents[0])

        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value
        
        self.current_individual = child
        return [child]


    def get_number_of_parents(self) -> int:
        return 3


    def get_number_of_children(self) -> int:
        return 1


    def get_name(self) -> str:
        return 'Differential Evolution crossover'


class SelectiveMultiParentCrossover(Crossover[FloatSolution, FloatSolution]):

    def __init__(self, CR: float, F: float, K: float = 0.5, PN: int = 3):
        super(SelectiveMultiParentCrossover, self).__init__(probability=1.0)
        self.CR = CR
        self.F = F
        self.K = K
        self.PN =PN

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:

        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)
        if(child == None):
          child = copy.deepcopy(parents[0])

          
        number_of_variables = parents[0].number_of_variables
        rand = random.randint(0, number_of_variables - 1)

        parents_sample = random.sample(parents, 3)

        for i in range(number_of_variables):
            if random.random() < self.CR or i == rand:
                value1 = parents[0].variables[i] + self.F * (parents[1].variables[i] - parents[2].variables[i])
                value2 = parents[1].variables[i] + self.F * (parents[2].variables[i] - parents[0].variables[i])
                value3 = parents[2].variables[i] + self.F * (parents[0].variables[i] - parents[1].variables[i])

                value = mean([value1, value2, value3])

                if value < child.lower_bound[i]:
                    value = child.lower_bound[i]
                if value > child.upper_bound[i]:
                    value = child.upper_bound[i]
            else:
                value = child.variables[i]

            child.variables[i] = value

        self.current_individual = child
        return [child]

    def get_number_of_parents(self) -> int:
        if self.PN > 10:
            return 10
        if self.PN < 3:
            return 3
        return self.PN

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'SelectiveMultiParentCrossover'


class RandomMultiParentCrossoverWithStep(Crossover[FloatSolution, FloatSolution]):

    def __init__(self, PN: int = 3):
        super(RandomMultiParentCrossoverWithStep, self).__init__(probability=1.0)
        self.PN =PN

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)
        if(child == None):
          child = copy.deepcopy(parents[0])
          
        number_of_variables = parents[0].number_of_variables

        for i in range(number_of_variables):
            parent = random.sample(parents, 1)[0]
            selectionLen = randrange(number_of_variables)
            if i + selectionLen >  number_of_variables - 1 :
                selectionLen = number_of_variables - i
            child.variables[i:(i+selectionLen-1)] = parent.variables[i:(i+selectionLen-1)]
            i += selectionLen
        self.current_individual = child
        return [child]

    def get_number_of_parents(self) -> int:
        return self.PN

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'RandomMultiParentCrossoverWithStep'


class RandomMultiParentCrossover(Crossover[FloatSolution, FloatSolution]):

    def __init__(self, PN: int = 3):
        super(RandomMultiParentCrossover, self).__init__(probability=1.0)
        self.PN =PN

        self.current_individual: FloatSolution = None

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        if len(parents) != self.get_number_of_parents():
            raise Exception('The number of parents is not {}: {}'.format(self.get_number_of_parents(), len(parents)))

        child = copy.deepcopy(self.current_individual)
        if(child == None):
          child = copy.deepcopy(parents[0])
          
        number_of_variables = parents[0].number_of_variables

        for i in range(number_of_variables):
            parent = random.sample(parents, 1)[0]
            child.variables[i] = parent.variables[i]
            i += i
        self.current_individual = child
        return [child]

    def get_number_of_parents(self) -> int:
        return self.PN

    def get_number_of_children(self) -> int:
        return 1

    def get_name(self) -> str:
        return 'RandomMultiParentCrossover'
