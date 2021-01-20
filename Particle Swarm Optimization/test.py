from ParticleSwarmOptimization import WithNeighborsPSO

from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.archive import CrowdingDistanceArchive



# WithNeighborsPSO can accept following parameters:
# problem: FloatProblem
# swarm_size: int
# leaders: Optional[BoundedArchive]
# termination_criterion: TerminationCriterion
# swarm_generator: Generator
# swarm_evaluator: Evaluator
#
# Those weights should be in range [0, 1.0]:
# omega: float - the weight for overall speed change. Default is 0.8.
# phi_p: float - the weight for speed change influenced by personal best. Default is 1.0.
# phi_g: float - the weight for speed change influenced by global best. Defaulrs is 1.0.
# learning_rate: float - how much speed influences the position change. Default is 1.0.
# phi_n: float - the weight for speed change influenced by neighborhood. Default is 0.3.
#
# n_neighbors: int - neighborhood size. Default is 30.
#
# hops: int - How is neighborhood determined (if neighbors' neighobrs should be also taken into account etc.)
# I would not change the number of hops from the default value 1, because it becomes computationally to expensive
# due to inefficient implementation.
#
# use_global: bool - whether to use global best while determining speed change. Default is False and it should not be
# changed because it gives poor results, but is left to be compliant with the specification given during lab exercises.
#

algorithm = WithNeighborsPSO(
    problem=Rastrigin(100),
    swarm_size=100,
    leaders=CrowdingDistanceArchive(100),
    termination_criterion=StoppingByEvaluations(100) # To get better results increase the number of iterations, I tested for 10 000-50 000.
)

algorithm.run()

solutions = algorithm.get_result()
objectives = solutions[0].objectives
variables = solutions[0].variables

print("PSO with neighbors")
print("Fitness: {}".format(objectives))
# print("Variables: {}".format(variables))
