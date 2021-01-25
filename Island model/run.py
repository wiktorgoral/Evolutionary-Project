from model import Islands
from jmetal.problem.singleobjective.unconstrained import Rastrigin

problem = Rastrigin(50)
x = Islands(3, 10, 0.2, True, True, 1000, problem)
# porownac wyspy - porownac do algorytmu bazowego - jednej wyspy - podobna liczba osobnikow np. 50 i 5*10
# w sensownej liczbie wymiarow
# startegie emigracji/imigracji
x.run()
# You can access result of each island
result = x.islands[str(0)].get_result()
print('Algorithm: {}'.format(x.islands[str(0)].get_name()))
print('Solution: {}'.format(result.variables))
print('Fitness: {}'.format(result.objectives[0]))
