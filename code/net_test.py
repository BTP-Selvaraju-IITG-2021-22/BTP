from cProfile import label
from non_cooperative import Route, Player, Network
import matplotlib.pyplot as plt
from random import random
import graphviz

baseRoute = [
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 0, 0, 0, 1],
]
baseGamma = [1, 0, 0, 1, 0, 0]
outBase =   [0, 0, 0, 0, 1, 1]

def genRandomRoute(totArrv):
    N = len(baseRoute)
    route = [[0]*N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            route[i][j] = random()*baseRoute[i][j]
        p0 = outBase[i]*random()
        norm = p0 + sum(route[i])
        route[i] = [x/norm for x in route[i]]
    
    gammas = [x*random() for x in baseGamma]
    norm = sum(gammas)
    gammas = [totArrv*x/norm for x in gammas]
    
    print(sum(gammas))
    # print(route)
    # print('\n\n')
    return Route(gammas, route)

# b/w 20 and 30
serviceRates = [random()*10 + 200 for _ in range(len(baseRoute))]

arrv = [5.0, 8.0]
nChoices = 2

players = []
for i in range(len(arrv)):
    players.append(Player([genRandomRoute(arrv[i]) for _ in range(nChoices)]))

network = Network(serviceRates, players)

# network = Network([20., 10.0], [p0, p1])
# print(network.waitTime(0))
# print(network.calcPureNashEquilibrium())
# print(network.correlatedEquilbrium())
# print(network.calcDiscreteCosts())

# mwew = network.multiplicativeWeights(eta=0.5, nSteps=50)
# for i in range(len(mwew[0])):
#     plt.plot([x[i] for x in mwew], '.-', label=f'Player {i}')
# plt.xlabel('num iterations')
# plt.ylabel('expected waiting time')
# plt.legend()
# plt.show()


g = graphviz.Digraph('G', engine="neato", filename='ex.gv',format='pdf')
g.attr(size='30')
g.attr('node', shape='circle', fixedsize='true', width='2')

ndpos = [(1,2),(2,3),(3.5,3),(2,1),(3.5,1),(4.5,2)]
scale=3
for i in range(6):
    g.node(str(i), pos=f'{ndpos[i][0]*scale},{ndpos[i][1]*scale}!')
    
g.view()

# g.node('0',pos='1,2!')
# g.node('1',pos='2,3!')
# g.node('2',pos='3,3!')
# g.node('3',pos='2,1!')
# g.node('4',pos='3,1!')
# g.node('5',pos='4,2!')